import sys
import time
import torch
import threading
import librosa
import numpy as np

import torch.nn.functional as F
import torchaudio.transforms as tat
import rvc.tools.rvc_for_realtime as rvc_for_realtime

from settings import RVCSettings, VCSettings
from rvc.configs.config import Config
from rvc.tools.torchgate import TorchGate
from rvc.infer.modules.vc.modules import VC
from multiprocessing import Queue
from collections import deque

class VCWrapper:
    def __init__(self, model_path = None):
        self.config = Config()
        self.pconfig = VCSettings()
        self.vc = VC(self.config)
        self.vc.get_vc(self.pconfig.model_path if model_path is None else model_path)
    
    def rvc_process(
        self,
        input_path,
        f0_up_key=None,
        f0_method=None,
        file_index=None,
        file_index2=None,
        index_rate=None,
        filter_radius=None,
        resample_sr=None,
        rms_mix_rate=None,
        protect=None):
        info, (tgt_sr, audio_opt) = self.vc.vc_single(
            sid=0,
            input_audio_path=input_path,
            f0_up_key=self.pconfig.f0_up_key if f0_up_key is None else f0_up_key,
            f0_file=None,
            f0_method=self.pconfig.f0method if f0_method is None else f0_method,
            file_index=self.pconfig.file_index if file_index is None else file_index,
            file_index2=self.pconfig.file_index2 if file_index2 is None else file_index2,
            index_rate=self.pconfig.index_rate if index_rate is None else index_rate,
            filter_radius=self.pconfig.filter_radius if filter_radius is None else filter_radius,
            resample_sr=self.pconfig.resample_sr if resample_sr is None else resample_sr,
            rms_mix_rate=self.pconfig.rms_mix_rate if rms_mix_rate is None else rms_mix_rate,
            protect=self.pconfig.protect if protect is None else protect)
        print(info)
        return audio_opt, tgt_sr

class RVCWrapper:
    def __init__(self):
        self.config = Config()
        self.pconfig = RVCSettings()
        self.delay_time = 0
        self.latency = .5
        self.stop = False
        self.hostapis = None

        #RVC
        self.inp_q = None
        self.opt_q = None

        #Pre+Post Processing
        self.tg = None
        self.resampler = None
        self.resampler2 = None

        #Block Size
        self.zc = None
        self.block_frame = None
        self.block_frame_16k = None
        self.crossfade_frame = None
        self.sola_buffer_frame = None
        self.sola_search_frame = None
        self.extra_frame = None

        #Buffers
        self.rms_buffer = None
        self.input_wav = None
        self.input_wav_res = None
        self.input_wav_denoise = None
        self.nr_buffer = None
        self.fade_in_window = None
        self.fade_out_window = None
        self.output_buffer = None
        self.sola_buffer = None

        self.calculate_delay_time()


    def calculate_delay_time(self):
        crossfade_time_min = min(self.pconfig.crossfade_time, 0.04)
        noise_reduce_adjustment = (1 if self.pconfig.I_noise_reduce else -1) * crossfade_time_min
        self.delay_time = self.latency + self.pconfig.block_time + self.pconfig.crossfade_time + 0.01 + crossfade_time_min + noise_reduce_adjustment

    def initialize_rvc_realtime(self):
        #RVC
        self.inp_q = Queue()
        self.opt_q = Queue()
        self.rvc = rvc_for_realtime.RVC(
            key=self.pconfig.pitch,
            pth_path=self.pconfig.pth_path,
            index_path=self.pconfig.index_path,
            index_rate=self.pconfig.index_rate,
            n_cpu=self.pconfig.n_cpu,
            inp_q=self.inp_q,
            opt_q=self.opt_q,
            config=self.config,
            last_rvc=self.rvc if hasattr(self, "rvc") else None
        )
        # self.pconfig.samplerate = self.rvc.tgt_sr

        #block size
        self.zc = self.pconfig.samplerate // 100
        self.block_frame = (int(np.round(self.pconfig.block_time * self.pconfig.samplerate / self.zc)) * self.zc)
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (int(np.round(self.pconfig.crossfade_time * self.pconfig.samplerate / self.zc)) * self.zc)
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (int(np.round(self.pconfig.extra_time * self.pconfig.samplerate / self.zc)) * self.zc)

        #Buffers
        self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
        self.input_wav: torch.Tensor = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_res: torch.Tensor = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
        self.sola_buffer: torch.Tensor = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
        self.output_buffer: torch.Tensor = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window

        #Threshold
        self.tg = TorchGate(
            sr=self.pconfig.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.config.device)

        #Resampler
        self.resampler = tat.Resample(
            orig_freq=self.pconfig.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        if self.rvc.tgt_sr != self.pconfig.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.pconfig.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        else:
            self.resampler2 = None

    def rvc_process(self, indata, outdata):

        start_time = time.perf_counter()
        if self.pconfig.threshold > -60:
            self._apply_threshold(indata)

        self._shift_and_append_audio_buffers(indata)

        if self.pconfig.I_noise_reduce:
            self._input_noise_reduction()
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = (self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[160:])
        # Infer
        if self.pconfig.function == "vc":
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.pconfig.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
        elif self.pconfig.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()

        if self.pconfig.O_noise_reduce and self.pconfig.function == "vc":
            infer_wav = self._output_noise_reduction(infer_wav)
        
        if self.pconfig.rms_mix_rate < 1 and self.pconfig.function == "vc":
            self.infer_wav = self.pconfig.volume_envelope_mixing(infer_wav)
            
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        infer_wav = self._sola_algorithmn(infer_wav)

        outdata[:] = (
            infer_wav[: self.block_frame]
            .t()
            .cpu()
            .numpy()
        )
        total_time = time.perf_counter() - start_time
        self._printt("Infer time: %.2f", total_time)

    def _apply_threshold(self, indata):
        indata = np.append(self.rms_buffer, indata)
        rms = librosa.feature.rms(
            y=indata, frame_length=4 * self.zc, hop_length=self.zc
        )[:, 2:]
        self.rms_buffer[:] = indata[-4 * self.zc :]
        indata = indata[2 * self.zc - self.zc // 2 :]
        db_threshold = (
            librosa.amplitude_to_db(rms, ref=1.0)[0] < self.pconfig.threshold
        )
        for i in range(db_threshold.shape[0]):
            if db_threshold[i]:
                indata[i * self.zc : (i + 1) * self.zc] = 0
        indata = indata[self.zc // 2 :]
        return indata
    
    def _shift_and_append_audio_buffers(self, indata):
        self.input_wav[: -self.block_frame] = self.input_wav[
            self.block_frame :
        ].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
            self.config.device
        )
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
    
    def _input_noise_reduction(self):
        self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
            self.block_frame :
        ].clone()
        input_wav = input_wav[-self.sola_buffer_frame - self.block_frame :]
        input_wav = self.tg(
            input_wav.unsqueeze(0), input_wav.unsqueeze(0)
        ).squeeze(0)
        input_wav[: self.sola_buffer_frame] *= self.fade_in_window
        input_wav[: self.sola_buffer_frame] += (
            self.nr_buffer * self.fade_out_window
        )
        self.input_wav_denoise[-self.block_frame :] = input_wav[
            : self.block_frame
        ]
        self.nr_buffer[:] = input_wav[self.block_frame :]
        self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
            self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
        )[160:]

    def _output_noise_reduction(self, infer_wav):
        self.output_buffer[: -self.block_frame] = self.output_buffer[
            self.block_frame :
        ].clone()
        self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
        infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
        return infer_wav
    
    def _volume_envelope_mixing(self, infer_wav):
        if self.pconfig.I_noise_reduce:
            input_wav = self.input_wav_denoise[self.extra_frame :]
        else:
            input_wav = input_wav[self.extra_frame :]
        rms1 = librosa.feature.rms(
            y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
            frame_length=4 * self.zc,
            hop_length=self.zc,
        )
        rms1 = torch.from_numpy(rms1).to(self.config.device)
        rms1 = F.interpolate(
            rms1.unsqueeze(0),
            size=infer_wav.shape[0] + 1,
            mode="linear",
            align_corners=True,
        )[0, 0, :-1]
        rms2 = librosa.feature.rms(
            y=infer_wav[:].cpu().numpy(),
            frame_length=4 * self.zc,
            hop_length=self.zc,
        )
        rms2 = torch.from_numpy(rms2).to(self.config.device)
        rms2 = F.interpolate(
            rms2.unsqueeze(0),
            size=infer_wav.shape[0] + 1,
            mode="linear",
            align_corners=True,
        )[0, 0, :-1]
        rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
        infer_wav *= torch.pow(rms1 / rms2, torch.tensor(1 - self.pconfig.rms_mix_rate))
        return infer_wav
    
    def _sola_algorithmn(self, infer_wav):
        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]
        if "privateuseone" in str(self.config.device) or not self.pconfig.use_pv:
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
        else:
            infer_wav[: self.sola_buffer_frame] = self._phase_vocoder(
                self.sola_buffer,
                infer_wav[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        return infer_wav
    
    def _phase_vocoder(a, b, fade_out, fade_in):
        window = torch.sqrt(fade_out * fade_in)
        fa = torch.fft.rfft(a * window)
        fb = torch.fft.rfft(b * window)
        absab = torch.abs(fa) + torch.abs(fb)
        n = a.shape[0]
        if n % 2 == 0:
            absab[1:-1] *= 2
        else:
            absab[1:] *= 2
        phia = torch.angle(fa)
        phib = torch.angle(fb)
        deltaphase = phib - phia
        deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
        w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
        t = torch.arange(n).unsqueeze(-1).to(a) / n
        result = (
            a * (fade_out**2)
            + b * (fade_in**2)
            + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
        )
        return result
    
    def _printt(self, strr, *args):
        if len(args) == 0:
            print(strr)
        else:
            print(strr % args)

    def audio_stream(self, audio_queue: deque, ws_out, event: threading.Event):
        # Initialize buffers
        in_buffer = np.zeros(self.block_frame, dtype=np.float32)
        out_buffer = np.zeros(self.block_frame, dtype=np.float32)

        while not self.stop:
            event.wait(timeout=60.0)  # Wait for up to 1 second
            if self.stop:
                break  # If stop was called, exit the loop
            if audio_queue:
                # Extract data from the queue
                block = np.array([audio_queue.popleft() for _ in range(min(len(audio_queue), self.block_frame))], dtype=np.float32)

                # Copy the data to in_buffer
                in_buffer[:len(block)] = block

                # Pad the remaining elements with zeros if needed
                if len(block) < self.block_frame:
                    in_buffer[len(block):] = 0.0

                # Process in_buffer and send to ws_out
                self.rvc_process(in_buffer, out_buffer)
                ws_out.append(out_buffer.copy())

                if not audio_queue:
                    event.clear()
                    
    def stop_stream(self):
        self.stop = True