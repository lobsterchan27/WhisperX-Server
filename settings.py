import os

from dotenv import load_dotenv
from multiprocessing import cpu_count

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_DIR = os.getenv("MODEL_DIRECTORY")
OUTPUT_DIR = os.getenv("OUTPUT_DIRECTORY")
VOICES_DIRECTORY = os.getenv("VOICES_DIRECTORY")
RVC_MODEL_DIR = os.getenv("weight_root")
INDEX_DIR = os.getenv("index_root")

class RVCSettings:
    def __init__(self):
        self.pth_path = os.path.join(RVC_MODEL_DIR, "FrierenFrierenv3_e150_s15000.pth")
        self.index_path = os.path.join(INDEX_DIR, "added_IVF3217_Flat_nprobe_1_FrierenFrierenv3_v2.index")
        self.I_noise_reduce: bool = True
        self.O_noise_reduce: bool = False
        self.use_pv: bool = False
        self.pitch: int = 0
        self.samplerate: int = 24000
        self.channels: int = 1

        self.threshold: int = -40
        self.index_rate: float = 1
        self.crossfade_time: float = 0.05
        self.block_time: float = .5
        self.extra_time: float = 2.5
        self.rms_mix_rate: float = 1
        self.sr_type = "sr_model"
        self.function = "vc"
        self.f0method = "rmvpe"

        self.n_cpu: int = min(cpu_count(), 4)

class VCSettings:
    def __init__(self):
        self.model_path = os.path.join(RVC_MODEL_DIR, "FrierenFrierenv3_e150_s15000.pth")
        self.f0_up_key=0
        self.input_path=None
        self.output_dir_path=None
        self._is_half="False"
        self.f0method="rmvpe"
        self.file_index=""
        self.file_index2=""
        self.index_rate=1
        self.filter_radius=3
        self.resample_sr=None
        self.rms_mix_rate=1.0
        self.protect=0.33
        self.verbose=False
          