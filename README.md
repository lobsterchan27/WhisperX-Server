# WhisperX Server

WhisperX Server is a powerful backend application that provides advanced audio and video processing capabilities, including transcription, text-to-speech conversion, and voice conversion. It's designed to work in conjunction with the Banana Client project.

## Features

- Audio transcription using WhisperX
- Text-to-speech synthesis with multiple voices and backends
- Voice conversion using RVC (Retrieval-based Voice Conversion)
- YouTube video downloading and processing
- Subtitle generation
- Storyboard creation from videos
- API endpoints for integration with frontend applications

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/whisperx-server.git
   cd whisperx-server
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv my-venv
   source my-venv/bin/activate  # On Windows, use: my-venv\Scripts\activate
   ```

3. Install PyTorch following the instructions at https://pytorch.org/

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: Some dependencies may need to be installed manually or may require additional setup.

5. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   HF_TOKEN=your_huggingface_token
   MODEL_DIRECTORY=path/to/model/directory
   OUTPUT_DIRECTORY=path/to/output/directory
   VOICES_DIRECTORY=path/to/voices/directory
   API_TOKEN=your_api_token
   ```

## Usage

1. Start the server:
   ```
   python main.py
   ```

2. The server will be available at `http://localhost:8127` (or the port specified in your config.ini file).

3. Use the provided API endpoints to interact with the server. For example:
   - Transcribe a YouTube video: `POST /api/transcribe/url`
   - Generate text-to-speech: `POST /api/text2speech`
   - Process audio with voice conversion: `POST /api/rvc`

## Configuration

- Network settings, model parameters, and other options can be configured in the `config.ini` file.
- Additional settings are available in the `settings.py` file.

## TODO

- [ ] Implement chunk determination for improved transcription accuracy
- [ ] Add support for outputting video in different languages
- [ ] Implement unit tests and integration tests
- [ ] Refactor project structure and codebase:
  - [ ] Organize modules and files into logical directories
  - [ ] Standardize naming conventions across the project
  - [ ] Improve code documentation and comments
  - [ ] Reduce code duplication and increase reusability
  - [ ] Optimize import statements and remove unused imports
  - [ ] Implement proper error handling and logging
  - [ ] Create a consistent API structure across endpoints
- [ ] Update and maintain requirements.txt for easier dependency management
- [ ] Improve configuration management (consider using a proper config management library)
- [ ] Enhance security measures, especially for API endpoints
- [ ] Optimize performance for large-scale audio and video processing
- [ ] Improve integration and documentation for use with Banana Client project

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact and Support

I'm still learning and improving my skills with this project. If you have any questions, suggestions, or if you'd like to contribute, please don't hesitate to reach out:

- **GitHub Issues**: For bug reports, feature requests, or general questions, please [open an issue](https://github.com/your-username/whisperx-server/issues) on this repository.
- **Discussions**: For broader conversations about the project, use the [GitHub Discussions](https://github.com/your-username/whisperx-server/discussions) feature in this repository.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperx) for improved transcription capabilities
- [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for voice conversion
- [YT-DLP](https://github.com/yt-dlp/yt-dlp) for YouTube video downloading
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [PyTorch](https://pytorch.org/) for deep learning capabilities

## Note

This project is designed to work in conjunction with the Banana Client project. Make sure to set up and configure both projects for full functionality.
