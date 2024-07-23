# Azure OpenAI and Speech SDK Integration

[![Snyk](https://snyk.io/test/github/cnkang/rpi-voice/badge.svg)](https://snyk.io/test/github/cnkang/rpi-voice)![BuildTest](https://img.shields.io/github/actions/workflow/status/cnkang/rpi-voice/codecov.yaml)![codecov](https://img.shields.io/codecov/c/github/cnkang/rpi-voice)[![CodeFactor](https://www.codefactor.io/repository/github/cnkang/rpi-voice/badge)](https://www.codefactor.io/repository/github/cnkang/rpi-voice)[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=cnkang_rpi-voice&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=cnkang_rpi-voice)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fcnkang%2Frpi-voice.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fcnkang%2Frpi-voice?ref=badge_shield)[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fcnkang%2Frpi-voice.svg?type=shield&issueType=security)](https://app.fossa.com/projects/git%2Bgithub.com%2Fcnkang%2Frpi-voice?ref=badge_shield&issueType=security)


This project demonstrates the integration of Azure's Cognitive Services, specifically OpenAI and the Speech SDK, to create a custom voice assistant. The assistant utilizes speech-to-text and text-to-speech capabilities for conversational user interaction.

## Features

- Asynchronous communication with Azure OpenAI for robust natural language processing.
- Real-time speech recognition using WhisperSTT for precise voice processing.
- Dynamic generation of speech with Azure Cognitive Services Speech SDK.
- Secure management of environment variables using `python-dotenv`.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/cnkang/rpi-voice.git
    ```

2. **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **Create a `.env` file** in the root directory of your project.
2. Populate the `.env` file with your Azure API keys and other settings as outlined below. Replace the placeholder values with your actual credentials and preferences.
   - AZURE_OPENAI_API_KEY='your_azure_openai_api_key'
   - AZURE_API_VERSION='2024-02-15-preview'
   - AZURE_OPENAI_ENDPOINT='your_azure_openai_endpoint'
   - TTS_VOICE_NAME='your_preferred_voice'  # Default "alloy" or choose another
   - TTS_MODEL_NAME='tts-1-hd'  # Default or another TTS model
   - WHISPER_MODEL_NAME='your_model_name_for_whisper'

## Usage

**To run the script, execute the following command:**

```bash
python app.py
```

This initializes the Azure OpenAI client, records speech via microphone, transcribes it to text using WhisperSTT, sends the transcribed text to the chat model, and synthesizes the response into speech using the chosen voice from Azure Speech Services.

## Customizations

Modify the following parameters in the `.env` file to customize the voice assistant:

- **Voice Name:** Change `TTS_VOICE_NAME` to use different voices available within the Azure platform.
- **API Versions:** Adjust `AZURE_API_VERSION` to test different versions of Azure's Cognitive Services APIs.
- **Endpoints:** Modify `AZURE_OPENAI_ENDPOINT` to cater to specific geographical or organizational requirements.

## Note

Ensure that your environment supports audio recording and playback functionalities as required by the `sounddevice` and `pydub` libraries. You might need to install additional system dependencies depending on your operating system.

## Help

For troubleshooting, refer to the official documentation of the libraries and Azure services utilized in this project. Ensure your API keys and other credentials are valid and have the necessary permissions to access the services.

## License

This project is licensed under the Apache License 2.0. You may not use this file except in compliance with the License, which you can obtain at:

   http://www.apache.org/licenses/LICENSE-2.0

Software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for specific language governing permissions and limitations under the License.

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fcnkang%2Frpi-voice.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fcnkang%2Frpi-voice?ref=badge_large)

