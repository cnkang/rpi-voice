# Azure OpenAI and Speech SDK Integration

This project demonstrates how to use Azure's Cognitive Services, specifically the OpenAI and Speech SDK, to create a custom voice assistant. The assistant leverages speech-to-text and text-to-speech capabilities to interact with users in a conversational manner.

## Features

- Asynchronous communication with Azure OpenAI for natural language processing.
- Real-time speech recognition with WhisperSTT for accurate voice processing.
- Dynamic text-to-speech generation using Azure Cognitive Services Speech SDK.
- Environment variable management using `python-dotenv`.

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```

2. **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **Create a `.env` file** in the root directory of your project.
2. Populate the `.env` file with your Azure API keys and other configurations as shown below. Replace placeholder values with your actual credentials and preferences.
   - AZURE_OPENAI_API_KEY='your_azure_openai_api_key'
   - AZURE_API_VERSION='2024-02-15-preview' 
   - AZURE_OPENAI_ENDPOINT='your_azure_openai_endpoint'
   - TTS_VOICE_NAME='your_preferred_voice'  # Default "alloy" or choose your own
   - TTS_MODEL_NAME='tts-1-hd'  # Default value or your model of choice for TTS
   - WHISPER_MODEL_NAME='your_model_name_for_whisper'


## Usage

**Run the script using the following command:**

```bash
python app.py
```
This will initialize the Azure OpenAI client, record speech via microphone, transcribe the speech to text using WhisperSTT, send the transcribed text to the chat model, and synthesize the response into speech using the specified voice from Azure Speech Services.

## Customizations
You can customize the following functionalities by modifying the parameters in the .env file:

- **Voice Name:** Change TTS_VOICE_NAME to use different voices available in the Azure platform.
- **API Versions:** Modify AZURE_API_VERSION to test different versions of Azure's Cognitive Services APIs.
- **Endpoints:** Adjust AZURE_OPENAI_ENDPOINT based on your geographical or organizational configurations.

## Note
Ensure that your environment supports audio recording and playback functionalities as required by the sounddevice and pydub libraries. You might need to install additional system dependencies depending on your operating system.

## Help
For troubleshooting common issues, refer to the official documentation of the libraries and Azure services used in this project. Ensure that your API keys and other credentials are valid and have the necessary permissions for accessing the services.