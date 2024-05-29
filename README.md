# Azure OpenAI and Speech SDK Integration

This project demonstrates how to use Azure's Cognitive Services, specifically the OpenAI and Speech SDK, to create a chatbot that responds with synthesized voice.

## Features

- Asynchronous communication with Azure OpenAI.
- Text-to-speech capabilities using Azure Cognitive Services Speech SDK.
- Environment variable management using `python-dotenv`.

## Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. Create a `.env` file in the root directory of your project.
2. Populate the `.env` file with your Azure API keys and other configurations:

    ```plaintext
    AZURE_OPENAI_API_KEY='your_azure_openai_api_key'
    AZURE_API_VERSION='api_version'
    AZURE_OPENAI_ENDPOINT='your_azure_openai_endpoint'
    AZURE_SPEECH_KEY='your_azure_speech_key'
    AZURE_SPEECH_REGION='your_azure_region'
    VOICE_NAME='zh-CN-XiaoxiaoMultilingualNeural'
    ```

## Usage

Run the script using the following command:

```bash
python app.py
```
This will initialize the Azure OpenAI client, send a message to the chat model, and synthesize the response using the specified voice from Azure Speech Services.