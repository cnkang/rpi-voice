from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set parameters for Azure OpenAI Service Whisper
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = os.getenv("AZURE_API_VERSION")
)

audio_file_path = "./num.m4a"

# Read and transcribe the audio file
with open(audio_file_path, "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model = os.getenv("WHISPER_MODEL_NAME"),
        file=audio_file
    )

# Print transcription results
    print(transcript)
