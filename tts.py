import os
import requests
import logging
from io import BytesIO
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
# Load environment variables from .env file
load_dotenv()

# Configure logging settings
logging.basicConfig(level=logging.INFO)

# Azure API details and credentials
api_key = os.getenv("AZURE_OPENAI_API_KEY", '')
content_type = 'application/json'
tts_model = os.getenv("TTS_MODEL_NAME", 'tts-1-hd')
voice_name = os.getenv("TTS_VOICE_NAME", 'alloy')
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", '')
api_version = '2024-02-15-preview'
deployment = os.getenv("TTS_MODEL_NAME", 'tts-1-hd')

# Set headers and payload for the POST request
headers = {
    'api-key': api_key,
    'Content-Type': content_type,
}

payload = {
    "model": tts_model,
    "input": "Today is a wonderful day to build something people love!",
    "voice": voice_name
}

# Construct the URL for the Speech API request
api_url = f"{endpoint}/openai/deployments/{deployment}/audio/speech?api-version={api_version}"

# Make the POST request to the Azure API
try:
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()

    # Load the audio content in memory
    audio_stream = BytesIO(response.content)
    sound = AudioSegment.from_file(audio_stream, format="mp3")
    logging.info("Speech has been successfully synthesized.")

    # Play the synthesized speech
    logging.info("Playing the synthesized speech now...")
    play(sound)

except requests.exceptions.HTTPError as http_err:
    # Handle HTTP errors
    logging.error(f"HTTP error occurred: {http_err} - {response.status_code} {response.text}")
except Exception as e:
    # Handle other possible exceptions
    logging.error("An error occurred during the API call.", exc_info=True)
