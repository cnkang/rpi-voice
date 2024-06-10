import os
import requests
import logging
import io
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Prepare the headers and payload for the POST request
headers = {
    'api-key': os.getenv("AZURE_OPENAI_API_KEY", ''),
    'Content-Type': 'application/json',
}

data = {
    "model": os.getenv("TTS_MODEL_NAME", 'tts-1-hd'),  # Use default value if env var is not set
    "input": "Today is a wonderful day to build something people love!",
    "voice": os.getenv("TTS_VOICE_NAME", 'alloy')  # Use default voice if env var is not set
}

# Construct the URL for the request
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", '')
api_version = '2024-02-15-preview'  # Update as needed
deployment_name = os.getenv("TTS_MODEL_NAME", 'tts-1-hd')  # Update with your actual deployment name
url = f"{endpoint}/openai/deployments/{deployment_name}/audio/speech?api-version={api_version}"

# Perform the POST request
try:
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        # Convert audio bytes to AudioSegment
        audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
        logging.info("Speech synthesized successfully.")
        # Play the audio file
        logging.info("Playing the speech...")
        play(audio)

    else:
        logging.error(f"Failed to synthesize speech: {response.status_code} {response.text}")

except Exception as e:
    logging.error("Error during API call", exc_info=True)
