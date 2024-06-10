import os
import requests
import logging
from dotenv import load_dotenv
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

# Configuration and setup
load_dotenv()
logging.basicConfig(level=logging.INFO)

class TextToSpeech:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.voice_name = os.getenv("TTS_VOICE_NAME", "alloy")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = "2024-02-15-preview"
        self.tts_model = os.getenv("TTS_MODEL_NAME", "tts-1-hd")
        self.headers = {"api-key": self.api_key, "Content-Type": "application/json"}

    def construct_request_url(self):
        # Build the URL for the Azure API request
        url = f"{self.endpoint}/openai/deployments/{self.tts_model}/audio/speech?api-version={self.api_version}"
        return url

    def synthesize_speech(self, text):
        # Create payload for POST request
        data = {"model": self.tts_model, "input": text, "voice": self.voice_name}

        # Perform POST request to Azure API
        try:
            response = requests.post(self.construct_request_url(), headers=self.headers, json=data)
            response.raise_for_status()  # Raises an HTTPError for bad requests
            logging.info("Speech synthesis request successful.")
            return BytesIO(response.content)  # Returns the audio content as a bytes object
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}", exc_info=e)
            return None
    def play_speech(self, audio_stream):
        # Play the audio stream
        if audio_stream:
            sound = AudioSegment.from_file(audio_stream, format="mp3")
            logging.info("Playback started.")
            play(sound)
        else:
            logging.error("Failed to play speech: No audio stream.")

# Usage
if __name__ == "__main__":
    tts = TextToSpeech()
    text_to_synthesize = "Today is a wonderful day to build something people love!"
    audio_stream = tts.synthesize_speech(text_to_synthesize)
    if audio_stream:
        tts.play_speech(audio_stream)
