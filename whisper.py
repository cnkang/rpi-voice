import asyncio
import logging
import os
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import httpx
import io
import tempfile
import shutil
from voicerecorder import VoiceRecorder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class WhisperSTT:
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY",None)
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT",None)
        if not self.api_key or not self.endpoint:
            raise EnvironmentError("Environment variables for Azure OpenAI Service not set")
        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=os.getenv("AZURE_API_VERSION", "2024-05-01-preview"),
            http_client=httpx.AsyncClient(http2=True)
        )

    async def transcribe_audio(self, file_path: str) -> str:
        """Transcribes the audio from a file path."""
        try:
            with open(file_path, 'rb') as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model=os.getenv("WHISPER_MODEL_NAME"), file=audio_file
                )
            return response.text
        except Exception as e:
            logging.error("Error during transcription: %s", e)
            return "Failed to transcribe audio"
            
    async def transcribe_audio_stream(self, audio_stream: io.BytesIO) -> str:
        """Transcribes audio from an io.BytesIO stream."""
        temp_file_path = save_temp_wav_file(audio_stream)
        try:
            with open(temp_file_path, 'rb') as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model=os.getenv("WHISPER_MODEL_NAME"), file=audio_file
                )
            return response.text
        except Exception as e:
            logging.error("Error during transcription: %s", e)
            return "Failed to transcribe audio"
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    

def save_temp_wav_file(audio_stream: io.BytesIO) -> str:
    """
    Save the audio stream to a temporary WAV file.

    Args:
        audio_stream (io.BytesIO): The audio stream to be saved.

    Returns:
        str: The path of the temporary WAV file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as tmp_file:
        # Copy the audio stream to the temporary file
        shutil.copyfileobj(audio_stream, tmp_file)
        # Return the path of the temporary file
        tmp_file_path = tmp_file.name  # type: ignore
        return tmp_file_path

async def main():
    whisper_stt = WhisperSTT()
    voice_recorder = VoiceRecorder()
    temp_wav_file_path = None
    try:
        audio_frames = await voice_recorder.record_audio_vad()
        # Convert the recorded audio frames to WAV bytes for better compatibility with transcription services
        wav_audio_buffer = voice_recorder.array_to_wav_bytes(audio_frames)
        temp_wav_file_path = save_temp_wav_file(wav_audio_buffer)

        # Transcribe the audio using the WAV formatted buffer
        transcription = await whisper_stt.transcribe_audio(temp_wav_file_path)
        logging.info("Transcription result: %s", transcription)
    except Exception as e:
        logging.error("An error occurred: %s", e)
    finally:
        if temp_wav_file_path and os.path.exists(temp_wav_file_path):
            os.remove(temp_wav_file_path)

if __name__ == "__main__":
    asyncio.run(main())
