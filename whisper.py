import httpx
import asyncio
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import uuid
import os
import logging
import webrtcvad
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WhisperSTT:
    def __init__(self):
        """Initializes the WhisperSTT by setting up environment variables and AzureOpenAI client for audio transcription."""
        load_dotenv()
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        assert (
            azure_endpoint
        ), "Environment variable 'AZURE_OPENAI_ENDPOINT' cannot be empty"
        assert api_key, "Environment variable 'AZURE_OPENAI_API_KEY' cannot be empty"

        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=os.getenv("AZURE_API_VERSION", "2024-05-01-preview"),
            http_client=httpx.AsyncClient(http2=True),
        )
        self.sample_rate = 16000
        self.stop_event = asyncio.Event()

    def array_to_bytes(self, audio_array):
        """Converts a numpy audio array to a WAV file stored temporarily."""
        unique_filename = uuid.uuid4()
        tmp_audio_file_path = os.path.join(
            tempfile.gettempdir(), f"temp_audio_{unique_filename}.wav"
        )
        try:
            write(tmp_audio_file_path, self.sample_rate, audio_array.astype(np.int16))
            return tmp_audio_file_path
        except Exception as e:
            logging.error(f"Failed to write audio to file: {e}")
            raise

    async def record_audio_vad(self, max_duration=120, max_silence_duration=1, simulation_input=None):
        vad = webrtcvad.Vad(3)  # Mode 3 is the most aggressive.
        logging.info("Start recording...")
        recorded_frames = []
        current_silence_duration = 0
        num_silent_frames_to_stop = int(max_silence_duration * self.sample_rate / 160)
        recording_active = True

        def process_frame(frame_data):
            nonlocal current_silence_duration, recording_active
            is_speech = vad.is_speech(frame_data.tobytes(), self.sample_rate)
            logging.debug(f"Frame speech status: {is_speech}, Frame count: {len(recorded_frames)}")
            
            if not is_speech:
                current_silence_duration += 1
                logging.debug(f"Current silence duration: {current_silence_duration}")
            else:
                current_silence_duration = 0
            
            if current_silence_duration >= num_silent_frames_to_stop:
                logging.debug("Stopping due to maximum silence threshold exceeded.")
                recording_active = False
            
            recorded_frames.append(frame_data.copy())
            
            if len(recorded_frames) * (160 / self.sample_rate) >= max_duration:
                recording_active = False

        # If simulation input is provided, process it directly.
        if simulation_input is not None:
            logging.info("Using simulation input")
            for frame in simulation_input:
                process_frame(frame)
                if not recording_active:
                    break
        else:
            # Use sounddevice to capture real-time audio if no simulation is provided.
            try:
                with sd.InputStream(callback=lambda indata, frames, time_x, status: process_frame(indata),
                                    samplerate=self.sample_rate, channels=1, dtype='int16', blocksize=160):
                    while recording_active:
                        await asyncio.sleep(0.1)  # Adjusted for asyncio compatibility
            except Exception as e:
                logging.error("Error occurred during recording: %s", e)

        final_audio = np.concatenate(recorded_frames)
        logging.info(f"Recording finished. Total duration: {len(final_audio) / self.sample_rate}s")
        return final_audio


    async def transcribe_audio(self, audio_array):
        """Transcribes the audio."""
        temp_file_path = self.array_to_bytes(audio_array)
        try:
            with open(temp_file_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model=os.getenv("WHISPER_MODEL_NAME"), file=audio_file
                )
                return transcript.text
        except Exception as e:
            logging.error("Error transcribing audio: %s", e)
            return "Failed to transcribe audio"
        finally:
            self.cleanup_temp_file(temp_file_path)

    def cleanup_temp_file(self, file_path):
        try:
            os.unlink(file_path)
        except Exception as e:
            logging.error(f"Failed to delete temp file {file_path}: {e}")


async def main():
    whisper_stt = WhisperSTT()
    try:
        audio_data = await whisper_stt.record_audio_vad()
        transcription = await whisper_stt.transcribe_audio(audio_data)
        logging.info(f"Transcription result: {transcription}")
    except asyncio.CancelledError:
        logging.warning("The recording was cancelled.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
