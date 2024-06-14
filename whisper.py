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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WhisperSTT:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize Azure OpenAI client with the necessary credentials
        self.client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            http_client=httpx.AsyncClient(http2=True),
        )
        self.sample_rate = 16000  # Sample rate for audio recording, suitable for voice

    def array_to_bytes(self, audio_array):
        # Generate a unique filename for the temporary audio file using UUID
        unique_filename = uuid.uuid4()
        tmp_audio_file_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{unique_filename}.wav")
        # Write the numpy array to a WAV file
        write(tmp_audio_file_path, self.sample_rate, audio_array.astype(np.int16))
        return tmp_audio_file_path

    def record_audio_vad(self, max_duration=120, simulation_input=None):
        # This function now accepts a simulation input array for testing purposes
        if simulation_input is not None:
            logging.info("Using simulation input")
            return np.concatenate(simulation_input)

        # Initialize Voice Activity Detector with an aggressiveness mode
        vad = webrtcvad.Vad(3)
        logging.info("Start recording...")

        recorded_frames = []  # to store audio frames
        current_silence_duration = 0  # to track duration of non-speaking segments
        max_silence_duration = 1  # maximum allowed silence duration in seconds
        num_silent_frames_to_stop = int(max_silence_duration * self.sample_rate / 160)  # compute the number of silent frames permitted before stopping
        recording_active = True

        def callback(indata, frames, time_x, status):
            nonlocal current_silence_duration, recording_active
            if status and status.input_overflow:
                logging.warning("Input overflow: %s", status)
            is_speech = sum(vad.is_speech(frame.tobytes(), self.sample_rate) for frame in np.split(indata, np.arange(160, len(indata), 160)))  # detect speech activity within the frame
            if is_speech > 0:
                current_silence_duration = 0
            else:
                current_silence_duration += 1
            recorded_frames.append(indata.copy())
            if current_silence_duration >= num_silent_frames_to_stop or len(recorded_frames) * (160 / self.sample_rate) >= max_duration:
                recording_active = False

        try:
            # Continuously record audio while 'recording_active' is True
            with sd.InputStream(callback=callback, samplerate=self.sample_rate, channels=1, dtype='int16', blocksize=160):
                while recording_active:
                    sd.sleep(100)
        except Exception as e:
            logging.error("Error occurred during recording: %s", e)

        logging.info("Recording finished.")
        return np.concatenate(recorded_frames)

    async def transcribe_audio(self, audio_array):
        # Convert the audio array to bytes and save as a temporary file
        temp_file_path = self.array_to_bytes(audio_array)
        try:
            with open(temp_file_path, "rb") as audio_file:
                try:
                    # Make a request to transcribe the audio file
                    transcript = await self.client.audio.transcriptions.create(
                        model=os.getenv("WHISPER_MODEL_NAME"),
                        file=audio_file
                    )
                    logging.info("Transcript text: %s", transcript.text)
                    return transcript.text
                except Exception as e:
                    logging.error("Error transcribing audio: %s", e)
                    return "Failed to transcribe audio"
        finally:
            # Cleanup the temporary file used for storing the audio
            self.cleanup_temp_file(temp_file_path)

    def cleanup_temp_file(self, file_path):
        os.unlink(file_path)  # Remove the file from the filesystem


# Define the asynchronous main function
async def main():
    # Create an instance of the WhisperSTT class
    whisper_stt = WhisperSTT()
    # Record audio using the record_audio_vad function
    recorded_audio = whisper_stt.record_audio_vad()
    # Transcribe recorded audio using the transcribe_audio function
    transcript = await whisper_stt.transcribe_audio(recorded_audio)
    if transcript:
        logging.info("Successfully transcribed audio.")
    else:
        logging.error("Failed to transcribe audio.")
    # Logging the transcription result
    logging.info("Transcription result: %s", transcript)



if __name__ == '__main__':
    # Run the main function using asyncio.run()
    asyncio.run(main())