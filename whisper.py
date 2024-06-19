import asyncio
import tempfile
import uuid
import os
import logging
from typing import Optional, List

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from openai import AsyncAzureOpenAI
import httpx
import webrtcvad
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WhisperSTT:
    def __init__(self) -> None:
        """Initializes the WhisperSTT by setting up environment variables and AzureOpenAI client for audio transcription."""
        load_dotenv()
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        assert azure_endpoint, "Environment variable 'AZURE_OPENAI_ENDPOINT' cannot be empty"
        assert api_key, "Environment variable 'AZURE_OPENAI_API_KEY' cannot be empty"

        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=os.getenv("AZURE_API_VERSION", "2024-05-01-preview"),
            http_client=httpx.AsyncClient(http2=True),
        )
        self.sample_rate: int = 16000
        self.stop_event: asyncio.Event = asyncio.Event()

    def array_to_bytes(self, audio_array: np.ndarray) -> str:
        """
        Converts a numpy audio array to a WAV file stored temporarily.

        Args:
            audio_array: Audio array to convert.

        Returns:
            Path to the temporary WAV file.

        Raises:
            Exception: If the audio array cannot be written to a file.
        """
        unique_filename = str(uuid.uuid4())
        tmp_audio_file_path = os.path.join(
            tempfile.gettempdir(), f"temp_audio_{unique_filename}.wav"
        )
        try:
            write(tmp_audio_file_path, self.sample_rate, audio_array.astype(np.int16))
            return tmp_audio_file_path
        except Exception as e:
            logging.error("Failed to write audio to file: %s", str(e))  # Extracting the error message only
            raise


    async def record_audio_vad(
        self,
        max_duration: float = 120.0,
        max_silence_duration: float = 1.0,
        simulation_input: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Record audio using voice activity detection (VAD).

        Args:
            max_duration: Maximum duration of the recorded audio in seconds.
            max_silence_duration: Maximum duration of silence in seconds after which recording is stopped.
            simulation_input: Simulated audio input for testing purposes.

        Returns:
            The recorded audio as a numpy array.

        Raises:
            AssertionError: If sample rate is not greater than 0.
            ValueError: If simulation input is not a list of numpy arrays of dtype int16.
            AssertionError: If error occurs during recording with sounddevice.
        """
        assert self.sample_rate > 0, "Sample rate must be greater than 0"

        vad = webrtcvad.Vad(mode=3)  # Mode 3 is the most aggressive.
        logging.info("Start recording...")
        recorded_frames: List[np.ndarray] = []
        current_silence_duration: int = 0
        num_silent_frames_to_stop = int(max_silence_duration * self.sample_rate / 160)
        recording_active: bool = True

        def update_recording_status(frame_data,is_speech: bool) -> None:
            """
            Update the silence duration and recording status based on whether the frame contains speech or not.

            Args:
                is_speech (bool): Whether the frame contains speech or not.

            Returns:
                None
            """
            nonlocal current_silence_duration, recording_active
            if not is_speech:
                current_silence_duration += 1
                logging.debug("Current silence duration: %s", current_silence_duration)
            else:
                current_silence_duration = 0

            if current_silence_duration >= num_silent_frames_to_stop:
                logging.debug("Stopping due to maximum silence threshold exceeded.")
                recording_active = False

            recorded_frames.append(frame_data.copy())

            if len(recorded_frames) * (160 / self.sample_rate) >= max_duration:
                recording_active = False

        def process_frame(frame_data: np.ndarray) -> None:
            """
            Process a single audio frame and update the silence duration and recording status.

            Args:
                frame_data (np.ndarray): The audio frame data to be processed.

            Returns:
                None

            Raises:
                AssertionError: If the frame data dtype is not int16.
            """
            nonlocal recording_active
            assert frame_data.dtype == np.int16, "Frame data must be of dtype int16"
            is_speech = vad.is_speech(frame_data.tobytes(), self.sample_rate)
            logging.debug(
                "Frame speech status: %s, Frame count: %s",
                is_speech,
                len(recorded_frames),
            )
            update_recording_status(frame_data,is_speech)

        if simulation_input is not None:
            assert all(
                isinstance(frame, np.ndarray) and frame.dtype == np.int16
                for frame in simulation_input
            ), "Simulation input must be a list of numpy arrays of dtype int16"
            logging.info("Using simulation input")
            for frame in simulation_input:
                process_frame(frame)
                if not recording_active:
                    break
        else:
            try:
                with sd.InputStream(
                    callback=lambda indata, frames, time_x, status: process_frame(indata),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=160,
                ):
                    while recording_active:
                        await asyncio.sleep(0.1)  # Adjusted for asyncio compatibility
            except sd.PortAudioError as e:
                raise AssertionError(f'Error occurred during recording: {e}') from e

        if recorded_frames:
            final_audio = np.concatenate(recorded_frames)
        else:
            # Return an empty array if no frames have been recorded
            final_audio = np.array([], dtype=np.int16)
        assert final_audio.dtype == np.int16, "Final audio must be of dtype int16"
        logging.info(
            "Recording finished. Total duration: %s seconds",
            len(final_audio) / self.sample_rate,
        )
        return final_audio

    async def transcribe_audio(self, audio_array: np.ndarray) -> str:
        """
        Transcribes the audio.

        Args:
            audio_array: The audio array to be transcribed.

        Returns:
            The transcribed text.
        """
        # Convert the audio array to a temporary WAV file
        temp_file_path = self.array_to_bytes(audio_array)

        try:
            # Open the temporary WAV file in binary mode
            with open(temp_file_path, "rb") as audio_file:
                # Send the audio file to the Azure OpenAI API for transcription
                response = await self.client.audio.transcriptions.create(
                    model=os.getenv("WHISPER_MODEL_NAME"), file=audio_file
                )
                # Extract the transcribed text from the response
                transcript = response.text
                return transcript
        except asyncio.CancelledError:
            # If the recording is cancelled, return a message indicating this
            logging.warning("The recording was cancelled.")
            return "The recording was cancelled"
        except Exception as e:
            # If an error occurs during transcription, return a message indicating this
            logging.error("Error transcribing audio: %s", e)
            return "Failed to transcribe audio"
        finally:
            # Clean up the temporary WAV file
            self.cleanup_temp_file(temp_file_path)

    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Deletes the temporary file at the specified file path.
        """
        try:
            os.remove(file_path)
        except FileNotFoundError:
            logging.error("Temp file not found: %s", file_path)  # Updated error message for clarity
        except OSError:
            logging.error("Failed to delete temp file: %s", file_path)  # Updated error message for clarity

async def main():
    """
    Asynchronously records audio using voice activity detection (VAD), transcribes the audio using WhisperSTT,
    and logs the transcription result. If any error occurs during the process, it is logged.

    Returns:
        None

    Raises:
        Exception: If an error occurs during the recording or transcription process.
    """
    whisper_stt = WhisperSTT()
    try:
        audio_data = await whisper_stt.record_audio_vad()
        transcription = await whisper_stt.transcribe_audio(audio_data)
        logging.info("Transcription result: %s", transcription)
    except Exception as e:
        logging.error("An error occurred: %s", e)

if __name__ == "__main__":
    asyncio.run(main())
