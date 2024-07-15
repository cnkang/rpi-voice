import asyncio
import io
import logging
from typing import List

import sounddevice as sd
import webrtcvad

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class VoiceRecorder:
    """
    Class for recording voice and processing audio frames.
    """

    def __init__(self) -> None:
        """
        Initialize the VoiceRecorder object with the default audio sample rate of 16,000.
        """
        self.sample_rate: int = 16000

    def array_to_pcm_bytes(self, audio_frames: List[bytes]) -> io.BytesIO:
        """
        Convert a list of audio frames into a single PCM byte stream.

        This function takes a list of audio frames (each as bytes) and concatenates
        them into a single byte stream. This is useful for converting raw audio frames
        into a format that can be processed by audio analysis tools or saved to disk.

        Args:
            audio_frames: A list of byte strings, each representing an audio frame.

        Returns:
            A BytesIO object containing the concatenated PCM byte stream of the audio frames.

        Raises:
            Exception: If an error occurs during processing.
        """
        audio_buffer = io.BytesIO()  # Create a new BytesIO object to hold the PCM data.
        try:
            for frame in audio_frames:
                audio_buffer.write(frame)  # Write each frame to the buffer.
            audio_buffer.seek(0)  # Reset buffer position to the start for reading.
            return audio_buffer
        except Exception as e:
            logging.error("Failed to write audio to buffer: %s", str(e))  # Log any errors.
            raise  # Re-raise the exception to handle it further up the call stack.

    async def record_audio_vad(
        self,
        max_duration: float = 60.0,
        max_silence_duration: float = 1.0
    ) -> List[bytes]:
        """
        Record audio until either the maximum duration or silence duration is exceeded.

        This function records audio frames using the WebRTC VAD library and returns a list of
        the recorded frames as byte strings. The recording continues until either the maximum
        duration (in seconds) or the maximum silence duration (in seconds) is exceeded.

        Args:
            max_duration: The maximum duration of the recording in seconds. Defaults to 60.0.
            max_silence_duration: The maximum silence duration in seconds. Defaults to 1.0.

        Returns:
            A list of byte strings, each representing an audio frame.

        Raises:
            AssertionError: If an error occurs during recording.
        """
        vad = webrtcvad.Vad(mode=3)
        logging.info("Start recording...")
        recorded_frames: List[bytes] = []
        current_silence_duration: int = 0
        num_silent_frames_to_stop = int(max_silence_duration * self.sample_rate / 160)
        recording_active: bool = True

        def update_recording_status(frame_data: bytes, is_speech: bool) -> None:
            """
            Update the recording status based on the current frame's speech status.

            Args:
                frame_data: The current audio frame as a byte string.
                is_speech: A boolean indicating whether the current frame contains speech.
            """
            nonlocal current_silence_duration, recording_active
            if not is_speech:
                current_silence_duration += 1
            else:
                current_silence_duration = 0

            if current_silence_duration >= num_silent_frames_to_stop:
                recording_active = False

            recorded_frames.append(frame_data)

            if len(recorded_frames) * (160 / self.sample_rate) >= max_duration:
                recording_active = False

        def process_frame(frame_data: bytes) -> None:
            """
            Process a single audio frame.

            Args:
                frame_data: The current audio frame as a byte string.

            This function takes in an audio frame as a byte string and
            processes it by detecting whether it contains speech or not. It
            then updates the recording status based on the speech status
            of the frame.
            """
            nonlocal recording_active
            # Detect whether the frame contains speech or not
            is_speech = vad.is_speech(frame_data, self.sample_rate)
            # Update the recording status based on the speech status of the frame
            update_recording_status(frame_data, is_speech)

        try:
            with sd.InputStream(
                callback=lambda indata, frames, time, status: process_frame(indata.tobytes()),
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=160
            ):
                while recording_active:
                    await asyncio.sleep(0.1)
        except sd.PortAudioError as e:
            raise AssertionError(f'Error occurred during recording: {e}') from e

        return recorded_frames

async def main():
    """
    Main function to initiate voice recording process and handle the audio data.

    This function asynchronously records audio via the VoiceRecorder's VAD (Voice Activity Detection)
    mechanism, converts the recorded audio frames into PCM byte format, and logs the completion
    with the total byte size of the recorded audio.
    """
    voice_recorder = VoiceRecorder()  # Initialize the VoiceRecorder object

    try:
        # Record audio with voice activity detection
        audio_frames = await voice_recorder.record_audio_vad()
        # Convert the recorded audio frames to PCM bytes
        audio_buffer = voice_recorder.array_to_pcm_bytes(audio_frames)
        # Log the completion of the recording and the total size of the audio buffer
        logging.info("Audio recording completed, total bytes: %d", audio_buffer.getbuffer().nbytes)
    except Exception as e:
        # Log any exceptions that occur during the recording process
        logging.error("An error occurred: %s", str(e))

if __name__ == "__main__":
    asyncio.run(main())
