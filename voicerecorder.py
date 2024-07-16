import asyncio
import io
import time
import logging
from typing import List
import wave

import sounddevice as sd
import webrtcvad

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VoiceRecorder:
    """
    VoiceRecorder is a class that provides methods for recording audio using the SoundDevice library and 
    converting the recorded audio frames to PCM bytes for processing.
    """
    
    def __init__(self) -> None:
        """
        Initializes the VoiceRecorder class and sets the sample rate to 16kHz.
        """
        self.sample_rate: int = 16000


    def record_audio(self, callback: Optional[asyncio.Task] = None) -> None:
        """
        Records audio from the default device using the SoundDevice library and stores the audio frames in a list.

        Args:
            callback (Optional[asyncio.Task]): An optional callback function that can be used for progress
                tracking or other purposes.

        Returns:
            None
        """
        audio_frames = []  # List to store the recorded audio frames
        vad = webrtcvad.Vad()  # Create a WebRTC Voice Activity Detector (VAD) instance
        sample_width = sd.default.samplerate * sd.default.channels // 1000  # Calculate the sample width
        
        try:
            with sd.RawInputStream(
                device=sd.default.device["device_index"],  # Use the default device
                dtype="int16",  # Set the data type to 16-bit signed integer
                channels=sd.default.channels,  # Use the default number of channels
                samplerate=self.sample_rate,  # Set the sample rate
                callback_streaming_callback=callback,  # Optional callback function
                blocksize=int(self.sample_rate * 0.02)  # Read audio data in chunks of 20ms
            ) as stream:
                while True:
                    # Read audio data in chunks of 20ms
                    audio_data = stream.read(int(self.sample_rate * 0.02),.exception=True)
                    
                    # Apply VAD to the audio data
                    if vad.is_speech(audio_data.data, sample_rate=self.sample_rate):
                        # Append speech frames to the list
                        audio_frames.append(audio_data)
                    else:
                        # Discard non-speech frames
                        continue
        except Exception as e:
            logging.error(f"Error during audio recording: {e}")

    def array_to_pcm_bytes(self, audio_frames: List[bytes]) -> io.BytesIO:
        """
        Converts a list of audio frames to a buffer in PCM format.

        Args:
            audio_frames (List[bytes]): List of audio frames.

        Returns:
            io.BytesIO: Buffer containing the audio in PCM format.

        Raises:
            Exception: If there is an error writing the audio to the buffer.
        """
        # Create a buffer to store the audio
        audio_buffer = io.BytesIO()

        try:
            # Write each audio frame to the buffer
            for frame in audio_frames:
                audio_buffer.write(frame)

            # Reset the buffer's position to the beginning
            audio_buffer.seek(0)

            # Return the buffer
            return audio_buffer

        except Exception as e:
            # Log the error and raise an exception
            logging.error("Failed to write audio to buffer: %s", str(e))
            raise Exception("Failed to write audio to buffer") from e

    def array_to_wav_bytes(self, audio_frames: List[bytes]) -> io.BytesIO:
        """
        Converts a list of audio frames to a buffer in WAV format.

        Args:
            audio_frames (List[bytes]): List of audio frames.

        Returns:
            io.BytesIO: Buffer containing the audio in WAV format.
        """
        # Create a buffer to store the audio
        wav_buffer = io.BytesIO()

        # Open the buffer in write binary mode as a WAV file
        with wave.open(wav_buffer, "wb") as wav_file:
            # Set the WAV file properties
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)  # Sample rate

            # Write each audio frame to the WAV file
            for frame in audio_frames:
                wav_file.writeframes(frame)

        # Reset the buffer's position to the beginning
        wav_buffer.seek(0)

        # Return the buffer containing the audio in WAV format
        return wav_buffer

    async def record_audio_vad(self, max_duration: float = 59.5, max_silence_duration: float = 1.0) -> List[bytes]:
        """
        Records audio using VAD (Voice Activity Detection) and returns the recorded audio frames.

        Args:
            max_duration (float, optional): Maximum duration of the recording in seconds. Defaults to 59.5.
            max_silence_duration (float, optional): Maximum duration of silence to stop the recording in seconds. Defaults to 1.0.

        Returns:
            List[bytes]: List of audio frames recorded.
        """
        # Create an instance of webrtcvad to detect voice activity
        vad = webrtcvad.Vad(mode=3)

        # Log the start of the recording
        logging.info("Start recording...")

        # Initialize variables to store the recorded frames and silence duration
        recorded_frames: List[bytes] = []
        current_silence_duration: int = 0

        # Calculate the number of silent frames to stop the recording based on the silence duration and sample rate
        num_silent_frames_to_stop = int(max_silence_duration * self.sample_rate / 160)

        # Flag to indicate if the recording is active
        recording_active: bool = True

        def update_recording_status(frame_data: bytes, is_speech: bool) -> None:
            """
            Update the recording status based on the speech detected.

            Args:
                frame_data (bytes): Audio frame data.
                is_speech (bool): Flag indicating if speech is detected.
            """
            nonlocal current_silence_duration, recording_active

            # If no speech is detected, increment the silence duration
            if not is_speech:
                current_silence_duration += 1
            else:
                # Reset the silence duration if speech is detected
                current_silence_duration = 0

            # Check if the maximum silence duration or maximum duration is reached to stop the recording
            if current_silence_duration >= num_silent_frames_to_stop or len(recorded_frames) * (160 / self.sample_rate) >= max_duration:
                recording_active = False

            # Append the frame data to the recorded frames
            recorded_frames.append(frame_data)

        def process_frame(frame_data: bytes) -> None:
            """
            Process an audio frame and update the recording status.

            Args:
                frame_data (bytes): Audio frame data.
            """
            nonlocal recording_active

            # Check if the speech is detected in the frame
            is_speech = vad.is_speech(frame_data, self.sample_rate)

            # Update the recording status based on the speech detected
            update_recording_status(frame_data, is_speech)

        try:
            # Start the audio input stream with the specified parameters
            with sd.InputStream(callback=lambda indata, frames, time, status: process_frame(indata.tobytes()),
                                samplerate=self.sample_rate, channels=1, dtype='int16', blocksize=160):
                # Continuously sleep for 0.1 seconds while the recording is active
                while recording_active:
                    await asyncio.sleep(0.1)
        except sd.PortAudioError as e:
            # Log the error and raise an exception if there is an error during recording
            logging.error("Recording error: %s", e)
            raise RuntimeError("Failed during recording") from e

        # Return the recorded frames
        return recorded_frames

async def main():
    """
    Main function that records audio using voice activity detection (VAD) and logs the duration and size of the recorded audio.
    """
    voice_recorder = VoiceRecorder()
    try:
        # Start recording audio
        start_time = time.time()
        audio_frames = await voice_recorder.record_audio_vad()
        end_time = time.time()

        # Calculate the duration of the recording
        duration = end_time - start_time
        logging.debug("Recording duration: %.2f seconds", duration)

        # Convert the recorded audio frames to a buffer
        audio_buffer = voice_recorder.array_to_pcm_bytes(audio_frames)

        # Log the total size of the recorded audio
        logging.info("Audio recording completed, total bytes: %d", audio_buffer.getbuffer().nbytes)
    except Exception as e:
        # Log any errors that occur during the recording
        logging.error("An error occurred: %s", str(e))

if __name__ == "__main__":
    asyncio.run(main())
