import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import random
import os
import webrtcvad
from dotenv import load_dotenv
from openai import AzureOpenAI

class WhisperSTT:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION")
        )
        self.sample_rate = 16000  # Sample rate for audio recording

    def array_to_bytes(self, audio_array):
        # Generate a random filename and temp path for saving an audio file
        random_digits = str(random.randint(10000000, 99999999))
        tmp_audio_file_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{random_digits}.wav")
        # Write the numpy audio array to the WAV file at the generated path
        write(tmp_audio_file_path, self.sample_rate, audio_array.astype(np.int16))
        return tmp_audio_file_path

    def record_audio_vad(self, max_duration=120):
        # Setup voice activity detector with highest sensitivity
        vad = webrtcvad.Vad(3)
        print("Recording...")

        # To keep track of recorded audio frames
        recorded_frames = []
        current_silence_duration = 0
        max_silence_duration = 1
        # Calculate number of silent frames after which recording should stop
        num_silent_frames_to_stop = int(max_silence_duration * self.sample_rate / 160)

        recording_active = True
        def callback(indata, frames, time_x, status):
            # Callback function to process blocks of audio input
            nonlocal current_silence_duration, recording_active
            if status and status.input_overflow:
                print("Input overflow:", status)
            # Check if the current frame contains speech
            is_speech = sum(vad.is_speech(frame.tobytes(), self.sample_rate) for frame in np.split(indata, np.arange(160, len(indata), 160)))
            if is_speech > 0:
                current_silence_duration = 0
            else:
                current_silence_duration += 1
            recorded_frames.append(indata.copy())
            # Stop recording if enough silence or max duration is reached
            if current_silence_duration >= num_silent_frames_to_stop or len(recorded_frames) * (160 / self.sample_rate) >= max_duration:
                recording_active = False
        try:
            with sd.InputStream(callback=callback, samplerate=self.sample_rate, channels=1, dtype='int16', blocksize=160):
                while recording_active:
                    sd.sleep(100)
        except Exception as e:
            print("Error occurred during recording:", e)
        print("Recording finished.")
        return np.concatenate(recorded_frames)

    def transcribe_audio(self, audio_array):
        # Convert audio array to WAV file
        temp_file_path = self.array_to_bytes(audio_array)
        try:
            # Open and process audio file for transcription
            with open(temp_file_path, "rb") as audio_file:
                try:
                    # Perform transcription using Whisper model
                    transcript = self.client.audio.transcriptions.create(
                        model=os.getenv("WHISPER_MODEL_NAME"),
                        file=audio_file
                    )
                    return transcript.text
                except Exception as e:
                    print("Error transcribing audio:", e)
                    return "Failed to transcribe audio"
        finally:
            # Clean up temporary audio file
            self.cleanup_temp_file(temp_file_path)

    def cleanup_temp_file(self, file_path):
        # Remove temporary file from file system
        os.unlink(file_path)

if __name__ == '__main__':
    # Example code to use the class
    whisper_stt = WhisperSTT()
    recorded_audio = whisper_stt.record_audio_vad()
    transcript = whisper_stt.transcribe_audio(recorded_audio)
    print(transcript)  # Print out the transcription result
