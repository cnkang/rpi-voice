import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import webrtcvad
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Set parameters for Azure OpenAI Service Whisper
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = os.getenv("AZURE_API_VERSION")
)

def array_to_bytes(audio_array, sample_rate):
    temp_audio_file_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    write(temp_audio_file_path, sample_rate, audio_array.astype(np.int16))
    return temp_audio_file_path

def record_audio_vad(max_duration=120, sample_rate=16000):
    vad = webrtcvad.Vad(1)  # Medium sensitivity mode
    print("Recording...")

    recorded_frames = []
    current_silence_duration = 0
    max_silence_duration = 2
    num_silent_frames_to_stop = int(max_silence_duration * sample_rate / 160)

    recording_active = True  # Use a boolean to keep track of active recording
    def callback(indata, frames, time_x, status):
        nonlocal current_silence_duration  # Allows modification of this variable
        if status and status.input_overflow:
            print("Input overflow:", status)

        # Check for speech presence in the current frame
        is_speech = sum(vad.is_speech(frame.tobytes(), sample_rate) for frame in np.split(indata, np.arange(160, len(indata), 160)))
        # Reset silence counter if speech is detected
        if is_speech > 0:
            current_silence_duration = 0
        else:
            current_silence_duration += 1

        recorded_frames.append(indata.copy())

        # Stop recording based on silence duration or max duration
        if current_silence_duration >= num_silent_frames_to_stop or len(recorded_frames) * (160 / sample_rate) >= max_duration:
            nonlocal recording_active
            recording_active = False
    try:
        # Process audio stream in blocking mode
        with sd.InputStream(callback=callback, samplerate=sample_rate, channels=1, dtype='int16', blocksize=160):
            # Loop until recording stops
            while recording_active:
                sd.sleep(100)  # Allow some time for audio processing
    except Exception as e:
        print("Error occurred during recording:", e)
    print("Recording finished.")
    return np.concatenate(recorded_frames), sample_rate

# Call the recording function
recorded_audio, audio_sample_rate = record_audio_vad()

# Assuming an existing function array_to_bytes for writing the numpy array to a wave file
temp_audio_file_path = array_to_bytes(recorded_audio, audio_sample_rate)

# Assuming an existing client setup for transcription
with open(temp_audio_file_path, "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
            model=os.getenv("WHISPER_MODEL_NAME"),
            file=audio_file
        )

print(transcript)  # Display the transcription result

# Cleanup the temporary file
os.unlink(temp_audio_file_path)
