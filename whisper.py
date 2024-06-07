import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import random
import os
import webrtcvad
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Set parameters for Azure OpenAI Service Whisper
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION")
)

def array_to_bytes(audio_array, sample_rate):
    random_digits = str(random.randint(10000000, 99999999))
    tmp_audio_file_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{random_digits}.wav")
    write(tmp_audio_file_path, sample_rate, audio_array.astype(np.int16))
    return tmp_audio_file_path

def record_audio_vad(max_duration=120, sample_rate=16000):
    vad = webrtcvad.Vad(1)  # Medium sensitivity mode
    print("Recording...")

    recorded_frames = []
    current_silence_duration = 0
    max_silence_duration = 2
    num_silent_frames_to_stop = int(max_silence_duration * sample_rate / 160)

    recording_active = True
    def callback(indata, frames, time_x, status):
        nonlocal current_silence_duration
        if status and status.input_overflow:
            print("Input overflow:", status)
        is_speech = sum(vad.is_speech(frame.tobytes(), sample_rate) for frame in np.split(indata, np.arange(160, len(indata), 160)))
        if is_speech > 0:
            current_silence_duration = 0
        else:
            current_silence_duration += 1
        recorded_frames.append(indata.copy())
        if current_silence_duration >= num_silent_frames_to_stop or len(recorded_frames) * (160 / sample_rate) >= max_duration:
            nonlocal recording_active
            recording_active = False
    try:
        with sd.InputStream(callback=callback, samplerate=sample_rate, channels=1, dtype='int16', blocksize=160):
            while recording_active:
                sd.sleep(100)
    except Exception as e:
        print("Error occurred during recording:", e)
    print("Recording finished.")
    return np.concatenate(recorded_frames), sample_rate

def transcribe_audio(audio_array, sample_rate=16000):
    # 创建临时文件
    temp_file_path = array_to_bytes(audio_array, sample_rate)
    try:
        with open(temp_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=os.getenv("WHISPER_MODEL_NAME"),
                file=audio_file
            )
        return transcript.text
    finally:
        cleanup_temp_file(temp_file_path)

def cleanup_temp_file(file_path):
    os.unlink(file_path)

def main():
    recorded_audio, audio_sample_rate = record_audio_vad()
    temp_audio_file_path = array_to_bytes(recorded_audio, audio_sample_rate)
    transcript = transcribe_audio(temp_audio_file_path,16000)
    print(transcript)  # Display the transcription result

if __name__ == '__main__':
    main()
