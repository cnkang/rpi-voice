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
        # 加载环境变量
        load_dotenv()
        
        # 初始化 Azure OpenAI 客户端
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION")
        )
        self.sample_rate = 16000  # 采样率

    def array_to_bytes(self, audio_array):
        random_digits = str(random.randint(10000000, 99999999))
        tmp_audio_file_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{random_digits}.wav")
        write(tmp_audio_file_path, self.sample_rate, audio_array.astype(np.int16))
        return tmp_audio_file_path

    def record_audio_vad(self, max_duration=120):
        vad = webrtcvad.Vad(3)
        print("Recording...")

        recorded_frames = []
        current_silence_duration = 0
        max_silence_duration = 1
        num_silent_frames_to_stop = int(max_silence_duration * self.sample_rate / 160)

        recording_active = True
        def callback(indata, frames, time_x, status):
            nonlocal current_silence_duration, recording_active
            if status and status.input_overflow:
                print("Input overflow:", status)
            is_speech = sum(vad.is_speech(frame.tobytes(), self.sample_rate) for frame in np.split(indata, np.arange(160, len(indata), 160)))
            if is_speech > 0:
                current_silence_duration = 0
            else:
                current_silence_duration += 1
            recorded_frames.append(indata.copy())
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
        temp_file_path = self.array_to_bytes(audio_array)
        try:
            with open(temp_file_path, "rb") as audio_file:
                try:
                    transcript = self.client.audio.transcriptions.create(
                        model=os.getenv("WHISPER_MODEL_NAME"),
                        file=audio_file
                    )
                    return transcript.text
                except Exception as e:
                    print("Error transcribing audio:", e)
                    return "Failed to transcribe audio"
        finally:
            self.cleanup_temp_file(temp_file_path)


    def cleanup_temp_file(self, file_path):
        os.unlink(file_path)

if __name__ == '__main__':
    # 示例代码使用类
    whisper_stt = WhisperSTT()
    recorded_audio = whisper_stt.record_audio_vad()
    transcript = whisper_stt.transcribe_audio(recorded_audio)
    print(transcript)  # 显示转录结果
