import asyncio
import tempfile
import uuid
import os
import logging
from typing import Optional, List

import sounddevice as sd
import webrtcvad

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class VoiceRecorder:
    def __init__(self) -> None:
        self.sample_rate: int = 16000
        self.stop_event: asyncio.Event = asyncio.Event()

    def array_to_pcm_bytes(self, audio_frames: List[bytes]) -> str:
        unique_filename = str(uuid.uuid4())
        tmp_audio_file_path = os.path.join(
            tempfile.gettempdir(), f"temp_audio_{unique_filename}.pcm"
        )
        try:
            with open(tmp_audio_file_path, 'wb') as audio_file:
                for frame in audio_frames:
                    audio_file.write(frame)
            return tmp_audio_file_path
        except Exception as e:
            logging.error("Failed to write audio to file: %s", str(e))
            raise

    async def record_audio_vad(
        self,
        max_duration: float = 60.0,
        max_silence_duration: float = 1.0
    ) -> List[bytes]:
        vad = webrtcvad.Vad(mode=3)
        logging.info("Start recording...")
        recorded_frames: List[bytes] = []
        current_silence_duration: int = 0
        num_silent_frames_to_stop = int(max_silence_duration * self.sample_rate / 160)
        recording_active: bool = True

        def update_recording_status(frame_data: bytes, is_speech: bool) -> None:
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
            nonlocal recording_active
            is_speech = vad.is_speech(frame_data, self.sample_rate)
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
    voice_recorder = VoiceRecorder()
    try:
        audio_frames = await voice_recorder.record_audio_vad()
        pcm_file_path = voice_recorder.array_to_pcm_bytes(audio_frames)
        logging.info("Audio saved to %s", pcm_file_path)
    except Exception as e:
        logging.error("An error occurred: %s", str(e))

if __name__ == "__main__":
    asyncio.run(main())
