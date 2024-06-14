import numpy as np
import pytest

from whisper import WhisperSTT

@pytest.mark.asyncio
async def test_record_audio_vad():
    whisper_stt = WhisperSTT()

    # 创建模拟音频数据，这里我们可以创建一些简单的静音数据或测试用的发音数据
    simulated_silence = np.zeros((1600,), dtype=np.int16)  # 一秒的静音数据
    simulated_audio = [simulated_silence]  # 模拟连续一秒的静音

    # 测试
    result = whisper_stt.record_audio_vad(simulation_input=simulated_audio)
    assert isinstance(result, np.ndarray), "Result should be a numpy array"
    assert result.shape[0] == 1600, "The shape of the result array should match the input audio duration."

@pytest.mark.asyncio
async def test_transcribe_audio():
    # Setup
    whisper_stt = WhisperSTT()
    audio = np.array([1, 2, 3], dtype=np.int16)
    
    try:
        # Attempt to run the transcription
        await whisper_stt.transcribe_audio(audio)
    except Exception as e:
        # If an exception occurs, the test should fail
        pytest.fail(f"transcribe_audio method failed with exception: {e}")
