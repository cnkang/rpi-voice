import io
import asyncio
import pytest
import time
import itertools
from unittest.mock import patch, AsyncMock, MagicMock
import sounddevice as sd

from voicerecorder import VoiceRecorder

@pytest.fixture
def voice_recorder():
    return VoiceRecorder()

def test_array_to_pcm_bytes_success(voice_recorder):
    sample_frames = [b'\x01\x02', b'\x03\x04']
    result = voice_recorder.array_to_pcm_bytes(sample_frames)
    assert isinstance(result, io.BytesIO)
    assert result.getvalue() == b'\x01\x02\x03\x04'

def test_array_to_pcm_bytes_exception(voice_recorder):
    with pytest.raises(Exception) as exc_info:
        voice_recorder.array_to_pcm_bytes(['abc', 'def'])
    assert "Failed to write audio to buffer" in str(exc_info.value)

def test_array_to_wav_bytes_length(voice_recorder):
    sample_frames = [b'\x01\x02\x03\x04' * 100]
    result = voice_recorder.array_to_wav_bytes(sample_frames)
    assert isinstance(result, io.BytesIO)
    assert len(result.getvalue()) == (44 + 400)  # 44 bytes header + data size

@pytest.mark.asyncio
@patch('sounddevice.InputStream', create=True)
@pytest.mark.timeout(10)
async def test_record_audio_vad_no_speech(mock_stream):
    mock_stream.return_value.__enter__.return_value.read = AsyncMock(return_value=(b'\x00\x00' * 8000, None))
    recorder = VoiceRecorder()
    recorder.sample_rate = 16000
    with patch('webrtcvad.Vad.is_speech', return_value=False) as mock_is_speech:
        mock_is_speech.side_effect = itertools.cycle([False])  # Continuously return False
        result = await recorder.record_audio_vad(max_duration=1.0, max_silence_duration=0.5)
        assert not result, "No speech should result in no frames recorded"

@pytest.mark.asyncio
async def test_record_audio_vad_portaudio_error():
    with patch('sounddevice.InputStream', side_effect=sd.PortAudioError('Test error')):
        voice_recorder = VoiceRecorder()
        with pytest.raises(RuntimeError, match="^Failed during recording"):
            await voice_recorder.record_audio_vad(max_duration=2.0)

@pytest.mark.asyncio
@patch('sounddevice.InputStream')
async def test_recorder_timeout(mock_input_stream):
    # Simulate delay by making each read operation take longer than the test timeout
    mock_input_stream.return_value.__enter__.return_value.read = AsyncMock(
        side_effect=lambda *args, **kwargs: asyncio.sleep(0.5))
    
    voice_recorder = VoiceRecorder()
    with pytest.raises(asyncio.TimeoutError):
        # Use a timeout shorter than the delay induced by stream read
        await asyncio.wait_for(voice_recorder.record_audio_vad(max_duration=2.0), timeout=0.1)
