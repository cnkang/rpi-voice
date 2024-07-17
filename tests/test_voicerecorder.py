import io
import asyncio
import pytest
from unittest.mock import patch, AsyncMock
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
    # Sample data which ensures sufficient length to test
    sample_frames = [b'\x01\x02\x03\x04' * 100]
    result = voice_recorder.array_to_wav_bytes(sample_frames)
    
    # Assert the expected output for buffer
    assert isinstance(result, io.BytesIO)
    # Check header plus frame size (WAV headers are generally 44 bytes long)
    assert len(result.getvalue()) == (44 + 400)  # 44 bytes header + data size


@pytest.mark.asyncio
async def test_record_audio_vad_no_speech():
    # Test recording with no speech detected
    v = VoiceRecorder()
    v.sample_rate = 16000
    with patch('webrtcvad.Vad.is_speech', return_value=False) as mock_is_speech:
        mock_is_speech.side_effect = lambda frame, rate: False
        
        v.recorder = AsyncMock()
        v.recorder.read.side_effect = [b'\x00\x00' for _ in range(16000)] + [b'']
        
        result = await v.record_audio_vad(max_duration=2.0, max_silence_duration=1.0)
        assert len(result) - 16000 // 160 < 10, "Frames processed incorrectly as silence not detected"

@pytest.mark.asyncio
async def test_record_audio_vad_portaudio_error():
    # Test recording handling with a simulated PortAudioError
    v = VoiceRecorder()
    with patch('sounddevice.InputStream', side_effect=sd.PortAudioError('Test error')):
        with pytest.raises(RuntimeError, match="^Failed during recording"):
            await v.record_audio_vad(max_duration=2.0)
