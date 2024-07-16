import io
import pytest
from unittest.mock import patch

from voicerecorder import VoiceRecorder

@pytest.fixture
def voice_recorder():
    return VoiceRecorder()

def test_array_to_pcm_bytes_success(voice_recorder):
    # Prepare test data
    sample_frames = [b'\x01\x02', b'\x03\x04']

    # Call the method
    result = voice_recorder.array_to_pcm_bytes(sample_frames)

    # Assert the expected output
    assert isinstance(result, io.BytesIO)
    assert result.getvalue() == b'\x01\x02\x03\x04'

def test_array_to_pcm_bytes_exception(voice_recorder):
    # Prepare faulty test data that raises an exception when written to io.BytesIO
    with patch('io.BytesIO.write', side_effect=Exception("Write failure")):
        with pytest.raises(Exception) as exc_info:
            voice_recorder.array_to_pcm_bytes([b'test'])

        # Assert the exception message
        assert "Failed to write audio to buffer" in str(exc_info.value)

def test_array_to_wav_bytes_length(voice_recorder):
    # Prepare test data
    sample_frames = [b'\x01\x02\x03\x04' * 100]  # Sample data repeated to create sufficient length

    # Call the method
    result = voice_recorder.array_to_wav_bytes(sample_frames)

    # Assert the expected output
    assert isinstance(result, io.BytesIO)
    # Check header plus frame size (WAV headers are generally 44 bytes long)
    assert len(result.getvalue()) == (44 + 400)

@pytest.mark.asyncio
async def test_record_audio_vad(voice_recorder):
    # Mock sounddevice.InputStream to simulate audio input without real audio hardware
    with patch('sounddevice.InputStream'):
        # Execute record_audio_vad with short max_duration and max_silence_duration
        frames = await voice_recorder.record_audio_vad(max_duration=0.1, max_silence_duration=0.05)
        
        assert isinstance(frames, list)  # Output should be a list of frames
        # Ensure there are frames generated depending on mock setup/behavior in `process_frame`

# To run these tests, you can use the following command in the terminal:
# pytest test_voicerecorder.py
