import io
import asyncio
import pytest
import itertools
from threading import Timer
from unittest.mock import Mock,patch, AsyncMock, MagicMock
import sounddevice as sd

from voicerecorder import VoiceRecorder,main

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


def test_record_audio_with_speech_frames(voice_recorder):
    callback_mock = Mock()
    audio_data = b'speech_audio_data'
    vad_mock = Mock()
    
    # Return True for a limited number of calls to simulate speech, then return False
    is_speech_responses = [True] * 10 + [False] * 100  # True for first 10 calls, then False
    vad_mock.is_speech = Mock(side_effect=is_speech_responses)
    
    stream_mock = MagicMock()
    stream_mock.read.return_value = (audio_data, None)
    stream_mock.__enter__.return_value = stream_mock

    voice_recorder.sample_rate = 16000
    voice_recorder.vad = vad_mock

    with patch('voicerecorder.sd.RawInputStream', return_value=stream_mock), \
         patch('voicerecorder.webrtcvad.Vad', return_value=vad_mock):
        voice_recorder.record_audio(callback=callback_mock)
        
        # Validate that is_speech was called expected number of times
        assert vad_mock.is_speech.call_count < 112

@patch('voicerecorder.VoiceRecorder.record_audio_vad')
@patch('voicerecorder.time.time')
def test_main_records_audio_and_logs_duration_and_size(mock_time: MagicMock, mock_record_audio_vad: MagicMock) -> None:
    """Test case: recording audio and logging duration and size."""
    mock_time.side_effect = [1000, 1020]
    mock_record_audio_vad.return_value = [b'frame1', b'frame2']
    with patch('voicerecorder.logging.debug') as mock_debug, \
         patch('voicerecorder.logging.info') as mock_info, \
         patch('voicerecorder.logging.error') as mock_error:
        asyncio.run(main())
        mock_debug.assert_called_with("Recording duration: %.2f seconds", 20.0)
        # Adjusted to the correct total bytes:
        mock_info.assert_called_with("Audio recording completed, total bytes: %d", 12)
        mock_error.assert_not_called()

def test_not_speech(voice_recorder):
    mock_vad = Mock()
    mock_vad.is_speech.return_value = False
    recorded_frames = []
    current_silence_duration = 5
    num_silent_frames_to_stop = 10

    voice_recorder._process_audio_frame(b'', mock_vad, recorded_frames, current_silence_duration, num_silent_frames_to_stop)

    # Expect current_silence_duration to increment by 1
    assert current_silence_duration == 6, "Expected current_silence_duration to increment by 1"

def test_not_speech(voice_recorder):
    mock_vad = Mock()
    mock_vad.is_speech.return_value = False
    recorded_frames = []
    current_silence_duration = 5
    num_silent_frames_to_stop = 10

    # Capture the potentially updated duration
    updated_duration = voice_recorder._process_audio_frame(b'', mock_vad, recorded_frames, current_silence_duration, num_silent_frames_to_stop)

    assert updated_duration == 6, "Expected current_silence_duration to increment by 1"

def test_speech(voice_recorder):
    mock_vad = Mock()
    mock_vad.is_speech.return_value = True
    recorded_frames = []
    current_silence_duration = 5
    num_silent_frames_to_stop = 10

    frame_data = b'your_expected_data'
    # Capture the potentially updated duration
    updated_duration = voice_recorder._process_audio_frame(frame_data, mock_vad, recorded_frames, current_silence_duration, num_silent_frames_to_stop)

    assert updated_duration == 0, "Expected current_silence_duration to reset to 0"
    assert frame_data in recorded_frames, "Expected the frame data to be recorded"
