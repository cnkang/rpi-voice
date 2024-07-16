"""
This module contains pytest fixtures and test functions for testing the WhisperSTT class.
"""
import logging
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock, mock_open
import pytest
import numpy as np
import io
import os
from pytest import raises

from whisper import WhisperSTT, main as whisper_main, save_temp_wav_file

@pytest.fixture(autouse=True)
def set_log_level(caplog):
    caplog.set_level(logging.DEBUG)

@pytest.fixture
def setup_env_vars():
    with patch.dict('os.environ', {
        "AZURE_OPENAI_ENDPOINT": "https://example.com",
        "AZURE_OPENAI_API_KEY": "abc123",
        "AZURE_API_VERSION": "2024-05-01-preview",
        "WHISPER_MODEL_NAME": "whisper-1"
    }):
        yield

@pytest.fixture
def whisper_client(setup_env_vars):
    return WhisperSTT()

@pytest.mark.asyncio
async def test_transcription_success(whisper_client):
    expected_transcription = "Hello, world!"
    with patch("builtins.open", mock_open(read_data="audio data")), \
         patch.object(whisper_client.client.audio.transcriptions, 'create', 
                      AsyncMock(return_value=MagicMock(text=expected_transcription))):
        result = await whisper_client.transcribe_audio("path/to/mock_audio.wav")
        assert result == expected_transcription, "The transcription result should match the expected output."


@pytest.mark.asyncio
async def test_transcription_failure(whisper_client):
    # Simulate an exception on transcription failure
    with patch.object(whisper_client.client.audio.transcriptions, 'create', AsyncMock(side_effect=Exception("Network Error"))):
        result = await whisper_client.transcribe_audio("path/to/mock_audio.wav")
        assert result == "Failed to transcribe audio", "Should handle transcription service failures gracefully."

def test_save_temp_wav_file():
    # Simulating saving a temporary WAVE file
    mock_audio_stream = io.BytesIO(b"fake wav data")
    file_path = save_temp_wav_file(mock_audio_stream)
    assert file_path.endswith('.wav'), "Should produce a .wav temporary file."
    os.remove(file_path)  # Clean up the generated file after the test

@pytest.mark.asyncio
async def test_main_success(caplog):
    # Ensuring proper logging and function execution in the main function
    with patch('voicerecorder.VoiceRecorder') as mock_recorder:
        mock_future = asyncio.Future()
        mock_future.set_result(np.zeros((1600,), dtype=np.int16))
        mock_recorder.return_value.record_audio_vad.return_value = mock_future
        mock_recorder.return_value.array_to_wav_bytes.return_value = io.BytesIO(b"WAV data")
        
        with patch.object(WhisperSTT, 'transcribe_audio', AsyncMock(return_value="Transcription successful")):
            await whisper_main()
        
        assert "Transcription successful" in caplog.text, "Main function should log successful transcription."

@pytest.mark.asyncio
async def test_main_exception_handling(caplog):
    # Testing error handling in the main function
    with patch('voicerecorder.VoiceRecorder.record_audio_vad', side_effect=Exception("Test Error")):
        await whisper_main()
        assert "An error occurred: Test Error" in caplog.text, "Errors should be logged properly in the main function."

@pytest.mark.asyncio
async def test_environment_validation_missing_key():
    with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': '', 'AZURE_OPENAI_ENDPOINT': ''}, clear=True):
        with pytest.raises(EnvironmentError, match="Environment variables for Azure OpenAI Service not set"):
            WhisperSTT()
