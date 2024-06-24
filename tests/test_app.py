# tests/test_app.py
import os
import pytest
from unittest.mock import patch, AsyncMock
import openai

from app import (
    initialize_env, create_openai_client, transcribe_speech_to_text,
    interact_with_openai, synthesize_and_play_speech, main
)

@pytest.fixture(autouse=True)
def set_up_environment(monkeypatch):
    monkeypatch.setattr(os, 'environ', {
        'AZURE_OPENAI_API_KEY': 'mock-api-key',
        'AZURE_OPENAI_ENDPOINT': 'mock-endpoint',
        'AZURE_API_VERSION': '2024-05-01-preview',
        'VOICE_NAME': 'zh-CN-XiaoxiaoMultilingualNeural',
        'MODEL_NAME': 'chat-model'
    })

@pytest.mark.asyncio
async def test_create_openai_client_missing_env_vars():
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(openai.OpenAIError) as exc_info:
            await create_openai_client()
        assert "Missing credentials" in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_openai_client():
    # Test creation of OpenAI client with all environment variables set
    client = await create_openai_client()
    assert client is not None

@pytest.mark.asyncio
async def test_interact_with_openai_error_handling():
    client = AsyncMock()
    with pytest.raises(AssertionError):
        await interact_with_openai(client, ["wrong format"])
    
    # Testing with incorrect role
    with pytest.raises(AssertionError):
        await interact_with_openai(client, [{"role": "alien", "content": "Hello."}])

@pytest.mark.asyncio
async def test_transcribe_speech_to_text_error_handling(caplog):
    mock_whisper = AsyncMock()
    mock_whisper.transcribe_audio.side_effect = Exception("Transcription Error")
    await transcribe_speech_to_text(mock_whisper)
    assert "Speech-to-text conversion error: Transcription Error" in caplog.text


@pytest.mark.asyncio
async def test_synthesize_and_play_speech_error_handling(caplog):
    with patch('app.TextToSpeech') as MockTextToSpeech:
        mock_tts_instance = AsyncMock()
        MockTextToSpeech.return_value = mock_tts_instance
        mock_tts_instance.synthesize_speech.side_effect = Exception("Synthesis Error")
        with pytest.raises(Exception):
            await synthesize_and_play_speech("Hello world")
        assert "Error while synthesizing speech: Synthesis Error" in caplog.text


@pytest.fixture
def setup_env_vars():
    os.environ['AZURE_OPENAI_API_KEY'] = 'test-key'
    os.environ['AZURE_OPENAI_ENDPOINT'] = 'test-endpoint'
    os.environ['AZURE_API_VERSION'] = 'test-version'
    os.environ['VOICE_NAME'] = 'test-voice'
    os.environ['MODEL_NAME'] = 'test-model'

@pytest.mark.asyncio
async def test_main_flow_success(setup_env_vars):
    # Mocking the Async OpenAI client and WhisperSTT
    with patch('app.create_openai_client', AsyncMock()) as mock_client_creator, \
         patch('app.transcribe_speech_to_text', AsyncMock(return_value="test transcription")) as mock_transcribe, \
         patch('app.synthesize_and_play_speech', AsyncMock()) as mock_synth:
        
        openai_client_mock = AsyncMock()
        openai_client_mock.chat.completions.create.return_value = AsyncMock(choices=[AsyncMock(message=AsyncMock(content="Mocked response"))])
        mock_client_creator.return_value = openai_client_mock
        
        await main()
        
        mock_client_creator.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_synth.assert_called_once_with("Mocked response")

@pytest.mark.asyncio
async def test_main_flow_no_openai_client(setup_env_vars):
    with patch('app.create_openai_client', AsyncMock(return_value=None)), \
         patch('app.transcribe_speech_to_text') as mock_transcribe, \
         patch('app.synthesize_and_play_speech') as mock_synth:

        await main()
        mock_transcribe.assert_not_called()
        mock_synth.assert_not_called()
