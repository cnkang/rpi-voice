# tests/test_app.py
import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock, call

# Adapt imports as needed
from app import create_openai_client, interact_with_openai, synthesize_and_play_speech, main, transcribe_speech_to_text

@pytest.mark.asyncio
async def test_interact_with_openai_error_handling():
    client = AsyncMock()
    prompts = ["incorrect format string"]
    
    with pytest.raises(AssertionError) as excinfo:
        await interact_with_openai(client, prompts)
    
    assert "Prompts are not in the correct format" in str(excinfo.value)

@pytest.mark.asyncio
async def test_main_flow():
    # Mock the main function dependencies
    # Assume main executes all major components at least once
    with patch('app.create_openai_client') as mock_create_client, \
         patch('app.transcribe_speech_to_text', AsyncMock(return_value="This is a test")) as mock_transcribe, \
         patch('app.synthesize_and_play_speech') as mock_synth:
        client_instance = AsyncMock()
        mock_create_client.return_value = client_instance
        client_instance.chat.completions.create.return_value = AsyncMock(choices=[AsyncMock(message=AsyncMock(content="Hello"))])

        await main()
        mock_create_client.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_synth.assert_called_once_with("Hello")
        # Additionally, you might want to validate the ordering
        assert mock_create_client.called
        assert mock_transcribe.called
        assert mock_synth.called

@pytest.mark.asyncio
async def test_invalid_client_creation():
    # Set up the environment for testing missing API key
    with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': '', 'AZURE_API_VERSION': 'dummy-version', 'AZURE_OPENAI_ENDPOINT': 'dummy-endpoint'}):
        with pytest.raises(ValueError) as exc_info:
            await create_openai_client()
        assert "AZURE_OPENAI_API_KEY is required but missing" in str(exc_info.value), "Should raise ValueError for missing API key"

@pytest.mark.asyncio
async def test_transcribe_speech_error_handling(caplog):
    with patch('app.WhisperSTT.transcribe_audio', side_effect=Exception("Mock Exception")):
        await transcribe_speech_to_text()

    # Check that the "Mock Exception" appears in any of the log messages
    found = any("Mock Exception" in record.message for record in caplog.records)
    assert found, "Expected 'Mock Exception' but wasn't found in logged output."



@pytest.fixture
def setup_environment():
    with patch.dict(os.environ, {'VOICE_NAME': 'default-voice', 'MODEL_NAME': 'default-model'}):
        yield

def test_env_initialization(setup_environment):
    assert os.getenv('VOICE_NAME') == 'default-voice'
    assert os.getenv('MODEL_NAME') == 'default-model'
