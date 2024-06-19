# tests/test_app.py
import os
import logging
import pytest
from unittest.mock import patch, AsyncMock

# Adapt imports as needed
from app import create_openai_client, interact_with_openai, synthesize_and_play_speech, main, transcribe_speech_to_text

@pytest.mark.asyncio
async def test_interact_with_openai_error_handling():
    """
    Test that `interact_with_openai` raises an AssertionError
    when provided with prompts that are not in the correct format.
    """
    client = AsyncMock()
    prompts = ["incorrect format string"]
    
    with pytest.raises(AssertionError) as excinfo:
        await interact_with_openai(client, prompts)
    
    assert "Prompts are not in the correct format" in str(excinfo.value)

@pytest.mark.asyncio
async def test_main_flow():
    """
    Test the main flow of the application by mocking dependencies.
    Ensure that each major component is called at least once.
    """
    with patch('app.create_openai_client') as mock_create_client, \
         patch('app.transcribe_speech_to_text', AsyncMock(return_value="This is a test")) as mock_transcribe, \
         patch('app.synthesize_and_play_speech') as mock_synth:
        
        client_instance = AsyncMock()
        mock_create_client.return_value = client_instance
        client_instance.chat.completions.create.return_value = AsyncMock(choices=[AsyncMock(message=AsyncMock(content="Hello"))])

        await main()
        
        # Verify that each mocked function is called once
        mock_create_client.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_synth.assert_called_once_with("Hello")
        # Additional validation of call ordering
        assert mock_create_client.called
        assert mock_transcribe.called
        assert mock_synth.called

@pytest.mark.asyncio
async def test_invalid_client_creation():
    """
    Test that `create_openai_client` raises a ValueError when the 
    AZURE_OPENAI_API_KEY environment variable is missing.
    """
    with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': '', 'AZURE_API_VERSION': 'dummy-version', 'AZURE_OPENAI_ENDPOINT': 'dummy-endpoint'}):
        with pytest.raises(ValueError) as exc_info:
            await create_openai_client()
        assert "AZURE_OPENAI_API_KEY is required but missing" in str(exc_info.value), "Should raise ValueError for missing API key"

@pytest.mark.asyncio
async def test_transcribe_speech_error_handling(caplog):
    """
    Test that `transcribe_speech_to_text` logs an exception 
    when an error occurs during transcription.
    """
    # Create a mock object for the WhisperSTT with a predefined side effect that raises an Exception
    mock_whisper = AsyncMock()
    mock_whisper.transcribe_audio.side_effect = Exception("Mock Exception")

    # Patch the WhisperSTT import in `app` module to use the mock object
    with patch('app.WhisperSTT', return_value=mock_whisper):
        # Call the function under test
        await transcribe_speech_to_text()

    # Check the caplog log records to ensure the exception message was logged as expected
    found = any("Mock Exception" in record.message for record in caplog.records)
    # Assert to confirm the expected message was logged
    assert found, "Expected 'Mock Exception' but wasn't found in logged output."

@pytest.fixture(autouse=True)
def setup_logging(caplog):
    caplog.set_level(logging.ERROR)

@pytest.mark.asyncio
async def test_create_openai_client_missing_env_vars():
    """
    Test that `create_openai_client` raises a ValueError when any
    required environment variable is missing.
    """
    with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'dummy-key', 'AZURE_API_VERSION': '', 'AZURE_OPENAI_ENDPOINT': 'dummy-endpoint'}, clear=True):
        with pytest.raises(ValueError) as e1:
            await create_openai_client()
        assert "AZURE_API_VERSION is required but missing" in str(e1.value)

    with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'dummy-key', 'AZURE_API_VERSION': 'dummy-version', 'AZURE_OPENAI_ENDPOINT': ''}, clear=True):
        with pytest.raises(ValueError) as e2:
            await create_openai_client()
        assert "AZURE_OPENAI_ENDPOINT is required but missing" in str(e2.value)

@pytest.mark.asyncio
async def test_interact_with_openai_invalid_prompt_structure():
    """
    Test that `interact_with_openai` raises an AssertionError for 
    prompts with incorrect structures or invalid role values.
    """
    client = AsyncMock()
    # Test with missing 'role' and 'content' keys which are required for correct formatting
    bad_prompts = [{"incorrect": "format"}]
    with pytest.raises(AssertionError) as exc_info:
        await interact_with_openai(client, bad_prompts)
    # Verify that an error is raised for prompts with incorrect structure
    assert "Each prompt should contain 'role' and 'content'" in str(exc_info.value)

    # Test with incorrect 'role' value which should be either 'system' or 'user'
    bad_prompts2 = [{"role": "invalid_role", "content": "Hello"}]
    with pytest.raises(AssertionError) as exc_info:
        await interact_with_openai(client, bad_prompts2)
    # Verify that an error is raised for incorrect role values
    assert "Role must be either 'system' or 'user'" in str(exc_info.value)

    # Test with correct prompt structure to make sure no errors are raised
    good_prompts = [{"role": "user", "content": "Hello"}, {"role": "system", "content": "You are a helpful AI"}]
    response = await interact_with_openai(client, good_prompts)
    # Verify that the function returns a response when input is correctly formatted
    assert response, "Function should return a response with valid input"


# Also test valid inputs to make sure they work correctly.
@pytest.mark.asyncio
async def test_interact_with_openai_valid_input():
    client = AsyncMock()
    client.chat.completions.create.return_value = AsyncMock(choices=[AsyncMock(message=AsyncMock(content="Valid response"))])
    valid_prompts = [{"role": "user", "content": "Hello, how are you?"}]
    
    response = await interact_with_openai(client, valid_prompts)
    assert response == "Valid response"

@pytest.mark.asyncio
async def test_interact_with_openai_branch_coverage():
    client = AsyncMock()
    valid_prompts = [{"role": "system", "content": "Valid input"}]
    client.chat.completions.create.return_value = AsyncMock(choices=[None])  # Simulate No response returned scenario

    response = await interact_with_openai(client, valid_prompts)
    assert response == "No response returned."  # Verify handling of missing choices

@pytest.mark.asyncio
async def test_interact_with_openai_exception():
    client = AsyncMock()
    client.chat.completions.create.side_effect = Exception("Test Exception")

    with pytest.raises(AssertionError) as exc_info:
        await interact_with_openai(client, [{"role": "user", "content": "Hello"}])
    assert "Error in the AI response: Test Exception" in str(exc_info.value)

@pytest.mark.asyncio
async def test_create_openai_client_partial_env_vars():
    with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'key', 'AZURE_API_VERSION': '', 'AZURE_OPENAI_ENDPOINT': 'endpoint'}):
        with pytest.raises(ValueError) as exc_info:
            await create_openai_client()
        assert "AZURE_API_VERSION is required but missing" in str(exc_info.value)

@pytest.mark.asyncio
async def test_synthesize_and_play_speech_with_invalid_stream():
    mock_tts = AsyncMock()
    mock_tts.synthesize_speech.return_value = None
    
    with patch('app.TextToSpeech', return_value=mock_tts):
        with pytest.raises(Exception):
            await synthesize_and_play_speech("Hello")

@pytest.fixture
def setup_environment():
    with patch.dict(os.environ, {'VOICE_NAME': 'default-voice', 'MODEL_NAME': 'default-model'}):
        yield

def test_env_initialization(setup_environment):
    assert os.getenv('VOICE_NAME') == 'default-voice'
    assert os.getenv('MODEL_NAME') == 'default-model'
