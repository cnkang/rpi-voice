import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import tts as tts_module
from httpx import HTTPStatusError, TimeoutException

# Make sure to simulate all components of an HTTP interaction including the request and response
def mock_http_status_error():
    request = MagicMock()
    response = MagicMock()
    return HTTPStatusError(message="HTTP error", request=request, response=response)


@pytest.mark.asyncio
async def test_synthesize_speech_success():
    tts = tts_module.TextToSpeech()
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_response = AsyncMock()
        mock_response.raise_for_status.side_effect = None  # No error raising
        mock_response.content = b"audio data"
        mock_post.return_value.__aenter__.return_value = mock_response
        await tts.synthesize_speech("Test speech synthesis.")



@pytest.mark.asyncio
async def test_synthesize_speech_empty_string():
    tts = tts_module.TextToSpeech()
    with pytest.raises(ValueError, match="Text cannot be empty"):
        await tts.synthesize_speech("")


@pytest.mark.asyncio
async def test_synthesize_speech_http_error():
    tts = tts_module.TextToSpeech()
    with patch('httpx.AsyncClient.post', side_effect=RuntimeError) as mock_post:
        mock_response = AsyncMock()
        mock_response.raise_for_status.side_effect = mock_http_status_error()
        mock_post.return_value.__aenter__.return_value = mock_response
        with pytest.raises(RuntimeError):
            await tts.synthesize_speech("This should fail due to HTTP error")


@pytest.mark.asyncio
async def test_get_azure_cognitive_access_token_successful():
    tts = tts_module.TextToSpeech()
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_response = AsyncMock()
        mock_response.raise_for_status.side_effect = None
        mock_response.text = "valid_token"
        mock_post.return_value.__aenter__.return_value = mock_response
        await tts.get_azure_cognitive_access_token()


@pytest.mark.asyncio
async def test_get_azure_cognitive_access_token_http_error():
    tts = tts_module.TextToSpeech()
    with patch('httpx.AsyncClient.post', side_effect=RuntimeError):
        mock_response = AsyncMock()
        with pytest.raises(RuntimeError):
            await tts.get_azure_cognitive_access_token()


@pytest.mark.asyncio
async def test_get_azure_cognitive_access_token_timeout_error():
    tts = tts_module.TextToSpeech()
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = TimeoutException(message="Timeout")
        with pytest.raises(TimeoutException):
            await tts.get_azure_cognitive_access_token()


@pytest.mark.asyncio
async def test_convert_to_ssml_already_formatted():
    tts = tts_module.TextToSpeech()
    formatted_ssml = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='some-voice'>Hello, World!</voice></speak>"
    assert tts.convert_to_ssml(formatted_ssml) == formatted_ssml, "Should return the same SSML formatted text when already properly formatted."
