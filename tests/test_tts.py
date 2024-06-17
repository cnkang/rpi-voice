from unittest.mock import patch, AsyncMock
import httpx
import pytest
from httpx import Response, Request
from unittest.mock import patch
import pytest
from tts import TextToSpeech

async def test_synthesize_speech():
    """
    Tests that synthesizing speech from a valid string in normal conditions works properly,
    ensuring the system raises an AssertionError with an appropriate message if it fails.
    """
    tts = TextToSpeech()
    with pytest.raises(AssertionError, match=r"Failed to synthesize speech"):
        with patch('tts.TextToSpeech._make_request', side_effect=Exception("Simulated failure")):
            await tts.synthesize_speech("Test speech synthesis.")

async def test_synthesize_speech_empty_string():
    """
    Tests to ensure producing speech from an empty string raises an AssertionError.
    Verifies that invalid inputs are handled correctly.
    """
    tts = TextToSpeech()
    with pytest.raises(AssertionError, match=r"Failed to synthesize speech: Empty input provided"):
        await tts.synthesize_speech("")

async def test_synthesize_speech_http_error():
    """
    Tests exception handling during an HTTP error. This confirms that the TextToSpeech class
    can handle exceptions in the HTTP POST request gracefully by raising a custom AssertionError.
    """
    tts = TextToSpeech()
    with patch('tts.TextToSpeech._make_request', side_effect=httpx.HTTPStatusError(
        message="Error", request=Request(method="POST", url="dummy"), response=Response(status_code=500))):
        with pytest.raises(AssertionError, match=r"Failed after maximum retries."):
            await tts.synthesize_speech("This should fail but handle")

async def test_synthesize_speech_retry_logic():
    """
    Ensures the TextToSpeech class correctly retries HTTP POST requests under transient errors,
    simulating two failures followed by a success. Tests that retry logic will succeed
    after the maximum number of retry attempts.
    """
    tts = TextToSpeech()

    # Modify retry properties to shorten test execution time
    tts.max_retries = 3
    tts.retry_delay = 1

    # Simulate side effects for the mocked POST method: two failed attempts followed by a success
    http_error_response = httpx.Response(status_code=500)
    http_error = httpx.HTTPStatusError(message="Temporary Error", request=Request(method="POST", url="dummy"), response=http_error_response)
    successful_response = httpx.Response(status_code=200, content=b"Success binary content for audio", 
                                         request=Request(method="POST", url="dummy"))
    side_effects = [http_error, http_error, successful_response]

    with patch('tts.TextToSpeech._make_request', side_effect=side_effects) as mock_post:
        audio_stream = await tts.synthesize_speech("Testing retry logic.")
        
        assert mock_post.call_count == 3  # Verify that request was retried the correct number of times
        assert audio_stream is not None  # Confirm that audio stream is returned after successful retries

