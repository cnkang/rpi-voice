import os
from unittest.mock import patch, AsyncMock
import subprocess
import httpx
import pytest
from httpx import Response, Request
import pytest

import tts as tts_module

@pytest.mark.asyncio
async def test_synthesize_speech():
    """
    Tests that synthesizing speech from a valid string in normal conditions works properly,
    ensuring the system raises an AssertionError with an appropriate message if it fails.
    """
    tts = tts_module.TextToSpeech()
    with pytest.raises(AssertionError, match=r"Failed to synthesize speech"):
        with patch('tts.TextToSpeech._make_request', side_effect=Exception("Simulated failure")):
            await tts.synthesize_speech("Test speech synthesis.")

@pytest.mark.asyncio
async def test_synthesize_speech_empty_string():
    """
    Tests to ensure producing speech from an empty string raises an AssertionError.
    Verifies that invalid inputs are handled correctly.
    """
    tts = tts_module.TextToSpeech()
    with pytest.raises(AssertionError, match=r"Failed to synthesize speech: Empty input provided"):
        await tts.synthesize_speech("")

@pytest.mark.asyncio
async def test_synthesize_speech_http_error():
    """
    Tests exception handling during an HTTP error. This confirms that the TextToSpeech class
    can handle exceptions in the HTTP POST request gracefully by raising a custom AssertionError.
    """
    tts = tts_module.TextToSpeech()
    with patch('tts.TextToSpeech._make_request', side_effect=httpx.HTTPStatusError(
        message="Error", request=Request(method="POST", url="dummy"), response=Response(status_code=500))):
        with pytest.raises(AssertionError, match=r"Failed after maximum retries."):
            await tts.synthesize_speech("This should fail but handle")
@pytest.mark.asyncio
async def test_synthesize_speech_retry_logic():
    """
    Ensures the TextToSpeech class correctly retries HTTP POST requests under transient errors,
    simulating two failures followed by a success. Tests that retry logic will succeed
    after the maximum number of retry attempts.
    """
    tts = tts_module.TextToSpeech()

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

@pytest.mark.asyncio
async def test_main_function_normal_behavior():
    """
    Test the main function from the tts module to ensure that it can run without throwing an exception
    when provided with valid data.
    """
    with patch('tts.TextToSpeech.synthesize_speech', return_value=b"some audio data") as mock_synthesize:
        try:
            # 调用tts模块中的main函数
            await tts_module.main()
            assert mock_synthesize.called, "The synthesize_speech function should have been called."
        except AssertionError as e:
            pytest.fail(f"Unexpected AssertionError thrown: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected exception thrown: {e}")

@pytest.mark.asyncio
async def test_missing_api_key():
    with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': ''}):
        with pytest.raises(ValueError, match=r"Necessary configuration missing"):
            tts_module.TextToSpeech()

@pytest.mark.asyncio
async def test_retry_logic_on_temporary_failures():
    with patch('httpx.AsyncClient.post', side_effect=httpx.ReadTimeout("timeout")):
        tts = tts_module.TextToSpeech()
        with pytest.raises(AssertionError):
            await tts.synthesize_speech("Hello")
def test_script_main_executes():
    """
    Test that the whisper.py script executes its main functionality when run as a script.
    """
    # Path to the script
    script_path = 'tts.py'

    # Run the script as a subprocess
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    
    # Assert non-error exit or specific expected output
    assert result.returncode == 0, "Script should exit without error"
     # Check for '200 OK' in the output
    output_combined = result.stdout + result.stderr  # Combining stdout and stderr for comprehensive search
    assert "200 OK" in output_combined, "Output should contain '200 OK'"