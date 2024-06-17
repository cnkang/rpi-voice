from unittest.mock import patch, AsyncMock
import httpx
import pytest
from httpx import Response, Request
from unittest.mock import patch
import pytest
from tts import TextToSpeech

async def test_synthesize_speech():
    """
    Tests the synthesis of speech from text in normal conditions to verify the proper functionality
    of the TextToSpeech class. Asserts that an AssertionError is raised with a specific message when 
    synthesizing speech fails.
    """
    # Initialize the TextToSpeech class instance
    tts = TextToSpeech()
    # Assert that an AssertionError is raised with specific message when synthesizing speech
    with pytest.raises(AssertionError, match=r"Failed to synthesize speech"):
        await tts.synthesize_speech("Test speech synthesis.")

async def test_synthesize_speech_empty_string():
    """
    Tests the TextToSpeech class to ensure it raises an AssertionError when attempting to synthesize 
    speech from an empty string. This test verifies that the class can handle edge cases where invalid 
    input (empty string) is given, and it ensures that the error messaging is consistent with synthesis failures.
    """
    tts = TextToSpeech()# Initialize the TextToSpeech class instance
    # Assert that an AssertionError is raised with specific message when synthesizing with an empty string
    with pytest.raises(AssertionError, match=r"Failed to synthesize speech"):
        await tts.synthesize_speech("")

async def test_synthesize_speech_http_error():   
    """
    Tests the error handling of the TextToSpeech class during an HTTP error. The test checks if the class
    correctly handles exceptions that occur during the HTTP POST request by raising a customized AssertionError.
    This ensures that the system is resilient and addresses potential errors during speech synthesis network requests.
    """
    tts = TextToSpeech() # Initialize the TextToSpeech class instance
    # Mock HTTP errors during the POST request and assert that an exception is properly handled
    with patch('httpx.AsyncClient.post', side_effect=httpx.HTTPStatusError("Error", request="dummy request", response="dummy response")), pytest.raises(AssertionError, match=r"Failed to synthesize speech"):
        await tts.synthesize_speech("This should fail but handle")

async def test_synthesize_speech_retry_logic():
    """
    Tests the retry logic of the TextToSpeech class to ensure that it appropriately retries the HTTP POST request
    when transient errors (specifically HTTP 500 errors) occur. This test verifies that the class can handle 
    temporary communication issues and still successfully synthesize speech after a number of retry attempts.

    The test sets up mocked HTTP responses: two failures followed by a success. The assertions check if:
    - The HTTP POST method is called the correct number of times (3 times in this case due to two failures and one success).
    - The synthesized speech, represented as a stream of bytes, is returned correctly after successful retrying.

    This ensures the resilience of the TextToSpeech system by confirming that it can recover from intermittent network failures.
    """
    tts = TextToSpeech() # Initialize the TextToSpeech class instance

    
    # Set the retry parameters to reduce test duration
    tts.max_retries = 3
    tts.retry_delay = 1

    # Create a mocked request and response
    mock_request = httpx.Request(method="POST", url=tts.construct_request_url())
    
    # Create a failed HTTP error and a successful response for testing
    http_error = httpx.HTTPStatusError(message="Error", request=mock_request, response=httpx.Response(status_code=500, request=mock_request))
    successful_response = httpx.Response(status_code=200, content=b"Success binary content for audio", request=mock_request)
    
    # Define side effects to simulate retry behavior on failures
    side_effects = [http_error, http_error, successful_response]

    with patch('httpx.AsyncClient.post', side_effect=side_effects) as mock_post:
        # Execute the function under test
        audio_stream = await tts.synthesize_speech("Testing retry logic.")
        
        # Verify that the `post` method was called three times as expected for retries
        assert mock_post.call_count == 3
        
        # Verify that the correct audio stream is returned after retries
        assert audio_stream is not None
