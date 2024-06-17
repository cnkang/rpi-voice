import numpy as np
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pytest import raises

from whisper import WhisperSTT
from whisper import main as whisper_main

@pytest.fixture
def whisper_stt():
    """
    Pytest fixture to provide a WhisperSTT instantiation with mock environment variables.
    Provides a WhisperSTT instance with mocked environment variables for testing.
    
    This fixture uses patching to simulate the loading of environment variables necessary for
    WhisperSTT configuration. It provides a default setup where "AZURE_OPENAI_ENDPOINT" and
    "AZURE_OPENAI_API_KEY" are predefined, allowing any test that requires a WhisperSTT instance
    to use this fixture without the need to setup real environment variables.
    
    This fixture ensures that environment variables needed for WhisperSTT configuration are simulated,
    allowing tests to function without actual environment setup. The 'AZURE_OPENAI_ENDPOINT' and
    'AZURE_OPENAI_API_KEY' are mocked with dummy data.
    Yields:
        WhisperSTT: An instance of the WhisperSTT class with mocked environment variables.
        WhisperSTT: An instance of the WhisperSTT with configurations simulated via mocked environment variables.
    """
    with patch('whisper.load_dotenv'), patch('whisper.os.getenv') as mocked_getenv:
        # Maps the key lookups for environment variables to return specific values; others will return default or None.
        mocked_getenv.side_effect = lambda x, default=None: {"AZURE_OPENAI_ENDPOINT": "example.com", "AZURE_OPENAI_API_KEY": "abc123"}.get(x, default)
        yield WhisperSTT()

@pytest.mark.asyncio
async def test_record_audio_vad(whisper_stt):
    """
    Test the record_audio_vad function using simulated audio input.
    Validates the behavior of the record_audio_vad method with silent audio input.

    This test function checks if the `record_audio_vad` method correctly processes a simulated 
    silence input and returns expected results. It provides a simplistic test case scenario where 
    only silent audio data is fed into the function, to assert its basic functionality and output's integrity.
    
    This async test mimics silence as input to verify if the method manages such data correctly and
    ensures it returns a numpy array of expected dimensions representing audio data.
    Args:
        whisper_stt: Fixture that provides a WhisperSTT instance configured with mocked environment vars.
        whisper_stt (Fixture): Provides a configured WhisperSTT instance.

    Asserts:
        Confirm that the result is a numpy.ndarray.
        Confirm that the shape of the result array matches the input audio duration (1600 samples, assumed one second).
        Checks the type and shape of the output to confirm method integrity.
    """
    simulated_silence = np.zeros((1600,), dtype=np.int16)  # Represents one second of silence, assuming 1600 samples = 1 second.
    simulated_audio = [simulated_silence]  # Wrapping the silent data as a list, as expected input.
    result = whisper_stt.record_audio_vad(simulation_input=simulated_audio)
    assert isinstance(result, np.ndarray), "Result should be a numpy array"
    assert result.shape[0] == 1600, "The shape of the result array should match the input audio duration."

@pytest.mark.asyncio
async def test_transcribe_audio(whisper_stt):
    """
    Test the functionality of the transcribe_audio method.

    This test verifies that the `transcribe_audio` method of the WhisperSTT class can process
    input audio data and return the correct transcription. It uses patching to mock the behavior
    of the Azure OpenAI service's transcription creation method, ensuring the method under test
    relies on a controlled and predictable service response.

    Ensures correct transcription of audio data using a mocked transcription service.

    This test patches the transcription service to return a fixed transcript response.
    It verifies that the `transcribe_audio` method functions as expected by confirming it
    returns the correct string based on the audio input.
    Args:
        whisper_stt: Fixture that provides a WhisperSTT instance configured with mocked environment vars.
        whisper_stt (Fixture): Provides a configured WhisperSTT instance.

    Asserts:
        Assert that the result is a string.
        Assert that the transcribed text matches the expected output "test transcript".
        Checks if the transcribed result is a string and matches the expected content.
    """
    audio = np.array([1, 2, 3], dtype=np.int16)  # Sample audio input data.
    with patch.object(whisper_stt.client.audio.transcriptions, 'create', AsyncMock(return_value=MagicMock(text="test transcript"))):
        result = await whisper_stt.transcribe_audio(audio)
        assert isinstance(result, str), "Result should be a string"
        assert result == "test transcript", "Transcribed text should match the expected result."

@pytest.mark.asyncio
async def test_transcription_service_failure(whisper_stt):
    """
    Test the behavior of the transcribe_audio method when transcription service fails.
    Tests the error handling of the transcribe_audio method in case of service failure.

    This test evaluates the robustness of the `transcribe_audio` method by simulating a scenario where
    the transcription service throws an exception, representing a failure in the service. It checks whether
    the method properly handles the failure by returning a specific message indicating inability to transcribe
    the audio.

    Simulates a service failure via an exception during the transcription process to ensure
    that the method handles such scenarios as expected by returning a failure message.
    Args:
        whisper_stt: Fixture that provides a WhisperSTT instance configured with mocked environment vars.
        whisper_stt (Fixture): Provides a WhisperSTT instance setup with mocked environment variables.

    Asserts:
        Assert that the result matches the expected failure handling message.
        Ensures that the service failure is handled properly and returns an appropriate message.
    """
    audio = np.array([1, 2, 3], dtype=np.int16)
    audio = np.array([1, 2, 3], dtype=np.int16)  # Sample audio input.
    # Simulate a service failure by raising an exception when calling the transcription service.
    whisper_stt.client.audio.transcriptions.create = AsyncMock(side_effect=Exception("Service failure"))
    
    # Execute the transcription method and check for proper error handling.
    result = await whisper_stt.transcribe_audio(audio)
    assert result == "Failed to transcribe audio", "Should handle service failures gracefully"

@pytest.mark.asyncio
async def test_openai_api_key_missing():
    """
    Test the initialization of WhisperSTT when the 'AZURE_OPENAI_API_KEY' environment variable is missing.

    This test verifies that the WhisperSTT initialization process properly identifies and raises an error
    when the essential 'AZURE_OPENAI_API_KEY' environment variable is not set. The system should notify 
    the user to ensure all required configurations are provided during setup.

    It mocks the environment variable retrieval process to simulate the absence of the 'AZURE_OPENAI_API_KEY' variable,
    and checks if the correct assertion error, indicating a missing variable, is raised.

    Args:
        None

    Asserts:
        Asserts that an AssertionError with a specific message is raised when the 'AZURE_OPENAI_API_KEY' is missing.
    """
    # Mock the environment variable retrieval to simulate a missing API key
    with patch('whisper.load_dotenv'), patch('whisper.os.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: None if key == "AZURE_OPENAI_API_KEY" else "dummy_endpoint"
        with raises(AssertionError, match="Environment variable 'AZURE_OPENAI_API_KEY' cannot be empty"):
            WhisperSTT()

@pytest.mark.asyncio
async def test_openai_endpoint_missing():
    """
    Test the initialization of the WhisperSTT instance when the Azure OpenAI endpoint environment variable is missing.

    This test checks that the WhisperSTT class correctly raises an error during initialization if the 'AZURE_OPENAI_ENDPOINT'
    environment variable is not set. This ensures that the application verifies the presence of critical configuration 
    before proceeding, which is essential for the system to function correctly with the Azure OpenAI service.

    The test uses mocking techniques to simulate the scenario where the 'AZURE_OPENAI_ENDPOINT' environment variable is missing.
    It confirms if the proper assertions are raised with a specific error message, alerting about the absence of the required
    environment variable.

    Args:
        None

    Asserts:
        Asserts that an AssertionError is raised with a specific message when the 'AZURE_OPENAI_ENDPOINT' environment variable is missing.
    """
    # Assuming the 'AZURE_OPENAI_API_KEY' is set successfully in this part
    with patch('whisper.load_dotenv'), patch('whisper.os.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: None if key == "AZURE_OPENAI_ENDPOINT" else "dummy_key"
        with raises(AssertionError, match="Environment variable 'AZURE_OPENAI_ENDPOINT' cannot be empty"):
            WhisperSTT()

@pytest.mark.asyncio
async def test_whisper_main_integration():
    """
    Tests the main function from the whisper module to ensure that it can run without throwing an exception
    when properly configured.

    This test function will run the main function inside the whisper module, watching for any exceptions,
    including AssertionError or any other unexpected errors that may indicate a problem in the main execution flow
    or integration issues.
    """
    # Mock relevant functions to avoid external dependencies while testing
    with patch('whisper.WhisperSTT.record_audio_vad') as mock_record_audio, \
         patch('whisper.WhisperSTT.transcribe_audio', new_callable=AsyncMock) as mock_transcribe_audio:
        
        # Setting up mocked return values
        mock_record_audio.return_value = np.zeros((1600,), dtype=np.int16)  # Pretend there's 1 second of silent input
        mock_transcribe_audio.return_value = 'Simulated transcription'  # Set up a simulated speech-to-text result

        try:
            # Execute the main function assumed to be whisper_main()
            await whisper_main()
            # Verify if record_audio_vad function is called
            assert mock_record_audio.called, "The record_audio_vad function should have been called."
            # Verify if transcribe_audio function is called
            assert mock_transcribe_audio.called, "The transcribe_audio function should have been called."
        except AssertionError as e:
            # Fail the test if AssertionError, which indicated expected conditions failed
            pytest.fail(f"Unexpected AssertionError thrown: {e}")
        except Exception as e:
            # Fail the test if any unexpected exception is thrown, indicating unknown issues
            pytest.fail(f"Unexpected exception thrown: {e}")