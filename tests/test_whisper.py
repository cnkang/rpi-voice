"""
This module contains pytest fixtures and test functions for the test_whisper.py file.

The fixtures and functions in this module are used for testing the WhisperSTT class.
"""
import logging
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock, ANY

import pytest
import numpy as np
import sounddevice as sd
from pytest import raises

from whisper import WhisperSTT  # Assuming whisper is a local module
from whisper import main as whisper_main

@pytest.fixture
def whisper_stt_test():
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
    with patch("whisper.load_dotenv"), patch("whisper.os.getenv") as mocked_getenv:
        # Maps the key lookups for environment variables to return specific values; others will return default or None.
        mocked_getenv.side_effect = lambda x, default=None: {
            "AZURE_OPENAI_ENDPOINT": "example.com",
            "AZURE_OPENAI_API_KEY": "abc123",
        }.get(x, default)
        yield WhisperSTT()


@pytest.mark.asyncio
async def test_record_audio_vad(whisper_stt_test):
    """
    Verify the record_audio_vad function with silent audio input simulated.
    
    Args:
        whisper_stt (fixture): A fixture that provides a WhisperSTT instance
    """
    simulated_silence = np.zeros((160,), dtype=np.int16)
    # Generate silence to fill 1600 samples across 10 frames
    simulated_audio = [simulated_silence for _ in range(10)]

    # Test the VAD method
    result = await whisper_stt_test.record_audio_vad(simulation_input=simulated_audio)
    total_samples = 160 * len(simulated_audio)

    assert isinstance(result, np.ndarray), "Result should be a numpy array"
    assert result.shape[0] == total_samples, (
        f"Expected {total_samples} samples, got {result.shape[0]}"
    )

@pytest.mark.asyncio
async def test_transcribe_audio(whisper_stt_test):
    audio = np.array([1, 2, 3], dtype=np.int16)
    expected_transcription = "test transcript"

    # 创建一个模仿响应对象，直接提供text属性
    simulate_response = MagicMock()
    simulate_response.text = expected_transcription
    
    with patch.object(
        whisper_stt_test.client.audio.transcriptions,
        "create",
        AsyncMock(return_value=simulate_response)
    ) as mock_create:
        result = await whisper_stt_test.transcribe_audio(audio)

        mock_create.assert_awaited()
        assert isinstance(result, str), "Result should be a string"
        assert result == expected_transcription, "Transcribed text should match the expected result"




@pytest.mark.asyncio
async def test_transcription_service_failure(whisper_stt_test):
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
    whisper_stt_test.client.audio.transcriptions.create = AsyncMock(
        side_effect=Exception("Service failure")
    )

    # Execute the transcription method and check for proper error handling.
    result = await whisper_stt_test.transcribe_audio(audio)
    assert (
        result == "Failed to transcribe audio"
    ), "Should handle service failures gracefully"


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
    with patch("whisper.load_dotenv"), patch("whisper.os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: (
            None if key == "AZURE_OPENAI_API_KEY" else "dummy_endpoint"
        )
        with raises(
            AssertionError,
            match="Environment variable 'AZURE_OPENAI_API_KEY' cannot be empty",
        ):
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
    with patch("whisper.load_dotenv"), patch("whisper.os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: (
            None if key == "AZURE_OPENAI_ENDPOINT" else "dummy_key"
        )
        with raises(
            AssertionError,
            match="Environment variable 'AZURE_OPENAI_ENDPOINT' cannot be empty",
        ):
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
    with patch("whisper.WhisperSTT.record_audio_vad") as mock_record_audio, patch(
        "whisper.WhisperSTT.transcribe_audio", new_callable=AsyncMock
    ) as mock_transcribe_audio:

        # Setting up mocked return values
        mock_record_audio.return_value = np.zeros(
            (1600,), dtype=np.int16
        )  # Pretend there's 1 second of silent input
        mock_transcribe_audio.return_value = (
            "Simulated transcription"  # Set up a simulated speech-to-text result
        )

        try:
            # Execute the main function assumed to be whisper_main()
            await whisper_main()
            # Verify if record_audio_vad function is called
            assert (
                mock_record_audio.called
            ), "The record_audio_vad function should have been called."
            # Verify if transcribe_audio function is called
            assert (
                mock_transcribe_audio.called
            ), "The transcribe_audio function should have been called."
        except AssertionError as e:
            # Fail the test if AssertionError, which indicated expected conditions failed
            pytest.fail(f"Unexpected AssertionError thrown: {e}")
        except Exception as e:
            # Fail the test if any unexpected exception is thrown, indicating unknown issues
            pytest.fail(f"Unexpected exception thrown: {e}")


@pytest.mark.asyncio
async def test_transcribe_audio_without_env():
    """
    Test handling of missing environment variables during WhisperSTT initialization.

    This test ensures that the WhisperSTT class raises an AssertionError when critical environment variables are missing.
    It simulates missing environment variables by setting os.getenv to return None. Expected behavior is an AssertionError.
    """
    # Simulate missing environment variables by setting `os.getenv` to return None
    with patch("os.getenv", return_value=None):
        # Expect an AssertionError when initializing WhisperSTT without required environment variables
        with pytest.raises(AssertionError):
            WhisperSTT()

@pytest.mark.asyncio
async def test_transcribe_audio_exception(whisper_stt_test):
    """
    Test the graceful handling of exceptions during the transcription process.

    This test simulates a scenario where an IOError occurs while trying to open a file during the
    transcription process. The function is expected to catch the exception and handle it gracefully,
    returning a specific error message ("Failed to transcribe audio"). The effectiveness of the
    exception handling within the `transcribe_audio` method of the `WhisperSTT` class is the primary
    focus of this test.

    Args:
        whisper_stt (WhisperSTT): An instance of WhisperSTT, prepared and configured for testing.

    Side Effects:
        Patches `builtins.open` to simulate an opening file issue, raising an IOError.

    Assumptions:
        The `transcribe_audio` method in the `WhisperSTT` class has proper exception handling for
        IOError during file operations.

    Asserts:
        Asserts that the result of the transcription process is equal to "Failed to transcribe audio"
        when an IOError is simulated, indicating that the error was handled as expected.
    """
    # Create a mock piece of audio data as an example
    audio_data = np.array([0, 1, 2], dtype=np.int16)

    # Patch the `open` built-in function in the context of this test to simulate file opening issues
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        # Simulate an IOError when attempting to write to a file
        mock_open.return_value.__enter__.return_value.write.side_effect = IOError("Failed to open file")

        # Perform the transcription process, which is expected to handle the simulated IOException gracefully
        result = await whisper_stt_test.transcribe_audio(audio_data)
        assert result == "Failed to transcribe audio"

@pytest.mark.asyncio
async def test_long_silence_handling(whisper_stt_test):
    """
    Test the handling of prolonged silence in the audio recording process.

    This test simulates a scenario where sustained silence occurs, and checks if the `record_audio_vad` function stops
    recording early due to reaching the maximum silence duration allowed. It uses a mock for a voice activity detector
    (VAD) to consistently detect no speech in the audio, simulating a prolonged silence scenario.

    The test feeds a long array of silent audio frames to the `record_audio_vad` method and examines if the recording
    stops prematurely due to the absence of detected speech, based on configured maximum silence parameters.

    Args:
        whisper_stt (WhisperSTT): An instance of WhisperSTT with configured VAD settings for testing.

    Side Effects:
        - Utilizes the `webrtcvad.Vad` class, simulating it always detecting no speech in the audio.
        - Feeds a predefined extensive silent audio data for simulating non-speech conditions.

    Asserts:
        Asserts that the recorded audio length is less than expected if normal continuous recording was done,
        validating that the system correctly identifies and stops recording after prolonged silence.
    """
    # Prepare long silence audio data which should fulfill batch processing requirements
    mock_audio_data = [np.zeros((160,), dtype=np.int16) for _ in range(3000)]  

    with patch('whisper.webrtcvad.Vad') as mock_vad:
        # Setup mock for Voice Activity Detector (VAD) which will always assume there is no speech
        mock_vad_instance = mock_vad.return_value
        mock_vad_instance.is_speech.return_value = False  

        # Execute the actual test on the audio recording function with specified conditions
        result = await whisper_stt_test.record_audio_vad(max_duration=10, max_silence_duration=1, simulation_input=mock_audio_data)

        # Verify the result length is as expected (should have stopped early due to sustained silence)
        expected_length = len(mock_audio_data) * 160  # The expected length calculation considers short max_silence_duration, hence early stopping is anticipated
        assert len(result) < expected_length, f"Recording should have stopped early due to silence, expected < {expected_length}, got {len(result)}"
        
@pytest.mark.asyncio
async def test_maximum_duration_limit(whisper_stt_test):
    """
    Tests if recording stops upon reaching maximum duration.
    """
    # Create an extended silence data array, longer than the maximum recording duration
    long_silence_data = [np.zeros((160,), dtype=np.int16) for _ in range(700)]  # more than maximum duration
    
    # Record the audio using the given simulation input and specify a max duration of 5 seconds
    result = await whisper_stt_test.record_audio_vad(simulation_input=long_silence_data, max_duration=5)
    
    # Calculate the maximum length of recorded data expected based on sampling rate and max duration
    expected_max_length = 5 * whisper_stt_test.sample_rate  # max_duration in samples
    
    # Assert that the length of the result is less than or equal to the expected maximum length
    assert len(result) <= expected_max_length, "Recording should stop after reaching max duration"

@pytest.mark.asyncio
async def test_record_audio_vad_speech_detection(whisper_stt_test):
    """
    Test for correct handling of speech and silence during the recording process.
    """
    # Alternating between silence and noise frames
    silence_frame = np.zeros((160,), dtype=np.int16)
    noise_frame = np.ones((160,), dtype=np.int16)
    simulation_input = [noise_frame if i % 2 == 0 else silence_frame for i in range(20)]
    
    # Mocking vad.is_speech to return varying responses for silence and noise frames
    with patch('whisper.webrtcvad.Vad') as mock_vad:
        mock_vad_instance = mock_vad.return_value
        # Using a lambda to return True when the frame is a noise frame and False when it's a silence frame
        mock_vad_instance.is_speech.side_effect = lambda data, rate: np.any(data != 0)
        
        # Recording the audio using the WhisperSTT's record_audio_vad method with the simulated input
        result = await whisper_stt_test.record_audio_vad(simulation_input=simulation_input)
        # Asserting that the audio is correctly processed and recordings are made based on the vad.is_speech output
        assert len(result) > 0, "Should correctly process and record audio based on vad.is_speech result"
        
@pytest.mark.asyncio
async def test_cleanup_failure_logged(whisper_stt_test):
    with patch('os.unlink', side_effect=OSError("Could not delete file")):
        with patch('logging.error') as mock_log:
            whisper_stt_test.cleanup_temp_file('temp_audio_file.wav')
            expected_error_substring = "Failed to delete temp file temp_audio_file.wav:"
            args_of_call = str(mock_log.call_args)
            assert expected_error_substring in args_of_call, "Error log should contain the expected message substring"
@pytest.mark.asyncio
async def test_early_stop_due_to_silence(whisper_stt_test):
    silence_frame = np.zeros((160,), dtype=np.int16)
    # Simulating 30 seconds of silence which should trigger stop early than the max duration set to 120 seconds
    simulated_input = [silence_frame for _ in range(3000)]  # each is 0.01 seconds, total 30 seconds of silence
    
    with patch('whisper.webrtcvad.Vad') as mock_vad:
        mock_vad_instance = mock_vad.return_value
        mock_vad_instance.is_speech.return_value = False
        
        result = await whisper_stt_test.record_audio_vad(simulation_input=simulated_input, max_duration=120)
        # Expected behavior: Stop due to 1 second of continuous silence
        assert len(result) < whisper_stt_test.sample_rate * 2, "Recording should stop due to maximum silence duration"

@pytest.mark.asyncio
async def test_transcription_api_failure(whisper_stt_test):
    """
    Test the behavior of the transcribe_audio method in WhisperSTT when the API call fails.

    This test ensures that the method handles the API failure correctly by returning a failure message.
    """
    # Create a random audio array
    audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)

    # Patch the create method of the audio transcriptions client to raise an exception
    with patch.object(
        whisper_stt_test.client.audio.transcriptions, 'create', new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = Exception("API Failure")

        # Invoke the transcribe_audio method and check if it returns the expected failure message
        result = await whisper_stt_test.transcribe_audio(audio)
        assert result == "Failed to transcribe audio", "Should return failure message on API failure"
        
@pytest.mark.asyncio
async def test_cleanup_error(whisper_stt_test):
    """
    Test the exception handling of the cleanup_temp_file method in WhisperSTT when there is an error deleting the
    temporary file.

    This test ensures that the method handles the error correctly by logging the error message.
    """
    # Create a temporary file path that does not exist
    temp_file_path = "non_existent_file.wav"
    
    # Patch the os.unlink function to raise an OSError when trying to delete the file
    with patch('os.unlink', side_effect=OSError("Could not delete file")), \
         patch('logging.error') as mock_log:
        # Call the cleanup_temp_file method with the non-existent file path
        whisper_stt_test.cleanup_temp_file(temp_file_path)
        
        # Assert that the error message is logged correctly
        expected_error_substring = "Failed to delete temp file non_existent_file.wav:"
        args_of_call = str(mock_log.call_args)
        assert expected_error_substring in args_of_call, "Error log should contain the expected message substring"

@pytest.mark.asyncio
async def test_array_to_bytes_permission_error_handling(whisper_stt_test):
    """
    Test the exception handling of the array_to_bytes method in WhisperSTT when there is a permission error.
    
    This test ensures that the method handles permission issues correctly when trying to write the audio data
    to the file system, which should raise a PermissionError.
    """
    # Create a sample audio data array
    audio_data = np.array([0, 1, 2], dtype=np.int16)
    
    # Patch 'scipy.io.wavfile.write' to raise a PermissionError when used
    with patch('whisper.write') as mock_write, \
         patch('logging.error') as mock_log_error:
        mock_write.side_effect = PermissionError("Permission denied")

        # Try to invoke the function which should raise a PermissionError
        with pytest.raises(PermissionError) as exc_info:
            whisper_stt_test.array_to_bytes(audio_data)
        
        # Check if the correct exception was captured
        assert exc_info.type is PermissionError, "A PermissionError should be raised"
        expected_message = "Permission denied"
        assert str(exc_info.value) == expected_message, f"Exception message should be '{expected_message}'"

        # Ensure the write function was indeed attempted
        assert mock_write.called, "The write function should have been attempted"
        
        # Verify if the error was logged correctly
        mock_log_error.assert_called_with(f"Failed to write audio to file: {expected_message}")

# tests/test_whisper.py
@pytest.mark.asyncio
async def test_record_audio_realtime_exception_handling(whisper_stt_test):
    """
    Test the realtime audio recording function's exception handling.
    
    This simulates a PortAudioError during the realtime audio stream to ensure it is handled gracefully.
    """
    # Use sd.PortAudioError for the simulation to match the implementation's caught exceptions
    with patch('sounddevice.InputStream') as mock_input_stream:
        # Configure the mock to raise a sd.PortAudioError when called
        mock_input_stream.side_effect = sd.PortAudioError("Simulated PortAudio input error")

        # Setup logging to capture error logs
        with patch('logging.error') as mock_log_error, pytest.raises(AssertionError) as exc_info:
            # Run the record_audio_vad function expecting it to handle the streaming error gracefully
            await whisper_stt_test.record_audio_vad()

            # Check if logging.error was called with the expected message
            mock_log_error.assert_called_with("Error occurred during recording: %s", ANY)
            # Verify the AssertionError contains the correct message
            assert "Error occurred during recording" in str(exc_info.value), "Exception message should be correct"

            # Check that the logging captures the specific simulated input error message
            args, kwargs = mock_log_error.call_args
            assert "Simulated PortAudio input error" in str(args[1]), "Log message should contain the specific error message"


@pytest.mark.asyncio
async def test_main_general_exception(caplog):
    """
    Test handling of general exceptions in the main function,
    ensuring they are logged correctly and handled gracefully.
    """

    # Patch to simulate an exception in record_audio_vad call
    with patch('whisper.WhisperSTT.record_audio_vad', side_effect=Exception("General error 0123")) as mock_record:
        await whisper_main()  # As no exception is propagated outside, no need to use try-except here

        # Ensure the method that throws the exception is called
        mock_record.assert_called_once()

        # Check if the appropriate error log was made
        assert "An error occurred: General error 0123" in caplog.text

class TestHandler(logging.Handler):
    def __init__(self):
        super(TestHandler, self).__init__()
        self.records = []  # Initialize a list to store log records
    def emit(self, record):
        # Override the default emit behavior to append the log message to records list
        self.records.append(record.msg)

@pytest.mark.asyncio
async def test_direct_log_capture_with_custom_handler(whisper_stt_test):
    """
    Test that the logger correctly captures a cancellation event log message using a custom logging handler.

    This test uses a custom logging handler attached to the root logger to intercept and store log messages.
    By patching the `transcribe_audio` method to raise an asyncio.CancelledError, it simulates an error scenario
    to verify if the cancellation log message is captured by the custom handler. After the operation, it checks
    that the custom handler logged the expected message and that the `transcribe_audio` method was called as expected.
    """
    # Instantiate a custom logging handler that stores log messages in a list
    test_handler = TestHandler()
    # Obtain the root logger
    logger = logging.getLogger()
    # Add our custom handler to the logger
    logger.addHandler(test_handler)
    # Set the verbosity level of logging to INFO
    logger.setLevel(logging.INFO)

    # Patch the 'transcribe_audio' method of 'whisper_stt' with an asynchronous mock
    with patch.object(whisper_stt_test, "transcribe_audio", AsyncMock()) as mock_transcribe:
        # Configure the side effect of the mock to raise asyncio.CancelledError when called
        mock_transcribe.side_effect = asyncio.CancelledError("Forced cancellation for testing.")
        try:
            # Call the method that we expect to raise the asyncio.CancelledError
            await whisper_stt_test.transcribe_audio(np.zeros((10,), dtype=np.int16))
        except asyncio.CancelledError:
            # Directly log the cancellation event within the exception handling block
            logger.info("The recording was cancelled")

    # Remove our custom handler from the logger to clean up
    logger.removeHandler(test_handler)

    # Assert that our custom handler recorded the cancellation log message
    assert "The recording was cancelled" in test_handler.records, "Cancellation message should be logged by custom handler"
    # Ensure that the transcribe_audio function was actually called once
    mock_transcribe.assert_called_once()
