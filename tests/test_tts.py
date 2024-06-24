import pytest
from unittest.mock import patch, MagicMock
import azure.cognitiveservices.speech as speechsdk

import tts as tts_module


@pytest.mark.asyncio
async def test_synthesize_speech_success():
    tts = tts_module.TextToSpeech()
    mock_result = MagicMock()
    mock_result.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
    with patch.object(tts.synthesizer, 'speak_ssml_async', return_value=MagicMock(get=MagicMock(return_value=mock_result))):
        try:
            await tts.synthesize_speech("Test speech synthesis.")
        except Exception as e:
            pytest.fail(f"Unexpected exception thrown: {e}")



@pytest.mark.asyncio
async def test_synthesize_speech_empty_string():
    """
    Tests to ensure that synthesizing speech from an empty string raises a ValueError.
    """
    tts = tts_module.TextToSpeech()
    with pytest.raises(ValueError, match="Text cannot be empty"):
        await tts.synthesize_speech("")


@pytest.mark.asyncio
async def test_synthesize_speech_canceled_error():
    """
    Tests exception handling when the speech synthesis is canceled due to an error.
    """
    tts = tts_module.TextToSpeech()
    mock_cancellation = MagicMock()
    mock_cancellation.reason = speechsdk.CancellationReason.Error
    mock_cancellation.error_details = "Error during synthesis"
    mock_result = MagicMock(reason=speechsdk.ResultReason.Canceled, cancellation_details=mock_cancellation)
    
    with patch.object(tts.synthesizer, 'speak_ssml_async', return_value=MagicMock(get=MagicMock(return_value=mock_result))):
        with pytest.raises(RuntimeError, match="Synthesis canceled: CancellationReason.Error"):
            await tts.synthesize_speech("This should fail due to cancellation")

@pytest.mark.asyncio
async def test_synthesize_speech_failure():
    """
    Tests that the synthesizing function raises an exception when the result reason is not 
    SynthesizingAudioCompleted and it's not Canceled.
    """
    tts = tts_module.TextToSpeech()
    mock_result = MagicMock(reason=speechsdk.ResultReason.DeletedVoiceProfile)
    with patch.object(tts.synthesizer, 'speak_ssml_async', return_value=mock_result):
        with pytest.raises(RuntimeError, match="Speech synthesis failed"):
            await tts.synthesize_speech("This should fail")


@pytest.mark.asyncio
async def test_main_function_normal_behavior():
    """
    Test the main function from the tts module to ensure it handles speech synthesis operations as expected.
    """
    mock_result = MagicMock()
    mock_result.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
    with patch('azure.cognitiveservices.speech.SpeechSynthesizer.speak_ssml_async', return_value=MagicMock(get=MagicMock(return_value=mock_result))) as mock_synthesize:
        try:
            await tts_module.main()
        except AssertionError as e:
            pytest.fail(f"Unexpected AssertionError thrown: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected exception thrown: {e}")
        assert mock_synthesize.called, "The synthesize_speech function should have been called."



@pytest.mark.asyncio
async def test_environmental_error():
    """
    Ensures that failure to set required environment variables raises an EnvironmentError.
    """
    with patch.dict('os.environ', {'AZURE_SPEECH_KEY': '', 'AZURE_SPEECH_REGION': ''}):
        with pytest.raises(EnvironmentError, match="Environment variables for Azure Speech Service not set"):
            tts_module.TextToSpeech()


@pytest.mark.asyncio
async def test_convert_to_ssml_already_formatted():
    """
    Tests that the convert_to_ssml method returns input text as-it-is if it's already in formatted SSML.
    """
    tts = tts_module.TextToSpeech()
    formatted_ssml = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='some-voice'>Hello, World!</voice></speak>"
    assert tts.convert_to_ssml(formatted_ssml) == formatted_ssml, "Should return the same SSML formatted text when already properly formatted."
