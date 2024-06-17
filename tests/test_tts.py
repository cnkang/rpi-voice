from unittest.mock import patch
import httpx
import pytest
from tts import TextToSpeech

async def test_synthesize_speech():
    tts = TextToSpeech()
    with pytest.raises(AssertionError, match=r"Failed to synthesize speech"):
        await tts.synthesize_speech("Test speech synthesis.")

async def test_synthesize_speech_empty_string():
    tts = TextToSpeech()
    with pytest.raises(AssertionError, match=r"Failed to synthesize speech"):
        await tts.synthesize_speech("")

async def test_synthesize_speech_http_error():
    tts = TextToSpeech()
    with patch('httpx.AsyncClient.post', side_effect=httpx.HTTPStatusError("Error", request="dummy request", response="dummy response")), pytest.raises(AssertionError, match=r"Failed to synthesize speech"):
        await tts.synthesize_speech("This should fail but handle")
