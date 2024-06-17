from unittest.mock import patch
import httpx
import pytest
from tts import TextToSpeech

async def test_synthesize_speech():
    tts = TextToSpeech()
    result = await tts.synthesize_speech("Test speech synthesis.")
    assert result is not None, "synthesize_speech should return a result"

async def test_synthesize_speech_empty_string():
    tts = TextToSpeech()
    result = await tts.synthesize_speech("")
    assert result is None, "Empty speech synthesis should return None."

async def test_synthesize_speech_http_error():
    tts = TextToSpeech()
    with patch('httpx.AsyncClient.post', side_effect=httpx.HTTPStatusError("Error", request="dummy request", response="dummy response")) as mock_post:
        with pytest.raises(httpx.HTTPStatusError):
            await tts.synthesize_speech("This should fail")
