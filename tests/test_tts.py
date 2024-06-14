# tests/test_tts.py
import pytest
from tts import TextToSpeech
import httpx

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
    with pytest.raises(httpx.HTTPStatusError):
        # Simulate an HTTP error in the request
        async with httpx.AsyncClient() as client:
            client.post = pytest.raises(httpx.HTTPStatusError("Failed request"))
            await tts.synthesize_speech("This should fail")
