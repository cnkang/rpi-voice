from ../tts import TextToSpeech

async def test_synthesize_speech():
    tts = TextToSpeech()
    result = await tts.synthesize_speech("Test speech synthesis.")
    assert result is not None, "synthesize_speech should return a result"
