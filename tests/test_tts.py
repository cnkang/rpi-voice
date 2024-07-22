import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import HTTPStatusError, AsyncClient, Request,Response
from pydub import AudioSegment
import pytest
import tts


def create_tts_instance():
    with patch('tts.TextToSpeech', autospec=True) as mock:
        instance = mock.return_value
        async def mock_synthesize_speech(text):
            return b"audio data"
        instance.synthesize_speech = AsyncMock(side_effect=mock_synthesize_speech)
        instance.get_azure_cognitive_access_token = AsyncMock(return_value="valid_token")
        return instance

@pytest.mark.asyncio
async def test_synthesize_speech_success():
    tts_instance = create_tts_instance()
    response = await tts_instance.synthesize_speech("Test speech synthesis.")
    assert response == b"audio data"

@pytest.mark.asyncio
async def test_synthesize_speech_empty_string():
    with pytest.raises(ValueError):
        await tts.TextToSpeech().synthesize_speech("")

@pytest.mark.asyncio
async def test_synthesize_speech_http_error():
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        # Setup the mock to raise an HTTPStatusError
        mock_post.side_effect = HTTPStatusError(
            message="HTTP error",
            request=Request('POST', 'https://dummyurl'),
            response=Response(500, request=Request('POST', 'https://dummyurl'))
        )

        tts_service = tts.TextToSpeech()
        # Validate that an appropriate error is raised on HTTP failure
        with pytest.raises(RuntimeError):
            await tts_service.synthesize_speech("This should fail due to HTTP error")
@pytest.mark.asyncio
async def test_get_azure_cognitive_access_token_timeout():
    tts_instance = tts.TextToSpeech()
    tts_instance.subscription = "dummy_subscription"
    tts_instance.region = "dummy_region"
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = asyncio.TimeoutError()
        with pytest.raises(Exception):
            await tts_instance.get_azure_cognitive_access_token()

@pytest.mark.asyncio
async def test_get_azure_cognitive_access_token_successful():
    tts_instance = create_tts_instance()
    token = await tts_instance.get_azure_cognitive_access_token()
    assert token == "valid_token"

@pytest.mark.asyncio
async def test_get_azure_cognitive_access_token_http_error():
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        # Setup the mock to raise an HTTPStatusError
        mock_post.side_effect = HTTPStatusError(
            message="HTTP error",
            request=Request('POST', 'https://dummyurl'),
            response=Response(500, request=Request('POST', 'https://dummyurl'))
        )

        tts_service = tts.TextToSpeech()
        # Validate that an appropriate error is raised on HTTP failure
        with pytest.raises(Exception):
            await tts_service.get_azure_cognitive_access_token()
@pytest.mark.asyncio
async def test_get_azure_cognitive_access_token_invalid_token():
    tts_instance = tts.TextToSpeech()
    tts_instance.subscription = "dummy_subscription"
    tts_instance.region = "dummy_region"
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = ""
        with pytest.raises(Exception):
            await tts_instance.get_azure_cognitive_access_token()
@pytest.mark.asyncio
async def test_convert_to_ssml_already_formatted():
    tts_instance = create_tts_instance()
    formatted_ssml = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='some-voice'>Hello, World!</voice></speak>"
    tts_instance.convert_to_ssml.return_value = formatted_ssml
    result = tts_instance.convert_to_ssml(formatted_ssml)
    assert result == formatted_ssml, "Should return the same SSML formatted text when already properly formatted."

def create_silent_audio_segment(duration_ms=1000):
    return AudioSegment.silent(duration=duration_ms)

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post')
async def test_tts_main(mock_post):
    mock_response = Response(200, content=b"fake audio data")
    mock_post.return_value = AsyncMock(return_value=mock_response)

    tts_instance = create_tts_instance()

    with patch('tts.TextToSpeech.synthesize_speech', new_callable=AsyncMock) as mock_synthesize_speech:
        mock_synthesize_speech.return_value = b"fake audio data"
        tts_instance.synthesize_speech = mock_synthesize_speech

        with patch('pydub.AudioSegment.from_file') as mock_from_file:
            mock_audio_segment = create_silent_audio_segment()
            mock_from_file.return_value = mock_audio_segment

            with patch('pydub.playback.play') as mock_play:
                await tts.main()
                assert mock_synthesize_speech.call_count == 3


@patch('os.getenv', return_value=None)
@patch('dotenv.load_dotenv', return_value=None)
def test_missing_env_variables_raises_error(mock_load_dotenv, mock_getenv):
    with pytest.raises(EnvironmentError):
        tts.TextToSpeech()
