# tests/test_app.py
import os
import logging
from unittest.mock import patch, AsyncMock, call
import pytest
import openai
from app import (
    initialize_env,
    create_openai_client,
    transcribe_speech_to_text,
    AudioStreamError,
    interact_with_openai,
    synthesize_and_play_speech,
    main,
    _create_prompts,
    _create_system_prompt,
)

from whisper import WhisperSTT


@pytest.fixture(autouse=True)
def set_up_environment(monkeypatch):
    monkeypatch.setattr(
        os,
        "environ",
        {
            "AZURE_OPENAI_API_KEY": "mock-api-key",
            "AZURE_OPENAI_ENDPOINT": "mock-endpoint",
            "AZURE_API_VERSION": "2024-05-01-preview",
            "VOICE_NAME": "zh-CN-XiaoxiaoMultilingualNeural",
            "MODEL_NAME": "chat-model",
        },
    )


@pytest.mark.asyncio
async def test_create_openai_client_missing_env_vars():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(openai.OpenAIError) as exc_info:
            await create_openai_client()
        assert "Missing credentials" in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_openai_client():
    # Test creation of OpenAI client with all environment variables set
    client = await create_openai_client()
    assert client is not None


@pytest.mark.asyncio
async def test_interact_with_openai_error_handling():
    client = AsyncMock()
    with pytest.raises(AssertionError):
        await interact_with_openai(client, ["wrong format"])

    # Testing with incorrect role
    with pytest.raises(AssertionError):
        await interact_with_openai(client, [{"role": "alien", "content": "Hello."}])


async def test_transcription_successful() -> None:
    # Mocking the necessary objects and functions
    whisper_instance = AsyncMock()
    whisper_instance.transcribe_audio = AsyncMock(return_value="Hello, world!")
    temp_wav_path = "temp_audio.wav"

    with patch("app.WhisperSTT", return_value=whisper_instance), patch(
        "app.VoiceRecorder"
    ) as mock_voice_recorder:
        mock_voice_recorder.return_value.record_audio_vad = AsyncMock(
            return_value=[b"audio_data"]
        )
        mock_voice_recorder.return_value.array_to_wav_bytes.return_value = b"wav_data"

        # Test that the function doesn't return None
        assert await transcribe_speech_to_text(whisper_instance) is not None
        # Test that the function doesn't raise an exception
        await transcribe_speech_to_text(whisper_instance)


@pytest.mark.asyncio
async def test_transcribe_speech_to_text_error_handling(caplog):
    with caplog.at_level(logging.ERROR):
        mock_voice_recorder = AsyncMock()
        mock_whisper = AsyncMock(spec=WhisperSTT)
        mock_voice_recorder.record_audio_vad.return_value = [b"audio_data"]
        mock_voice_recorder.array_to_wav_bytes.return_value = b"wav_data"
        mock_whisper.transcribe_audio_stream.side_effect = AssertionError(
            "Transcription Error"
        )

        with patch("app.VoiceRecorder", return_value=mock_voice_recorder), patch(
            "app.WhisperSTT", return_value=mock_whisper
        ):
            with pytest.raises(AudioStreamError):
                await transcribe_speech_to_text()
            assert "Speech-to-text conversion error: Transcription Error" in caplog.text


@pytest.mark.asyncio
async def test_synthesize_and_play_speech_error_handling(caplog):
    with patch("app.TextToSpeech") as MockTextToSpeech:
        mock_tts_instance = AsyncMock()
        MockTextToSpeech.return_value = mock_tts_instance
        mock_tts_instance.synthesize_speech.side_effect = Exception("Synthesis Error")
        with pytest.raises(Exception):
            await synthesize_and_play_speech("Hello world")
        assert "Error while synthesizing speech: Synthesis Error" in caplog.text


@pytest.fixture
def setup_env_vars(monkeypatch):
    monkeypatch.setattr(
        os,
        "environ",
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "test-endpoint",
            "AZURE_API_VERSION": "test-version",
            "VOICE_NAME": "test-voice",
            "MODEL_NAME": "test-model",
            "LOOP_COUNT": "1",  # You can adjust this as needed for your tests
        },
    )
    yield


@pytest.mark.asyncio
async def test_missing_role_key_in_prompt():
    client = AsyncMock()
    prompts = [{"content": "Missing role key"}]
    with pytest.raises(AssertionError):
        await interact_with_openai(client, prompts)


@pytest.mark.asyncio
async def test_main_flow_success(setup_env_vars):
    mock_choice = AsyncMock()
    setattr(
        mock_choice, "message", type("obj", (object,), {"content": "Mocked response"})
    )

    mock_response = AsyncMock(choices=[mock_choice])
    openai_client_mock = AsyncMock()
    openai_client_mock.chat.completions.create.return_value = mock_response

    with patch(
        "app.create_openai_client", return_value=openai_client_mock
    ) as mock_client_creator, patch(
        "app.transcribe_speech_to_text", AsyncMock(return_value="test transcription")
    ) as mock_transcribe, patch(
        "app.synthesize_and_play_speech", AsyncMock()
    ) as mock_synth, patch("app.dialogue_history", []):
        await main(loop_count=1)

        assert mock_client_creator.call_count == 1
        assert mock_transcribe.call_count == 1
        mock_synth.assert_has_calls([call("Mocked response")])


@pytest.mark.asyncio
async def test_main_flow_no_openai_client(setup_env_vars):
    with patch("app.create_openai_client", AsyncMock(return_value=None)), patch(
        "app.transcribe_speech_to_text"
    ) as mock_transcribe, patch("app.synthesize_and_play_speech") as mock_synth:
        await main()
        mock_transcribe.assert_not_called()
        mock_synth.assert_not_called()


def test_load_env_true():
    with patch("app.load_dotenv") as mock_load_dotenv:
        initialize_env(load_env=True)
    mock_load_dotenv.assert_called_once()


def test_required_env_vars_present(monkeypatch):
    monkeypatch.setitem(os.environ, "AZURE_OPENAI_API_KEY", "test_api_key")
    monkeypatch.setitem(os.environ, "AZURE_OPENAI_ENDPOINT", "test_endpoint")
    initialize_env(load_env=False)  # Patch directly affects the module under test.


def test_required_env_vars_missing(monkeypatch):
    monkeypatch.setitem(os.environ, "AZURE_OPENAI_API_KEY", "")
    monkeypatch.setitem(os.environ, "AZURE_OPENAI_ENDPOINT", "")
    with pytest.raises(ValueError, match="Missing environment variables"):
        initialize_env(load_env=False)


@pytest.mark.asyncio
async def test_main_loop_count_none():
    with patch(
        "os.getenv",
        side_effect=lambda k: {
            "AZURE_OPENAI_API_KEY": "dummy_key",
            "AZURE_OPENAI_ENDPOINT": "dummy_endpoint",
            "AZURE_API_VERSION": "2024-05-01-preview",
        }.get(k, None),
    ):
        await main(loop_count=1)


@pytest.mark.asyncio
async def test_main_no_valid_response(caplog):
    # Mock the create_openai_client to return a valid client object
    openai_client_mock = AsyncMock()
    with patch(
        "app.create_openai_client", AsyncMock(return_value=openai_client_mock)
    ), patch("app.transcribe_speech_to_text", AsyncMock(return_value="Hello")), patch(
        "app.interact_with_openai", AsyncMock(return_value=None)
    ), patch("app.synthesize_and_play_speech") as mock_synth:
        # Call the main function with adjusted environment to not loop indefinitely
        await main(loop_count=1)

        # Check that the error was logged
        assert "No valid response received from OpenAI." in caplog.text
        # Ensure that synthesize_and_play_speech is not called
        mock_synth.assert_not_called()


def test_empty_dialogue_history():
    system_prompt = {"prompt": "System prompt"}
    user_prompt = {"prompt": "User prompt"}
    with patch("app.dialogue_history", []):
        expected_prompts = [system_prompt, user_prompt]
        actual_prompts = _create_prompts(system_prompt, user_prompt)
        assert actual_prompts == expected_prompts


def test_non_empty_dialogue_history():
    system_prompt = {"prompt": "System prompt"}
    user_prompt = {"prompt": "User prompt"}
    with patch(
        "app.dialogue_history",
        [{"prompt": "History prompt 1"}, {"prompt": "History prompt 2"}],
    ):
        dialogue_history = [
            {"prompt": "History prompt 1"},
            {"prompt": "History prompt 2"},
        ]
        expected_prompts = [system_prompt] + dialogue_history + [user_prompt]
        actual_prompts = _create_prompts(system_prompt, user_prompt)
        assert actual_prompts == expected_prompts


def test_create_system_prompt_with_valid_voice_name():
    with pytest.MonkeyPatch.context() as m:
        m.setattr("app.VOICE_NAME", "test_voice")
        prompt = _create_system_prompt()
        assert prompt.startswith(
            "Please respond naturally in the same language as the user"
        )
