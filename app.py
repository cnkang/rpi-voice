import httpx
import os
import asyncio
import logging
import concurrent.futures  # Import the concurrent.futures module
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from tts import TextToSpeech
from whisper import WhisperSTT

# Load environment variables
def initialize_env(load_env=True):
    if load_env:
        load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve the voice and other configurations from the .env file
VOICE_NAME = os.getenv("VOICE_NAME", "zh-CN-XiaoxiaoMultilingualNeural")
MODEL_NAME = os.getenv("MODEL_NAME")

async def create_openai_client():
    """Creates an async client for OpenAI using pre-loaded environment variables.
       Raises:
           ValueError: If any required configuration is missing.
    """
    # Dynamically load environment variables
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_API_VERSION")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    # Check if essential environment variables are missing
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY is required but missing.")
    if not api_version:
        raise ValueError("AZURE_API_VERSION is required but missing.")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is required but missing.")

    # Since all conditions are checked, it's safe to proceed
    return AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        http_client=httpx.AsyncClient(http2=True),
    )

async def transcribe_speech_to_text():
    """Converts speech to text using WhisperSTT and handles errors."""
    whisper = WhisperSTT()
    try:
        # Make sure to await both the recording and transcription processes
        audio = await asyncio.get_running_loop().run_in_executor(None, whisper.record_audio_vad)
        transcription = await whisper.transcribe_audio(audio)
        return transcription
    except Exception as e:
        logging.error("Speech-to-text conversion error: %s", str(e))
        return ""


async def interact_with_openai(client, prompts):
    """Send messages to OpenAI and get the response."""
    try:
        # Make sure prompts are serializable; typically, they should be.
        if not isinstance(prompts, list) or not all(isinstance(p, dict) for p in prompts):
            logging.error("Prompts are not in the correct format")
            return "Error in prompts format"
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages="",#prompts,
            max_tokens=4096,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        result = response.choices[0].message.content if response.choices else "No response returned."
        return result
    except Exception as e:
        logging.error("Error interacting with OpenAI: %s", str(e))
        return "Error in the AI response"


async def synthesize_and_play_speech(tscript):
    tts_processor = TextToSpeech()
    try:
        if tscript is None:
            logging.error("Text script is None, cannot synthesize speech")
            return

        stream = await tts_processor.synthesize_speech(tscript)
        if stream:
            await tts_processor.play_speech(stream)
        else:
            logging.error("Failed to synthesize speech or get valid audio stream")
    except Exception as e:
        logging.error("Error while synthesizing speech: %s", str(e))


async def main():
    """
    Orchestrate text and speech interactions with OpenAI.
    Ensure all asynchronous tasks are managed with 'await'.
    """
    openai_client = await create_openai_client()
    text_transcript = await transcribe_speech_to_text()  # Directly await the result
    if text_transcript:
        messages = [
            {
                "role": "system",
                "content": ("You are a helpful assistant. Please respond naturally in the language "
                            "used by the user, utilizing human-like expressions and emotions. "
                            "Show understanding, empathy, and adaptability. Avoid disclosing your AI identity.")
            },
            {
                "role": "user",
                "content": text_transcript
            },
        ]
        response_text = await interact_with_openai(openai_client, messages)  # Ensure interaction is awaited
        logging.info("OpenAI Response: %s", response_text)
        await synthesize_and_play_speech(response_text)  # Properly await this asynchronous call

if __name__ == "__main__":
    # Run the asynchronous main routine
    initialize_env()
    asyncio.run(main())
