import os
import asyncio
import logging
import concurrent.futures  # Import the concurrent.futures module
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from tts import TextToSpeech
from whisper import WhisperSTT

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve the voice and other configurations from the .env file
VOICE_NAME = os.getenv("VOICE_NAME", "zh-CN-XiaoxiaoMultilingualNeural")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")

async def create_openai_client():
    """Creates an async client for OpenAI using pre-loaded environment variables."""
    return AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

async def transcribe_speech_to_text():
    """Converts speech to text using WhisperSTT and handles errors."""
    whisper = WhisperSTT()
    try:
        # Using direct asyncio future with ThreadPoolExecutor for better management
        with concurrent.futures.ThreadPoolExecutor() as pool:
            audio = await asyncio.get_running_loop().run_in_executor(pool, whisper.record_audio_vad)
            transcription = await asyncio.get_running_loop().run_in_executor(pool, whisper.transcribe_audio, audio)
        return transcription
    except Exception as e:
        logging.error("Speech-to-text conversion error: %s", str(e))
        return ""

async def interact_with_openai(client, prompts):
    """Send messages to OpenAI and get the response."""
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompts,
            max_tokens=4096,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content if response.choices else "No response returned."
    except Exception as e:
        logging.error("Error interacting with OpenAI: %s", str(e))
        return "Error in the AI response"

def synthesize_and_play_speech(tscript):
    """Synthesizes speech from text and plays it using TextToSpeech class."""
    tts_processor = TextToSpeech()
    try:
        stream = tts_processor.synthesize_speech(tscript)
        if stream:
            tts_processor.play_speech(stream)
        else:
            logging.error("Failed to synthesize speech")
    except Exception as e:
        logging.error("Error while synthesizing speech: %s", str(e))

async def main():
    """Orchestrate text and speech interactions with OpenAI."""
    openai_client = await create_openai_client()
    text_transcript = await transcribe_speech_to_text()
    if text_transcript:
        messages = [
            {
                "role": "system",
                "content": ("You are a helpful voice assistant, please respond naturally in the same language as the user, using"
                            " human-like expressions and emotions. Your responses should reflect understanding, empathy, and adaptability,"
                            " focusing solely on the textual and emotional content without revealing your AI nature."),
            },
            {"role": "user", "content": text_transcript},
        ]
        response_text = await interact_with_openai(openai_client, messages)
        logging.info("OpenAI Response: %s", response_text)
        synthesize_and_play_speech(response_text)

if __name__ == "__main__":
    asyncio.run(main())
