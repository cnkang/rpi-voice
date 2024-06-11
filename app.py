import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from tts import TextToSpeech
from whisper import WhisperSTT
# Load environment variables
load_dotenv()

# Retrieve the voice name from the .env or use default
voice_name = os.getenv("VOICE_NAME", "zh-CN-XiaoxiaoMultilingualNeural")
async def create_openai_client():
    """Creates an async client for OpenAI using environment variables."""
    return AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

async def transcribe_speech_to_text():
    """Converts speech to text using WhisperSTT and handles errors."""
    try:
        whisper = WhisperSTT()
        loop = asyncio.get_running_loop()
        
        # Use run_in_executor to run synchronous code in a separate thread
        audio = await loop.run_in_executor(None, whisper.record_audio_vad)
        transcription = await loop.run_in_executor(None, whisper.transcribe_audio, audio)
        
        return transcription
    except Exception as e:
        print("Speech-to-text conversion error:", str(e))
        return ""

async def interact_with_openai(client, prompts):
    """Sends prompts to OpenAI and returns the response."""
    try:
        result = await client.chat.completions.create(
            model_name=os.getenv("MODEL_NAME"),
            messages=prompts,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
    )
        return result.choices[0].message.content if result.choices else "No response received."
    except Exception as e:
        print("Error querying OpenAI:", str(e))
        return "Error while querying OpenAI."


def synthesize_and_play_speech(tscript):
    """Synthesizes speech from text and plays it using TextToSpeech class."""
    tts_processor = TextToSpeech()
    stream = tts_processor.synthesize_speech(tscript)
    if stream:
        tts_processor.play_speech(stream)
    else:
        print("Error while synthesizing speech")

async def main():
    """Orchestrates the text and speech interactions with OpenAI."""
    openai_client = await create_openai_client()
    text_transcript = await transcribe_speech_to_text()
    if text_transcript:
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful voice assistant. Please respond naturally in the same language as the user, using human-like expressions and emotions. Your responses should reflect understanding, empathy, and adaptability, focusing solely on the textual and emotional content without revealing your AI nature.",
            },
            {"role": "user", "content": text_transcript},
        ]
        response_text = await interact_with_openai(openai_client, messages)
        print("OpenAI Response:", response_text)
        synthesize_and_play_speech(response_text)

if __name__ == "__main__":
    asyncio.run(main())
