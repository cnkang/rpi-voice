import httpx
import os
import asyncio
import logging
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from tts import TextToSpeech
from whisper import WhisperSTT

# Load environment variables
def initialize_env(load_env=True):
    if load_env:
        load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve the voice and other configurations from the .env file
VOICE_NAME = os.getenv("VOICE_NAME", "zh-CN-XiaoxiaoMultilingualNeural")
MODEL_NAME = os.getenv("MODEL_NAME")

class AudioStreamError(Exception):
    """Exception raised when the audio stream is invalid or synthesis fails."""
    pass

# Function to create an async OpenAI client
async def create_openai_client():
    """Creates an async client for OpenAI using pre-loaded environment variables.
       Raises:
           ValueError: If any required configuration is missing.
    """
    # Dynamically load environment variables
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_API_VERSION")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    # Check if essential environment variables are missing
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY is required but missing.")
    if not api_version:
        raise ValueError("AZURE_API_VERSION is required but missing.")
    if not azure_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is required but missing.")

    # Since all conditions are checked, it's safe to proceed
    return AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        http_client=httpx.AsyncClient(http2=True),
    )

# Function to transcribe speech to text using WhisperSTT
async def transcribe_speech_to_text(whisper_instance=None):
    """Converts speech to text using WhisperSTT and handles errors."""
    whisper = whisper_instance or WhisperSTT()
    try:
        # Make sure to await both the recording and transcription processes
        audio = await whisper.record_audio_vad()
        transcription = await whisper.transcribe_audio(audio)
        return transcription
    except Exception as e:
        logging.error("Speech-to-text conversion error: %s", str(e))
        return ""

# Function to interact with OpenAI
async def interact_with_openai(client, prompts):
    """Send messages to OpenAI and get the response, raise AssertionError on input errors."""
    try:
        # Ensure prompts are serializable; typically, they should be.
        if not isinstance(prompts, list) or not all(isinstance(p, dict) for p in prompts):
            error_message = "Prompts are not in the correct format: Should be a list of dictionaries."
            logging.error(error_message)
            raise AssertionError(error_message)

        # 检查每个prompt的结构是否合规
        required_keys = {'role', 'content'}
        valid_roles = {'system', 'user'}
        for prompt in prompts:
            if not required_keys <= prompt.keys():
                error_message = "Each prompt should contain 'role' and 'content'."
                logging.error(error_message)
                raise AssertionError(error_message)
            if prompt['role'] not in valid_roles:
                error_message = f"Role must be either 'system' or 'user', but got '{prompt['role']}'."
                logging.error(error_message)
                raise AssertionError(error_message)

        # Send the prompts to OpenAI and await the response
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompts,
            max_tokens=4096,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        if response.choices and response.choices[0]:
            result = response.choices[0].message.content
        else:
            result = "No response returned."

        return result
    except Exception as e:
        logging.error("Error interacting with OpenAI: %s", str(e))
        raise AssertionError(f"Error in the AI response: {str(e)}")

# Function to synthesize and play speech
async def synthesize_and_play_speech(tscript):
    # Create an instance of TextToSpeech to handle TTS operations
    tts_processor = TextToSpeech()
    try:
        # Check if the input text script is None, logging an error if so
        if tscript is None:
            logging.error("Text script is None, cannot synthesize speech")
            return

        # Use the TTS processor to synthesize the text into a streaming audio format
        stream = await tts_processor.synthesize_speech(tscript)
        
        # If the speech synthesis failed or returned no valid stream, log error and raise an exception
        if not stream:
            error_msg = "Failed to synthesize speech or get valid audio stream"
            logging.error(error_msg)
            raise AudioStreamError(error_msg)  # Ensure code raises an exception here if the stream is invalid

        # If valid stream is obtained, play the synthesized speech
        await tts_processor.play_speech(stream)
    except Exception as e:
        # Log any exceptions during the synthesis or playback process and rethrow the exception
        logging.error("Error while synthesizing speech: %s", str(e))
        raise  # Ensure the caught exception is rethrown


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
