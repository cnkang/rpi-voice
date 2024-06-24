import os
import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import httpx
from tts import TextToSpeech
from whisper import WhisperSTT

# Load environment variables
def initialize_env(load_env: bool = True) -> None:
    """
    Initialize environment variables for the application.

    Args:
        load_env (bool, optional): Whether to load environment variables from a .env file.
            Defaults to True.

    Raises:
        ValueError: If any of the required environment variables are missing.

    Returns:
        None
    """
    # Load environment variables from .env file
    if load_env:
        load_dotenv()

    # Required environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]

    # Check if any required environment variables are missing
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

    # Set default values for environment variables
    os.environ.setdefault("AZURE_API_VERSION", "2024-05-01-preview")
    os.environ.setdefault("VOICE_NAME", "zh-CN-XiaoxiaoMultilingualNeural")

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
    # Similar to earlier, creating client using API keys and checking environment variables
    return AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        http_client=httpx.AsyncClient(http2=True),
    )

# Function to transcribe speech to text
async def transcribe_speech_to_text(whisper_instance=None):
    whisper = whisper_instance or WhisperSTT()
    try:
        audio = await whisper.record_audio_vad()
        transcription = await whisper.transcribe_audio(audio)
        return transcription
    except Exception as e:
        logging.error("Speech-to-text conversion error: %s", e)
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

async def synthesize_and_play_speech(tscript):
    # Create and use instance of TextToSpeech
    tts_processor = TextToSpeech()
    try:
        await tts_processor.synthesize_speech(tscript)
    except Exception as e:
        logging.error("Error while synthesizing speech: %s", e)
        raise

async def main() -> None:
    """
    Main function to run the program asynchronously.
    """
    openai_client: Optional[AsyncAzureOpenAI] = await create_openai_client()
    if openai_client is None:
        return

    text_transcript: Optional[str] = await transcribe_speech_to_text()
    if text_transcript is None:
        return

    prompts = [
        {
            "role": "system",
            "content": (
                "Please respond naturally in the same language as the user, using human-like "
                "expressions and emotions. Synthesize the following user's input using SSML markup. "
                "Start with the appropriate SSML tags including the <speak> root tag with attributes "
                "version='1.0' and xmlns='http://www.w3.org/2001/10/synthesis', and ensure the "
                "xml:lang attribute is set to the appropriate response language. Use the <voice> tag "
                "with attributes name='{VOICE_NAME}'. Within the voice tag, adjust pitch, rate, and "
                "volume through <prosody> to convey emotions effectively, use <emphasis> to highlight "
                "important words, and <break> to add realistic pauses. Integrate natural elements like "
                "sighs, laughter, and exclamations using <audio> tags where appropriate to enhance realism. "
                "Ensure all tags are properly closed. Your responses should reflect understanding, empathy, "
                "and adaptability, focusing solely on the textual and emotional content without revealing "
                "your AI nature.".format(VOICE_NAME=VOICE_NAME)
            ),
        },
        {"role": "user", "content": text_transcript},
    ]

    response_text: Optional[str] = await interact_with_openai(openai_client, prompts)
    if response_text is None:
        return

    logging.info("OpenAI Response: %s", response_text)
    await synthesize_and_play_speech(response_text)

if __name__ == "__main__":
    initialize_env()
    asyncio.run(main())
