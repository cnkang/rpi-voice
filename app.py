import os
import asyncio
import logging
import re
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import httpx
from tts import TextToSpeech
from whisper import WhisperSTT

dialogue_history = []
remaining_tokens = 0
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

def strip_tags_and_newlines(xml_str):
    """
    Strips all XML tags and newlines from the provided string and returns plain text.
    
    Args:
        xml_str (str): The string containing XML tags.

    Returns:
        str: The plain text string without any XML tags or newlines.
    """
    # First remove all XML/SSML tags
    no_tags = re.sub(r'<[^>]+>', '', xml_str)
    # Then remove all newlines
    clean_text = no_tags.replace('\n', '').replace('\r', '').replace(' ',"")
    return clean_text

def manage_dialogue_history(user_prompt, assistant_response):
    """
    Manages the dialogue history by adding the user prompt and assistant response to the history.
    Removes the oldest records from the history if the total length of the history exceeds the remaining tokens.
    
    Args:
        user_prompt (str): The user's prompt.
        assistant_response (str): The assistant's response.
    """
    global dialogue_history
    global remaining_tokens
    try:
        plain_text_content = strip_tags_and_newlines(assistant_response['content'])
        # Update assistant_response with plain text
        assistant_response['content'] = plain_text_content
        # Add new user prompt and assistant response to the history
        logging.info(
        "Adding to history. User prompt: %s, Assistant response: %s", 
        user_prompt, assistant_response
        )

        dialogue_history.append(user_prompt)
        dialogue_history.append(assistant_response)
        
        # Remove the oldest records from the history if the total length exceeds the remaining tokens
        while remaining_tokens - sum(len(p['content']) for p in dialogue_history) < 200:
            # Remove the oldest pair of records from the history to maintain a paired logic
            if len(dialogue_history) > 2:
                dialogue_history.pop(0)  # Remove the oldest user prompt
                dialogue_history.pop(0)  # Remove the oldest assistant response

    except Exception as e:
            logging.error("Error in manage_dialogue_history: %s", e)

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
async def transcribe_speech_to_text(whisper_instance: Optional[WhisperSTT] = None) -> str:
    """
    Transcribe speech to text using WhisperSTT.

    Args:
        whisper_instance (Optional[WhisperSTT]): An instance of WhisperSTT. If not provided,
            a new instance will be created.

    Returns:
        str: The transcribed text.

    Raises:
        None

    This function records audio using voice activity detection (VAD) and transcribes the audio
    using WhisperSTT. If whisper_instance is not provided, a new instance of WhisperSTT is created.
    The transcribed text is returned if the transcription is successful. If an exception occurs,
    an empty string is returned.
    """
    # Create an instance of WhisperSTT if not provided
    whisper = whisper_instance or WhisperSTT()
    
    try:
        # Record audio using VAD and transcribe it
        audio = await whisper.record_audio_vad()
        transcription = await whisper.transcribe_audio(audio)
        
        # Log the transcription
        logging.info("Transcription received: %s", transcription)
        
        # Return the transcription
        return transcription
    
    except Exception as e:
        # Log the error and return an empty string
        logging.error("Speech-to-text conversion error: %s", e)
        return ""

# Function to interact with OpenAI
async def interact_with_openai(client, prompts):
    """
    Send messages to OpenAI and get the response, raise AssertionError on input errors.

    Args:
        client (AsyncAzureOpenAI): An instance of AsyncAzureOpenAI.
        prompts (list[dict]): List of dictionaries containing 'role' and 'content'.
            Each dictionary should have keys 'role' and 'content', where 'role' should be
            either 'system', 'user', or 'assistant', and 'content' should be a string.

    Returns:
        str: The response from OpenAI.

    Raises:
        AssertionError: If prompts are not in the correct format or if any prompt does not have
            'role' and 'content' keys.
        AssertionError: If any prompt has an invalid 'role' value.
        AssertionError: If an error occurs during interaction with OpenAI.
    """
    global remaining_tokens
    try:
        # Check if prompts are serializable and of the correct format
        if not isinstance(prompts, list) or not all(isinstance(p, dict) for p in prompts):
            error_message = "Prompts are not in the correct format: Should be a list of dictionaries."
            logging.error(error_message)
            logging.error("Current prompts: %s", prompts)
            raise AssertionError(error_message)

        # Check if each prompt has the correct structure
        required_keys = {'role', 'content'}
        valid_roles = {'system', 'user', 'assistant'}
        for prompt in prompts:
            if not required_keys <= prompt.keys():
                error_message = "Each prompt should contain 'role' and 'content'."
                logging.error(error_message)
                raise AssertionError(error_message)
            if prompt['role'] not in valid_roles:
                error_message = (
                    f"Role must be 'system' or 'user' or 'assistant', but got '{prompt['role']}'."
                )
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
        logging.info("OpenAI response object: %s", response)

        # Process the response and return the result
        if response.choices and response.choices[0]:
            result = response.choices[0].message.content
            remaining_tokens = 128000 - response.usage.total_tokens
            logging.debug("Remaining tokens: %s", remaining_tokens)
            logging.info("Response text: %s", result)
        else:
            result = "No response returned."
            remaining_tokens = 0

        return result
    except Exception as e:
        logging.error("Error interacting with OpenAI: %s", str(e))
        raise AssertionError('Error in the AI response: %s', {str(e)}) from e

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
    global dialogue_history
    openai_client: Optional[AsyncAzureOpenAI] = await create_openai_client()
    if openai_client is None:
        return

    while True:
        try:
            # Transcribe speech to text
            text_transcript: Optional[str] = await transcribe_speech_to_text()
            if text_transcript is None:
                return

            # Create prompts for OpenAI
            system_prompt = {
                "role": "system",
                "content": _create_system_prompt()
            }
            user_prompt = {
                "role": "user",
                "content": text_transcript,
            }
            prompts = _create_prompts(system_prompt, user_prompt)
            print(prompts)

            # Interact with OpenAI
            response_text: Optional[str] = await interact_with_openai(openai_client, prompts)
            if response_text is None:
                logging.error("No valid response received from OpenAI.")
                return

            # Synthesize and play speech
            assistant_response = {
                "role": "assistant",
                "content": response_text,
            }
            logging.info("OpenAI Response: %s", response_text)
            await synthesize_and_play_speech(response_text)

            # Manage dialogue history
            manage_dialogue_history(user_prompt, assistant_response)
            
        except Exception as e:
            logging.error("Error in the main loop: %s", e)


def _create_system_prompt() -> str:
    """
    Create the system prompt for OpenAI.
    """
    return (
        f"Please respond naturally in the same language as the user, using human-like "
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
    )


def _create_prompts(system_prompt: dict, user_prompt: dict) -> list:
    """
    Create the prompts for OpenAI.
    """
    if len(dialogue_history) > 0:
        prompts = [system_prompt] + dialogue_history + [user_prompt]
    else:
        prompts = [system_prompt, user_prompt]
    return prompts

if __name__ == "__main__":
    initialize_env()
    asyncio.run(main())
