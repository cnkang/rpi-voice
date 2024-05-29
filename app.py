import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import azure.cognitiveservices.speech as speechsdk

# Load environment variables
load_dotenv()

# Retrieve the preferred voice name from environment, defaulting to "zh-CN-XiaoxiaoMultilingualNeural"
voice_name = os.getenv('VOICE_NAME', "zh-CN-XiaoxiaoMultilingualNeural")

async def create_client():
    """
    Asynchronously creates a client for the Azure OpenAI service.

    Returns:
        AsyncAzureOpenAI: An instance of the Azure OpenAI client configured with API keys and endpoint.
    """
    return AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

async def call_openai_chat(client, messages):
    """
    Sends messages to the Azure OpenAI chat model and awaits the response.

    Args:
        client (AsyncAzureOpenAI): The Azure OpenAI client.
        messages (list): A list of message dictionaries defining the role and content.

    Returns:
        str: The content of the response message or a default message if no response is generated.
    """
    response = await client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content if response.choices else "No response generated."

def text_to_speech(ssml):
    """
    Converts SSML text to speech using the Azure Cognitive Services Speech SDK.

    Args:
        ssml (str): The SSML string to be synthesized.
    """
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
    speech_config.speech_synthesis_voice_name = voice_name

    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for SSML.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: ", cancellation_details.reason)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: ", cancellation_details.error_details)

def convert_to_ssml(text):
    """
    Converts plain text to SSML format.

    Args:
        text (str): Plain text to be converted.

    Returns:
        str: Text wrapped in SSML tags.
    """
    return f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>{text}</speak>"

async def main():
    """
    Main asynchronous function to handle the creation of the client, sending messages, and synthesizing responses.
    """
    client = await create_client()
    messages = [
        {"role": "system", "content": f"Please respond naturally in the same language as the user, using human-like expressions and emotions. Begin with the <voice> tag using the voice named '{voice_name}'. Ensure all tags are properly closed. Your responses should reflect understanding, empathy, and adaptability, focusing solely on the textual and emotional content without revealing your AI nature."},
        {"role": "user", "content": "Hello, how are you doing？"}
    ]
    response_text = await call_openai_chat(client, messages)
    print("AI Response:", response_text)
    ssml_response = convert_to_ssml(response_text)
    text_to_speech(ssml_response)

if __name__ == '__main__':
    asyncio.run(main())
