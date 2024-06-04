import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import azure.cognitiveservices.speech as speechsdk

# Load environment variables
load_dotenv()

# Retrieve the preferred voice name from environment, defaulting to "zh-CN-XiaoxiaoMultilingualNeural"
voice_name = os.getenv("AZURE_VOICE_NAME", "zh-CN-XiaoxiaoMultilingualNeural")


async def create_client():
    """Asynchronously creates a client for the Azure OpenAI service."""
    return AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )


def get_speech_config():
    """
    Configure the speech recognition service.
    """
    subscription = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not (subscription and region):
        raise EnvironmentError("Environment variables for Azure Speech Service not set")
    return speechsdk.SpeechConfig(subscription=subscription, region=region)


def get_auto_detect_language_config():
    """
    Specify a list of languages for automatic detection.
    """
    return speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=[
        "en-US", "zh-CN", "zh-TW", "zh-HK"
    ])


def recognize_speech(speech_recognizer):
    """
    Recognize speech input from the microphone.
    """
    print("Speak now.")
    result = speech_recognizer.recognize_once_async().get()
    return process_speech_result(result)


def process_speech_result(result):
    """
    Process speech recognition result.
    """
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"Recognized: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print(f"No speech recognized: {result.no_match_details}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print(f"Recognition canceled: {result.cancellation_details.reason}")
        if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error: {result.cancellation_details.error_details}")
    return ""


async def speech_to_text():
    """
    Convert speech to text.
    """
    try:
        speech_config = get_speech_config()
        language_config = get_auto_detect_language_config()
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, auto_detect_source_language_config=language_config)
        return recognize_speech(recognizer)
    except EnvironmentError as e:
        print(str(e))
        return ""


async def call_openai(client, messages):
    """
    Send messages to OpenAI and get response.
    """
    response = await client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content if response.choices else "No response returned."


def text_to_speech(text):
    """
    Convert text to speech.
    """
    speech_config = get_speech_config()
    speech_config.speech_synthesis_voice_name = voice_name
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_ssml_async(convert_to_ssml(text)).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print(f"Synthesis canceled: {result.cancellation_details.reason}")
        if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error: {result.cancellation_details.error_details}")


def convert_to_ssml(text):
    """
    Convert text to SSML.
    """
    return f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>{text}</speak>"


async def main():
    """
    Main function to sequence operations.
    """
    client = await create_client()
    recognized_text = await speech_to_text()

    if recognized_text:
        messages = [
            {
                "role": "system",
                "content": f"Please respond naturally in the same language as the user, using human-like expressions and emotions. Begin with the <voice> tag using the voice named '{voice_name}'. Ensure all tags are properly closed. Your responses should reflect understanding, empathy, and adaptability, focusing solely on the textual and emotional content without revealing your AI nature.",
            },
            {"role": "user", "content": recognized_text},
        ]
        response_text = await call_openai(client, messages)
        print("AI Response:", response_text)
        ssml_response = convert_to_ssml(response_text)
        text_to_speech(ssml_response)


if __name__ == "__main__":
    asyncio.run(main())
