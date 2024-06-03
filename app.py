import os
import asyncio
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from openai import AsyncAzureOpenAI

# Load environment variables
load_dotenv()

# Retrieve the preferred voice name from environment, defaulting to "zh-CN-XiaoxiaoMultilingualNeural"
voice_name = os.getenv('VOICE_NAME', "zh-CN-XiaoxiaoMultilingualNeural")

async def create_client():
    """
    Asynchronously creates a client for the Azure OpenAI service.
    """
    return AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

async def recognize_speech():
    """
    Captures audio from the default microphone and recognizes speech using Azure Cognitive Services.
    """
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    print("Listening...")
    result = await recognizer.recognize_once_async()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: ", cancellation_details.reason)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: ", cancellation_details.error_details)
    
    return ""

async def call_openai_chat(client, messages):
    """
    Sends messages to the Azure OpenAI chat model and awaits the response.
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
    """
    return f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>{text}</speak>"

async def main():
    """
    Main asynchronous function to handle the creation of the client, capturing and transcribing audio, sending messages, and synthesizing responses.
    """
    client = await create_client()
    recognized_text = await recognize_speech()

    if recognized_text:
        messages = [
            {"role": "system", "content": f"Please respond naturally in the same language as the user, using human-like expressions and emotions. Begin with the <voice> tag using the voice named '{voice_name}'. Ensure all tags are properly closed. Your responses should reflect understanding, empathy, and adaptability, focusing solely on the textual and emotional content without revealing your AI nature."},
            {"role": "user", "content": recognized_text}
        ]
        response_text = await call_openai_chat(client, messages)
        print("AI Response:", response_text)
        ssml_response = convert_to_ssml(response_text)
        text_to_speech(ssml_response)

if __name__ == '__main__':
    asyncio.run(main())
