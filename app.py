import os
import asyncio
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from whisper import record_audio_vad, transcribe_audio
import numpy as np
from openai import AsyncAzureOpenAI

# Load .env file to environment
load_dotenv()

# Load voice preference from environment variables
preferred_voice = os.getenv("AZURE_VOICE_NAME", "zh-CN-XiaoxiaoMultilingualNeural")

async def create_azure_client():
    """ Initialize an Azure OpenAI client asynchronously. """
    return AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

def setup_speech_recognition():
    """ Setup Azure speech recognition configuration. """
    sub_key = os.getenv("AZURE_SPEECH_KEY")
    sub_region = os.getenv("AZURE_SPEECH_REGION")
    if not all([sub_key, sub_region]):
        raise EnvironmentError("Azure Speech service environment variables are not fully set.")
    return speechsdk.SpeechConfig(subscription=sub_key, region=sub_region)

def text_to_speech_conversion(text):
    """ Convert provided text to speech using Azure Speech SDK. """
    speech_config = setup_speech_recognition()
    speech_config.speech_synthesis_voice_name = preferred_voice
    audio_output = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
    synthesis_result = synthesizer.speak_ssml_async(to_ssml(text)).get()

    if synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized successfully.")
    elif synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = synthesis_result.cancellation_details
        print(f"Synthesis canceled due to: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")

def to_ssml(text):
    """ Transform plain text into SSML format """
    return text if '<speak' in text else f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>{text}</speak>"

async def interact_with_openai(client, messages):
    """ Invoke OpenAI API and retrieve response for given messages. """
    openai_response = await client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=messages,
        max_tokens=4096,
        temperature=0.7
    )
    return openai_response.choices[0].message.content if openai_response.choices else "No response was returned."

async def main():
    """ Main execution sequence. """
    azure_client = await create_azure_client()
    recorded_audio, audio_sample_rate = record_audio_vad()
    if recorded_audio is not None and np.any(recorded_audio):
        recognized_text = transcribe_audio(recorded_audio, audio_sample_rate)
        if recognized_text:
            messages = [
                {"role": "system", "content": f"""Please respond naturally in the same language, your responses should be empathetic and adaptive. Respond using SSML format, ensuring that the <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'> node is the root element and update the "lang" setting. Use the voice name '{preferred_voice}'. Additionally, make frequent use of SSML prosody tags to adjust the pitch, rate, and volume as needed to convey the appropriate emotional tone and enhance the expressiveness of the speech. When including multiple languages, ensure to use appropriate <lang> tags within the <voice> tags to specify each language section within the SSML to prevent synthesis errors and ensure compatibility with various speech synthesis services. Avoid nesting <voice> tags within another <voice> as this can lead to synthesis errors; instead, define each <voice> separately within the <speak> structure. If there are multiple languages in the output, please nest each <lang xml:lang='en-US'> tag properly within the appropriate <voice> tags, and output all at once. Ensure that all SSML tags, especially <prosody> tags, are correctly nested within the <voice> tags to prevent mismatched tag errors. Verify the output format follows the SSML standards and ensure all elements are correctly closed as required."""},
                {"role": "user", "content": recognized_text}
            ]
            response_text = await interact_with_openai(azure_client, messages)
            print("Processed response from AI:", response_text)
            ssml_formatted_response = to_ssml(response_text)
            text_to_speech_conversion(ssml_formatted_response)

if __name__ == "__main__":
    asyncio.run(main())
