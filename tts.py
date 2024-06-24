"""
This module provides functionality to synthesize speech from text using the Azure
Speech Service API, handling various configurations and user interactions.
"""

import asyncio
import os
import re
import logging
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

class TextToSpeech:
    """
    Class for text-to-speech synthesis using Azure Speech Service.
    """
    def __init__(self):
        """
        Initializes the TextToSpeech class by loading environment variables, setting the
        voice name, and creating a SpeechConfig object for Azure Speech Service.
        Raises EnvironmentError if required environment variables are not set.
        """
        load_dotenv()
        self.voice_name = os.getenv("VOICE_NAME", "zh-CN-XiaoxiaoMultilingualNeural")
        subscription = os.getenv("AZURE_SPEECH_KEY")
        region = os.getenv("AZURE_SPEECH_REGION")
        if not subscription or not region:
            raise EnvironmentError("Environment variables for Azure Speech Service not set")
        self.tts_config = speechsdk.SpeechConfig(subscription=subscription, region=region)
        audio_output = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.tts_config, audio_config=audio_output)


    async def synthesize_speech(self, text: str) -> bytes:
        """
        Synthesizes speech from a text string asynchronously, utilizing the Azure Speech API.

        Args:
            text: A string to convert to speech.

        Returns:
            A byte-string containing the synthesized speech audio.

        Raises:
            RuntimeError: If speech synthesis fails or is cancelled.
        """
        if not text:
            raise ValueError("Text cannot be empty")

        try:
            ssml = self.convert_to_ssml(text)
            logging.debug("SSML: %s", ssml)
            result = self.synthesizer.speak_ssml_async(ssml).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data
            if result.reason == speechsdk.ResultReason.Canceled:
                if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print(f"Error details: {result.cancellation_details.error_details}")
                raise RuntimeError(f"Synthesis canceled: {result.cancellation_details.reason}")

            else:
                raise RuntimeError("Speech synthesis failed")
        except Exception as e:
            raise RuntimeError(f'Exception occurred during speech synthesis: {e}') from e

    def convert_to_ssml(self, text: str) -> str:
        """
        Ensures provided text is formatted according to SSML standards, including <speak>
        and <voice> tags.

        Args:
            text: The input text which may or may not be formatted in SSML.

        Returns:
            A properly formatted SSML string.

        Raises:
            ValueError: If the input text is empty.
        """

        # Regular expression pattern to match standard SSML format
        ssml_pattern = re.compile(
            r'^\s*<speak version=["\']1.0["\'] xmlns=["\']http://www\.w3\.org/2001/10/synthesis["\'] xml:lang=["\'][a-zA-Z-]+["\']>\s*<voice name=["\'][\w-]+["\']>.*</voice>\s*</speak>\s*$',
            re.DOTALL
        )

        # If the text is already in the proper SSML format, return it as is
        if ssml_pattern.match(text):
            return text

        # Otherwise, wrap the text in the SSML tags
        ssml_text = (
            # Opening <speak> tag with version and language attributes
            f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>"
            # Opening <voice> tag with voice name attribute
            f"<voice name='{self.voice_name}'>"
            # Text to be synthesized
            f"{text}"
            # Closing <voice> tag
            f"</voice>"
            # Closing <speak> tag
            f"</speak>"
        )
        return ssml_text


async def main() -> None:
    """
    Asynchronously runs the main function, synthesizing and playing speech.
    """
    tts = TextToSpeech()
    await tts.synthesize_speech("Hello, how are you! 你好吗")
    await tts.synthesize_speech(
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        "xml:lang='zh-CN'><voice name='zh-CN-XiaoxiaoMultilingualNeural'>一二三四五，数数真有趣！</voice></speak>"
    )
    await tts.synthesize_speech(
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        "xml:lang='en-US'><voice name='zh-CN-XiaoxiaoMultilingualNeural'><prosody rate='slow' "
        "pitch='+20%'>Welcome to our service,</prosody><prosody rate='medium' pitch='+10%'>where "
        "we offer <emphasis level='strong'>excellent customer experience</emphasis>.</prosody>"
        "<break time='500ms'/>How can I assist you <emphasis level='moderate'>today?哈哈哈</emphasis>"
        "</voice></speak>"
    )

if __name__ == "__main__":
    asyncio.run(main())
