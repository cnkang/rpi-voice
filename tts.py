"""
This module provides functionality to synthesize speech from text using the Azure
Speech Service API, handling various configurations and user interactions.
"""
import io
import asyncio
import os
import re
import logging
import httpx
from pydub import AudioSegment
from pydub.playback import play
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
        self.subscription = os.getenv("AZURE_SPEECH_KEY")
        self.region = os.getenv("AZURE_SPEECH_REGION")
        if not self.subscription or not self.region:
            raise EnvironmentError("Environment variables for Azure Speech Service not set")

        self.speechhost = self.region +  ".tts.speech.microsoft.com"

    async def get_azure_cognitive_access_token(self):
        """
        Obtains an access token from Azure Cognitive Services for authenticating API requests.

        Returns:
            A valid access token as a string.

        Raises:
            httpx.HTTPError: If the request fails.
            httpx.TimeoutException: If the request times out.
            Exception: If the response does not contain a valid token.
        """
        fetch_token_url = f"https://{self.region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        headers = {
            'Ocp-Apim-Subscription-Key': self.subscription
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(fetch_token_url, headers=headers, timeout=10)
            response.raise_for_status()
            token = response.text.strip()
        if not token:
            raise Exception("Token not found in response")
        return token
        

    async def synthesize_speech(self, text: str) -> bytes:
        """
        Synthesizes speech from a text string asynchronously, utilizing the Azure Speech API via RESTful requests.

        Args:
            text: A string to convert to speech.

        Returns:
            A byte-string containing the synthesized speech audio.

        Raises:
            RuntimeError: If speech synthesis fails.
        """
        if not text:
            raise ValueError("Text cannot be empty")

        try:
            ssml = self.convert_to_ssml(text)
            logging.debug("Converted SSML: %s", ssml)
            
            access_token = await self.get_azure_cognitive_access_token()
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-48khz-192kbitrate-mono-mp3",
                "Host": self.speechhost
            }
            
            endpoint = f"https://{self.speechhost}/cognitiveservices/v1"
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, headers=headers, content=ssml)
                response.raise_for_status()  # Raise an exception for HTTP error responses
                if(not response.content):
                    raise RuntimeError("Empty response content")
                else:
                    return response.content

        except httpx.HTTPStatusError as http_err:
            raise RuntimeError(f"HTTP error occurred during speech synthesis: {http_err}") from http_err
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
    audio_bytes = await tts.synthesize_speech("Hello, how are you! 你好吗")
    play(AudioSegment.from_file(io.BytesIO(audio_bytes)))
    audio_bytes = await tts.synthesize_speech(
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        "xml:lang='zh-CN'><voice name='zh-CN-XiaoxiaoMultilingualNeural'>一二三四五，数数真有趣！</voice></speak>"
    )
    play(AudioSegment.from_file(io.BytesIO(audio_bytes)))
    audio_bytes = await tts.synthesize_speech(
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        "xml:lang='en-US'><voice name='zh-CN-XiaoxiaoMultilingualNeural'><prosody rate='slow' "
        "pitch='+20%'>Welcome to our service,</prosody><prosody rate='medium' pitch='+10%'>where "
        "we offer <emphasis level='strong'>excellent customer experience</emphasis>.</prosody>"
        "<break time='500ms'/>How can I assist you <emphasis level='moderate'>today?哈哈哈</emphasis>"
        "</voice></speak>"
    )
    play(AudioSegment.from_file(io.BytesIO(audio_bytes)))

if __name__ == "__main__":
    asyncio.run(main())
