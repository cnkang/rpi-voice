import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Load environment variables
load_dotenv()

def from_mic():
    """
    This function initializes the Azure Cognitive Services Speech SDK for recognizing
    speech input from the microphone. It performs the following actions:

    1. Loads the subscription key and region for Azure Cognitive Services from environment variables.
    2. Configures the speech recognition service with these credentials.
    3. Specifies a list of languages for automatic language detection.
    4. Initializes a SpeechRecognizer instance.
    5. Waits for the user to speak into the microphone.
    6. Processes the speech input and prints the recognized text or appropriate error messages.
    
    Make sure that the environment variables 'AZURE_SPEECH_KEY' and 'AZURE_SPEECH_REGION' 
    are set properly before running this function.

    Raises:
        EnvironmentError: If the necessary environment variables are not set.
    """
    subscription = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")

    if not subscription or not region:
        raise EnvironmentError("Environment variables for Azure Speech Service not set")
    speech_config = speechsdk.SpeechConfig(subscription=subscription, region=region)
    auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=[
        "en-US", "zh-CN", "zh-TW", "zh-HK"
    ])

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, auto_detect_source_language_config=auto_detect_config)
    print("Speak into your microphone.")
    result = speech_recognizer.recognize_once_async().get()

    handle_recognition_result(result)

def handle_recognition_result(result):
    """
    Handle the recognition result and print appropriate messages.

    Args:
        result: The result object from the speech recognition.
    """
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

# Run the function
from_mic()