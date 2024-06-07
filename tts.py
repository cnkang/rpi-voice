import os
import requests
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Prepare the headers and payload for the POST request
headers = {
    'api-key': os.getenv("AZURE_OPENAI_API_KEY", ''),
    'Content-Type': 'application/json',
}

data = {
    "model": "tts-1-hd",
    "input": "Today is a wonderful day to build something people love!",
    "voice": "alloy"
}

# Construct the URL for the request
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", '')
api_version = '2024-02-15-preview'  # Update as needed
deployment_name = os.getenv("TTS_MODEL_NAME", 'tts-1-hd')  # Update with your actual deployment name
url = f"{endpoint}/openai/deployments/{deployment_name}/audio/speech?api-version={api_version}"


# Perform the POST request
try:
    response = requests.post(url, headers=headers, json=data, stream=True)

    if response.status_code == 200:
        # Save the audio file
        with open('speech.mp3', 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024): 
                f.write(chunk)
        logging.info("Speech synthesized successfully and saved to 'speech.mp3'.")
    else:
        logging.error(f"Failed to synthesize speech: {response.status_code} {response.text}")

except Exception as e:
    logging.error("Error during API call", exc_info=True)
