# -*- coding:utf-8 -*-
#
#   author: iflytek
#
import os
import websocket
from dotenv import load_dotenv
import logging
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread

# Constants representing the status of frames in the websocket communication
STATUS_FIRST_FRAME = 0  # Indicator for the first frame
STATUS_CONTINUE_FRAME = 1  # Indicator for intermediate frames
STATUS_LAST_FRAME = 2  # Indicator for the last frame


class Ws_Param(object):
    """
    The Ws_Param class is used to store the parameters needed for a websocket connection.
    """

    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        """
        Initialize the parameters for the WebSocket API.

        Args:
            APPID (str): The application ID.
            APIKey (str): The API key.
            APISecret (str): The API secret.
            AudioFile (str): The path to the audio file.
        """
        # Set the application ID
        self.APPID = APPID
        # Set the API key
        self.APIKey = APIKey
        # Set the API secret
        self.APISecret = APISecret
        # Set the path to the audio file
        self.AudioFile = AudioFile

        # Set the common parameters
        self.CommonArgs = {"app_id": self.APPID}
        # Set the business parameters with default values
        # More customizable parameters can be found on the official website
        self.BusinessArgs = {
            # The domain of the API
            "domain": "iat",
            # The language of the speech
            "language": "zh_cn",
            # The accent of the speech
            "accent": "mandarin",
            # Set to 1 to enable voice info
            "vinfo": 1,
            # Set the end of speech duration
            "vad_eos": 10000
        }

    # 生成url
    def create_url(self):
        """
        Generate the URL for websocket connection.

        Returns:
            str: The URL for websocket connection.
        """
        # URL for websocket connection
        url = 'wss://iat-api.xfyun.cn/v2/iat'

        # Generate RFC1123 formatted timestamp
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # Concatenate strings for signature
        signature_origin = "host: " + "iat-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"

        # Perform HMAC-SHA256 encryption
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        # Concatenate strings for authorization
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # Combine the authorization parameters into a dictionary
        v = {
            "authorization": authorization,
            "date": date,
            "host": "iat-api.xfyun.cn"
        }

        # Concatenate the authorization parameters to generate the URL
        url = url + '?' + urlencode(v)

        # Uncomment the following lines to print the generated URL for reference
        # print('Websocket URL:', url)

        return url


final_result=''
def on_message(ws, message):
    """
    Process the received message from the websocket.

    Args:
        ws (WebSocketApp): The websocket object.
        message (str): The received message.
    """
    global final_result
    try:
        # Parse the message as JSON
        message = json.loads(message)
        result = ""
        code = message["code"]
        status = message.get('data', {}).get('status')
        sid = message["sid"]
        
        # If there is an error code, log the error
        if code != 0:
            errMsg = message["message"]
            logging.error("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

        else:
            # Combine the words from all the continuation frames
            result = ''.join(cw['w'] for ws in message.get('data', {}).get('result', {}).get('ws', []) for cw in ws.get('cw', []))
            
            # If it's a continuation frame, append the result to final_result
            if(status == STATUS_FIRST_FRAME or status == STATUS_CONTINUE_FRAME):
                final_result += result
            # If it's the last frame, print the final result
            elif(status == STATUS_LAST_FRAME):
                final_result += result
                print(final_result)

    except Exception as e:
        # If there is an error during message parsing, log the error
        logging.error("receive msg,but parse exception: %s", e)



# 收到websocket错误的处理
def on_error(ws, error):
    """
    Handle websocket error.

    Args:
        ws (WebSocketApp): The websocket object.
        error (Exception): The error encountered.
    """
    # Log the error
    logging.error("### error ###: %s" % error)


# 收到websocket关闭的处理
def on_close(ws, close_status_code, close_msg):
    """
    Handle websocket close event.

    Args:
        ws (WebSocketApp): The websocket object.
        close_status_code (int): The status code of the close event.
        close_msg (str): The message accompanying the close event.
    """
    # Log the close status code and message
    logging.debug("### closed ###")
    logging.debug("WebSocket closed with status code: %s", close_status_code)
    logging.debug("Close message: %s", close_msg)


# 收到websocket连接建立的处理
def on_open(ws):
    """
    Handle websocket connection open event.

    Args:
        ws (WebSocketApp): The websocket object.
    """
    def run(*args):
        """
        Read audio file and send it to the websocket.
        """
        # Set audio parameters
        frameSize = 8000  # Each frame size in bytes
        intervel = 0.04  # Interval between sending audio frames
        status = STATUS_FIRST_FRAME  # Status of the audio frame

        # Open the audio file
        with open(wsParam.AudioFile, "rb") as fp:
            while True:
                # Read audio data from the file
                buf = fp.read(frameSize)

                # If there is no more audio data, it's the last frame
                if not buf:
                    status = STATUS_LAST_FRAME

                # If it's the first frame, send it with the business parameters
                if status == STATUS_FIRST_FRAME:
                    # Create the first frame data
                    d = {
                        "common": wsParam.CommonArgs,
                        "business": wsParam.BusinessArgs,
                        "data": {
                            "status": 0,  # Status of the audio frame
                            "format": "audio/L16;rate=16000",  # Audio format
                            "audio": str(base64.b64encode(buf), 'utf-8'),  # Base64 encoded audio data
                            "encoding": "raw"  # Encoding type
                        }
                    }
                    # Send the first frame data to the websocket
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME  # Change the status to continue frame

                # If it's not the first frame, send it as a continue frame
                elif status == STATUS_CONTINUE_FRAME:
                    # Create the continue frame data
                    d = {
                        "data": {
                            "status": 1,  # Status of the audio frame
                            "format": "audio/L16;rate=16000",  # Audio format
                            "audio": str(base64.b64encode(buf), 'utf-8'),  # Base64 encoded audio data
                            "encoding": "raw"  # Encoding type
                        }
                    }
                    # Send the continue frame data to the websocket
                    ws.send(json.dumps(d))

                # If it's the last frame, send it and close the websocket
                elif status == STATUS_LAST_FRAME:
                    # Create the last frame data
                    d = {
                        "data": {
                            "status": 2,  # Status of the audio frame
                            "format": "audio/L16;rate=16000",  # Audio format
                            "audio": str(base64.b64encode(buf), 'utf-8'),  # Base64 encoded audio data
                            "encoding": "raw"  # Encoding type
                        }
                    }
                    # Send the last frame data to the websocket
                    ws.send(json.dumps(d))
                    time.sleep(1)  # Wait for a second before closing the websocket
                    ws.close()
                    break

                # Sleep for the specified interval before sending the next frame
                time.sleep(intervel)

    # Start a new thread to read the audio file and send it to the websocket
    thread.start_new_thread(run, ())


if __name__ == "__main__":
    load_dotenv()
    time1 = datetime.now()
    app_ID=os.getenv('XH_APPID')
    api_Secret=os.getenv('XH_APISecret')
    api_Key=os.getenv('XH_APIKey')
    wsParam = Ws_Param(APPID=app_ID, APISecret=api_Secret,
                       APIKey=api_Key,
                       AudioFile=r'iat_pcm_16k.pcm')
    
    wsUrl = wsParam.create_url()
    # Establishing a WebSocket connection using the url obtained
    ws = websocket.WebSocketApp(wsUrl,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    time2 = datetime.now()
    logging.debug("Time used: %s", time2 - time1)
