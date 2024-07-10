# -*- coding:utf-8 -*-
#
#   author: iflytek
#

import _thread as thread
import time
from time import mktime
from dotenv import load_dotenv
import os
import websocket

import base64
import datetime
import hashlib
import hmac
import json
import ssl
from datetime import datetime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

STATUS_FIRST_FRAME = 0  # Indicator for the first frame of data
STATUS_CONTINUE_FRAME = 1  # Indicator for continuation frames that follow the first frame
STATUS_LAST_FRAME = 2  # Indicator for the final frame of data


class Ws_Param(object):

    """
    The Ws_Param class is used to store the parameters needed for a websocket connection.

    Args:
        APPID (str): The APPID of the API user.
        APIKey (str): The APIKey of the API user.
        APISecret (str): The APISecret of the API user.
        AudioFile (str): The path of the audio file to be sent.
    """

    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile
        """
        The iat_params dictionary stores the parameters needed for the automatic speech recognition API.

        Attributes:
            domain (str): The domain of the API.
            language (str): The language of the speech.
            accent (str): The accent of the speech.
            dwa (str): The domain of the speech.
            result (dict): The parameters for the result of the automatic speech recognition API.
        """
        self.iat_params = {
            "domain": "slm", "language": "zh_cn", "accent": "mandarin","dwa":"wpgs", "result":
                {
                    "encoding": "utf8",
                    "compress": "raw",
                    "format": "plain"
                }
        }

    # 生成url
    def create_url(self):
        """
        Generate the URL for websocket connection.

        Returns:
            str: The URL for websocket connection.
        """
        # URL for websocket connection
        url = 'ws://iat.xf-yun.com/v1'

        # Generate RFC1123 formatted timestamp
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # Concatenate strings for signature
        signature_origin = "host: " + "iat.xf-yun.com" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v1 " + "HTTP/1.1"

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
            "host": "iat.xf-yun.com"
        }

        # Concatenate the authorization parameters to generate the URL
        url = url + '?' + urlencode(v)

        # Print the URL for reference (comment out the above print statement to compare the generated URL with your own code)
        # print('websocket url :', url)

        return url

final_result = ""
def on_message(ws, message):
    """
    Handle incoming messages from the websocket.

    Args:
        ws (WebSocket): The websocket object.
        message (str): The message received from the websocket.
    """
    global final_result
    # Parse the JSON message
    message = json.loads(message)
    # Get the code and status from the message header
    code = message["header"]["code"]
    status = message["header"]["status"]
    # If the code is not 0, it means there was an error in the request
    if code != 0:
        print(f"request error: {code}")
        ws.close()
    else:
        # Get the payload from the message (if it exists)
        payload = message.get("payload")
        if payload:
            # Get the text from the payload result
            text = payload["result"]["text"]
            # Decode and parse the text
            text = json.loads(str(base64.b64decode(text), "utf8"))
            # Get the text from the ws field
            text_ws = text['ws']
            # Concatenate the words in the text_ws field
            result = ''
            for i in text_ws:
                for j in i["cw"]:
                    w = j["w"]
                    result += w
            # If it's a continuation frame, append the result to final_result
            if status == STATUS_CONTINUE_FRAME:
                final_result = result
            # If it's the last frame, append the result to final_result and close the websocket
            if status == STATUS_LAST_FRAME:
                final_result += result
                print(final_result)
                ws.close()



def on_error(ws, error):
    """
    Handle websocket error.

    Args:
        ws (WebSocket): The websocket object.
        error (Exception): The error encountered.
    """
    # Print the error
    print("### error:", error)


def on_close(ws, close_status_code, close_msg):
    """
    Handle websocket close event.

    Args:
        ws (WebSocket): The websocket object.
        close_status_code (int): The status code of the close event.
        close_msg (str): The message accompanying the close event.
    """
    # Print the close status code and message
    print("### closed ###")
    print("WebSocket closed with status code:", close_status_code)
    print("Close message:", close_msg)



def on_open(ws):
    """
    Handle websocket connection open event.

    Args:
        ws (WebSocket): The websocket object.
    """
    def run(*args):
        """
        Run the websocket stream.

        Args:
            args: The arguments passed to the function.
        """
        frameSize = 1280  # Size of each audio frame
        intervel = 0.04  # Interval between sending audio frames (in seconds)
        status = STATUS_FIRST_FRAME  # Status of the audio: first frame, middle frame, or last frame

        # Read audio file in chunks
        with open(wsParam.AudioFile, "rb") as fp:
            while True:
                # Read audio frame
                buf = fp.read(frameSize)
                # Encode audio frame as base64
                audio = str(base64.b64encode(buf), 'utf-8')

                # If the end of the file is reached
                if not buf:
                    status = STATUS_LAST_FRAME
                # If it's the first frame
                if status == STATUS_FIRST_FRAME:
                    # Prepare the first frame message
                    d = {
                        "header": {
                            "status": 0,
                            "app_id": wsParam.APPID
                        },
                        "parameter": {
                            "iat": wsParam.iat_params
                        },
                        "payload": {
                            "audio": {
                                "audio": audio,
                                "sample_rate": 16000,
                                "encoding": "raw"
                            }
                        }
                    }
                    ws.send(json.dumps(d))
                    status = STATUS_CONTINUE_FRAME
                # If it's a middle frame
                elif status == STATUS_CONTINUE_FRAME:
                    # Prepare the middle frame message
                    d = {
                        "header": {
                            "status": 1,
                            "app_id": wsParam.APPID
                        },
                        "parameter": {
                            "iat": wsParam.iat_params
                        },
                        "payload": {
                            "audio": {
                                "audio": audio,
                                "sample_rate": 16000,
                                "encoding": "raw"
                            }
                        }
                    }
                    ws.send(json.dumps(d))
                # If it's the last frame
                elif status == STATUS_LAST_FRAME:
                    # Prepare the last frame message
                    d = {
                        "header": {
                            "status": 2,
                            "app_id": wsParam.APPID
                        },
                        "parameter": {
                            "iat": wsParam.iat_params
                        },
                        "payload": {
                            "audio": {
                                "audio": audio,
                                "sample_rate": 16000,
                                "encoding": "raw"
                            }
                        }
                    }
                    ws.send(json.dumps(d))
                    break

                # Simulate audio sampling interval
                time.sleep(intervel)

    # Start a new thread to run the websocket stream
    thread.start_new_thread(run, ())


if __name__ == "__main__":
    load_dotenv()
    app_ID=os.getenv('XH_APPID')
    api_Secret=os.getenv('XH_APISecret')
    api_Key=os.getenv('XH_APIKey')
    wsParam = Ws_Param(APPID=app_ID, APISecret=api_Secret,APIKey=api_Key,
                       AudioFile=r'iat_pcm_16k.pcm')
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
