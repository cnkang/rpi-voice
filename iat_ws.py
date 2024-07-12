# -*- coding:utf-8 -*-

import os
import asyncio
from voicerecorder import VoiceRecorder
import websocket
from dotenv import load_dotenv
import logging
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
    def __init__(self, AudioStream):
        """
        Initializes the parameters for the WebSocket API.
        """
        load_dotenv()
        self.APPID = os.getenv('XH_APPID')
        self.APIKey = os.getenv('XH_APIKey')
        self.APISecret = os.getenv('XH_APISecret')
        self.AudioStream = AudioStream
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {
            "domain": "iat",
            "language": "zh_cn",
            "accent": "mandarin",
            "vinfo": 1,
            "vad_eos": 10000
        }

    def create_url(self):
        """
        Generate the URL for websocket connection.
        """
        url = 'wss://iat-api.xfyun.cn/v2/iat'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: iat-api.xfyun.cn\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET /v2/iat HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": "iat-api.xfyun.cn"
        }
        url = url + '?' + urlencode(v)
        return url

class XH_SpeechRecognizer:
    """
    A class to perform speech recognition using iFlytek's WebSocket API.
    """
    def __init__(self):
        """
        Initialize the SpeechRecognizer.
        """
        logging.basicConfig(level=logging.INFO)
        self.final_result = ''
        self.audio_stream = None

    def on_message(self, ws, message):
        """
        Handle incoming messages from the WebSocket.
        """
        try:
            message = json.loads(message)
            result = ""
            code = message["code"]
            sid = message["sid"]
            if code != 0:
                errMsg = message["message"]
                logging.error("sid:%s call error:%s code is:%s", sid, errMsg, code)
            else:
                result = ''.join(cw['w'] for ws in message.get('data', {}).get('result', {}).get('ws', []) for cw in ws.get('cw', []))
                if message['data']['status'] < STATUS_LAST_FRAME:
                    self.final_result += result
                else:
                    self.final_result += result
                    logging.info("Recognized text: %s", self.final_result)
            return self.final_result
        except Exception as e:
            logging.error("receive msg,but parse exception: %s", e)
            return None

    def on_error(self, ws, error):
        """
        Handle errors in the WebSocket.
        """
        logging.error("### error ###: %s",error)

    def on_close(self, ws, close_status_code, close_msg):
        """
        Handle the WebSocket closure.
        """
        logging.debug("### closed ###")
        logging.debug("WebSocket closed with status code: %s", close_status_code)
        logging.debug("Close message: %s", close_msg)
        self.audio_stream.close()
    def on_open(self, ws, wsParam):
        """
        Handle the WebSocket opening and start sending data.
        """
        def run(*args):
            frameSize = 8000  # Frame size
            intervel = 0.04  # Interval in seconds
            status = STATUS_FIRST_FRAME  # Start with the first frame

            wsParam.AudioStream.seek(0)  # Move to the start of the stream

            while True:
                buf = wsParam.AudioStream.read(frameSize)
                logging.debug('Buffer type: %s', type(buf))  # 这将显示buf的数据类型

                if not buf:
                    status = STATUS_LAST_FRAME  # No more data, mark last frame

                data = {
                    "status": status,
                    "format": "audio/L16;rate=16000",
                    "audio": base64.b64encode(buf).decode('utf-8'),
                    "encoding": "raw"
                }

                # Send data based on the frame status
                if status == STATUS_FIRST_FRAME:
                    d = {"common": wsParam.CommonArgs, "business": wsParam.BusinessArgs, "data": data}
                    ws.send(json.dumps(d))
                    status = STATUS_CONTINUE_FRAME
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": data}
                    ws.send(json.dumps(d))
                elif status == STATUS_LAST_FRAME:
                    d = {"data": data}
                    ws.send(json.dumps(d))
                    time.sleep(1)  # Wait a second to ensure the last frame is processed
                    ws.close()
                    break

                time.sleep(intervel)

        thread.start_new_thread(run, ())
    
    async def run_recognition(self):
        """
        Start the speech recognition.
        """
        voice_recorder = VoiceRecorder()
        try:
            # Record the audio data
            audio_data = await voice_recorder.record_audio_vad()
            # Save the recorded audio to a BytesIO stream
            self.audio_stream = voice_recorder.array_to_pcm_bytes(audio_data)

            # Prepare WebSocket parameters
            wsParam = Ws_Param(AudioStream=self.audio_stream)
            wsUrl = wsParam.create_url()

            # WebSocket event handlers
            ws = websocket.WebSocketApp(wsUrl,
                                        on_open=lambda ws: self.on_open(ws, wsParam),
                                        on_message=self.on_message,
                                        on_error=self.on_error,
                                        on_close=self.on_close)
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        except Exception as e:
            logging.error("An error occurred: %s", e)
            if self.audio_stream:
                self.audio_stream.close()
                logging.debug("Stream has been closed")

if __name__ == "__main__":
    recognizer = XH_SpeechRecognizer()
    asyncio.run(recognizer.run_recognition())
