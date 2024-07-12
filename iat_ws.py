# -*- coding:utf-8 -*-
#
# author: iflytek
#
import os
import asyncio
from voicerecorder import VoiceRecorder
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
    def __init__(self, AudioFile):
        """
        Initialize the parameters for the WebSocket API.
        """
        load_dotenv()
        self.APPID = os.getenv('XH_APPID')
        self.APIKey = os.getenv('XH_APIKey')
        self.APISecret = os.getenv('XH_APISecret')
        self.AudioFile = AudioFile
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

final_result = ''

def on_message(ws, message):
    global final_result
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
                final_result += result
            else:
                final_result += result
                logging.debug("Recognized text: %s", final_result)
                return final_result
    except Exception as e:
        logging.error("receive msg,but parse exception: %s", e)

def on_error(ws, error):
    logging.error("### error ###: %s" % error)

def on_close(ws, close_status_code, close_msg):
    logging.debug("### closed ###")
    logging.debug("WebSocket closed with status code: %s", close_status_code)
    logging.debug("Close message: %s", close_msg)

def on_open(ws, wsParam):
    def run(*args):
        frameSize = 8000
        intervel = 0.04
        status = STATUS_FIRST_FRAME  # Start with the first frame
        with open(wsParam.AudioFile, "rb") as fp:
            while True:
                buf = fp.read(frameSize)
                if not buf:
                    status = STATUS_LAST_FRAME  # No more data, mark last frame
                
                data = {
                    "status": status,
                    "format": "audio/L16;rate=16000",
                    "audio": base64.b64encode(buf).decode('utf-8'),
                    "encoding": "raw"
                }
                
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
                    time.sleep(1)  # Wait a second to ensure last frame is processed
                    ws.close()
                    break

                time.sleep(intervel)

    thread.start_new_thread(run, ())

async def main():
    voice_recorder = VoiceRecorder()
    try:
        # Record the audio data
        audio_data = await voice_recorder.record_audio_vad()
        # Save the recorded audio to a temporary PCM file
        pcm_file_path = voice_recorder.array_to_pcm_bytes(audio_data)

        
        # Prepare WebSocket parameters
        wsParam = Ws_Param(AudioFile=pcm_file_path)
        wsUrl = wsParam.create_url()

        # WebSocket event handlers
        ws = websocket.WebSocketApp(wsUrl, on_open=lambda ws: on_open(ws, wsParam), on_message=on_message, on_error=on_error, on_close=on_close)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    except Exception as e:
        logging.error("An error occurred: %s", e)
        if os.path.exists(pcm_file_path):
            os.remove(pcm_file_path)
            logging.info("Temporary file removed: %s", pcm_file_path)

if __name__ == "__main__":
    asyncio.run(main())
