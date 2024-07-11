import wave
import websocket
import datetime
import hashlib
import base64
import hmac
import json
import ssl
import os
import io
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play

STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2

class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {"aue": "raw", "auf": "audio/L16;rate=16000", "vcn": "xiaoyan", "tte": "utf8"}
        self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")}

    def create_url(self):
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'), hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode('utf-8')
        authorization_origin = f"api_key=\"{self.APIKey}\", algorithm=\"hmac-sha256\", headers=\"host date request-line\", signature=\"{signature_sha}\""
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        url = 'wss://tts-api.xfyun.cn/v2/tts' + '?' + urlencode(v)
        return url

audio_io = io.BytesIO() 
def on_message(ws, message):
    global audio_io
    message = json.loads(message)
    if message["code"] != 0:
        print("Error:", message["message"])
    else:
        audio_io.write(base64.b64decode(message["data"]["audio"]))
        if message["data"]["status"] == 2:
            print("Playback starts")
            audio_io.seek(0)
            pcm_data = AudioSegment.from_file(audio_io, format="raw", sample_width=2, channels=1, frame_rate=16000)
            play(pcm_data)
            audio_io.close()
            ws.close()

def on_error(ws, error):
    print("### error:", error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")
    print("Close status code:", close_status_code)
    print("Close message:", close_msg)

def on_open(ws):
    def run(*args):
        if os.path.exists('./demo.pcm'):
            os.remove('./demo.pcm')
        d = {"common": wsParam.CommonArgs, "business": wsParam.BusinessArgs, "data": wsParam.Data}
        ws.send(json.dumps(d))
    thread.start_new_thread(run, ())

if __name__ == "__main__":
    load_dotenv()
    app_ID = os.getenv('XH_APPID')
    api_Secret = os.getenv('XH_APISecret')
    api_Key = os.getenv('XH_APIKey')
    wsParam = Ws_Param(APPID=app_ID, APISecret=api_Secret, APIKey=api_Key, Text="你好啊，我是小火,good day）")
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})