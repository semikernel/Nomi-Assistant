import os
from dotenv import load_dotenv
import requests

load_dotenv()

url = "https://api.siliconflow.cn/v1/uploads/audio/voice"
headers = {
    "Authorization":  f"Bearer {os.getenv('SILICONFLOW_API')}" # 从 https://cloud.siliconflow.cn/account/ak 获取
}
with open("nomi_short.wav", "rb") as audio_file:
    files = {
        "file": audio_file  # 文件句柄自动关闭
    }
    data = {
        "model": "FunAudioLLM/CosyVoice2-0.5B",
        "customName": "nomi",
        "text": "你好呀，我是nomi mate，一个聪明可爱的小精灵。升级后的我每天都元气满满，声音比之前更清楚，音调也更自然了。我还会用多样的语气表达自己的情感。比如，当你遇到快乐的事情，我会被你感染，语气也会很欢快。"
    }

    response = requests.post(url, headers=headers, files=files, data=data)

print(response.status_code)
print(response.json())  # 打印响应内容（如果是JSON格式）