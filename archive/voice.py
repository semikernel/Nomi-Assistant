import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
load_dotenv()

speech_file_path = Path(__file__).parent / "siliconcloud-generated-speech.mp3"

client = OpenAI(
    api_key=f"{os.getenv('SILICONFLOW_API')}", # 从 https://cloud.siliconflow.cn/account/ak 获取
    base_url="https://api.siliconflow.cn/v1"
)

with client.audio.speech.with_streaming_response.create(
  model="FunAudioLLM/CosyVoice2-0.5B", # 支持 fishaudio / GPT-SoVITS / CosyVoice2-0.5B 系列模型
  voice="speech:nomi:520j5ipxv4:tqpcoclegrdtuezsqprl", # 用户上传音色名称，参考
  # 用户输入信息
  input="你好呀！我很高兴和你交流~如果你有什么问题，随时告诉我哦。我在这里，准备为你提供帮助！希望能为你的每一天带来一些便利和乐趣。如果你需要了解任何信息，或者只是想聊聊天，我都在呢！",
  response_format="mp3"
) as response:
    response.stream_to_file(speech_file_path)
