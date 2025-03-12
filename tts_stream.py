import os
import threading
import pyaudio
from loguru import logger
from queue import Empty
import time
from dotenv import load_dotenv
from openai import OpenAI
import warnings
import requests
import re

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'
load_dotenv()

class TTSManager:
    def __init__(self, response_queue):
        self.stop_event = threading.Event()
        self.response_queue = response_queue
        self.current_stream = None
        
        # 修复音频设备初始化
        self.pyaudio_instance = pyaudio.PyAudio()  
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 24000
        self.chunk_size = 1024
        # 修复API客户端初始化
        self.client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API"),
            base_url="https://api.siliconflow.cn/v1",
            timeout=30  # 增加超时设置
        )
    def _play_stream(self, audio_generator):
        try:
            stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,  # 增加缓冲区大小
                start=False,
                output_device_index=self.pyaudio_instance.get_default_output_device_info()["index"]
            )
            stream.start_stream()

            # 直接遍历 PCM 数据块生成器
            for pcm_chunk in audio_generator:
                if self.stop_event.is_set():
                    break

                # 🔥 直接写入 PCM 数据（无需格式转换）
                stream.write(pcm_chunk)  

        except OSError as e:
            logger.error(f"音频设备错误: {str(e)}")
        finally:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
            self.current_stream = None
    def add_endofprompt(self, text):
        """
        在非空格标点符号后添加 <|endofprompt|>
        这个是用来给tts增加一些自然语气的，比如开心，难过，低语等，暂时不考虑
        """
        # 转义特殊字符（如 .），使用原始字符串确保转义正确
        pattern = r'([!?。\.,;:~！])'  # 对 . 进行转义 → \.
        return re.sub(pattern, r'\1<|endofprompt|> ', text)
    def stop_tts(self):
        logger.debug('Stopping TTS')
        self.stop_event.set()
        if self.current_stream:
            self.current_stream.stop_stream()
            self.current_stream.close()
        self.pyaudio_instance.terminate()
        self.stop_event.clear()

    def start_tts(self):
        logger.info('流式TTS启动')
        while not self.stop_event.is_set():
            if self.response_queue.empty():
                time.sleep(0.1)
                continue

            text_chunks = []
            while True:
                try:
                    chunk = self.response_queue.get(timeout=1)
                    if chunk == "[END]":
                        break
                    text_chunks.append(chunk)
                except Empty:
                    break

            full_text = "".join(text_chunks)
            if not full_text.strip():
                continue

            # processed_text = self.add_endofprompt(full_text)

            # 添加指令前缀（根据API要求）
            # formatted_text = f"默认指令 <|endofprompt|> {processed_text}"


            try:
                with self.client.audio.speech.with_streaming_response.create(
                    model="FunAudioLLM/CosyVoice2-0.5B",
                    voice="speech:nomi:520j5ipxv4:tqpcoclegrdtuezsqprl",
                    input=full_text,
                    response_format="pcm",
                    speed=1.0,  # 添加有效参数
                ) as response:
                    
                    def audio_generator():
                        try:
                            for chunk in response.iter_bytes(chunk_size=1024 * 2):
                                if self.stop_event.is_set():
                                    break
                                yield chunk
                        except requests.exceptions.ChunkedEncodingError as e:
                            logger.error(f"流数据中断: {str(e)}")

                    play_thread = threading.Thread(
                        target=self._play_stream,
                        args=(audio_generator(),),
                        daemon=True
                    )
                    play_thread.start()
                    play_thread.join()

            except Exception as e:
                logger.error(f"TTS请求失败: {str(e)}")
                time.sleep(1)  # 错误冷却时间

if __name__ == "__main__":
    import queue
    q = queue.Queue()
    tts = TTSManager(q)
   
    q.put("你好呀！我很高兴和你交流~如果你有什么问题，随时告诉我哦。我在这里，准备为你提供帮助！希望能为你的每一天带来一些便利和乐趣。如果你需要了解任何信息，或者只是想聊聊天，我都在呢！")
    q.put("[END]")
    
    tts_thread = threading.Thread(target=tts.start_tts)
    tts_thread.start()
    tts_thread.join()