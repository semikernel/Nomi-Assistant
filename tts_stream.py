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
            timeout=30,
        )

    def _play_stream(self, audio_generator):
        stream = None #防止Unbound Error
        try:
            stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size, 
                start=False,
                output_device_index=self.pyaudio_instance.get_default_output_device_info()["index"]
            )
            stream.start_stream()
            # first_chunk = True
            # for pcm_chunk in audio_generator:
            #     if self.stop_event.is_set():
            #         break
            #     if first_chunk:
            #         logger.debug("收到第一个音频数据块")
            #         first_chunk = False
            #     stream.write(pcm_chunk)
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
        # self.pyaudio_instance.terminate()
        self.stop_event.clear()

    def start_tts(self):
        # 启动流式TTS
        logger.info('流式TTS启动')
        while not self.stop_event.is_set():
            try:
                # 如果响应队列中没有数据，则等待0.1秒
                if self.response_queue.empty():
                    time.sleep(0.1)
                    continue
                text = self.response_queue.get(timeout=0.1)  # 缩短超时时间
                if text == "[END]" or not text.strip():
                    continue

                # text_chunks = []
                # # 从响应队列中获取数据，直到遇到"[END]"或者超时
                # while True:
                #     try:
                #         chunk = self.response_queue.get(timeout=1)
                #         if chunk == "[END]":
                #             break
                #         text_chunks.append(chunk)
                #     except Empty:
                #         break

                # full_text = "".join(text_chunks)
                # 如果获取到的文本为空，则跳过
                # if not full_text.strip():
                #     continue

                # processed_text = self.add_endofprompt(full_text)

                # 添加指令前缀（根据API要求）
                # formatted_text = f"默认指令 <|endofprompt|> {processed_text}"


                
                    # 使用指定的模型和参数，创建一个流式响应
                with self.client.audio.speech.with_streaming_response.create(
                    model="FunAudioLLM/CosyVoice2-0.5B",
                    voice="speech:nomi:520j5ipxv4:tqpcoclegrdtuezsqprl",
                    input=text,
                    response_format="pcm",
                    speed=1.0,  # 添加有效参数
                    ) as response:
                        
                        # 定义一个生成音频数据的函数
                        def audio_generator():
                            try:
                                # 从响应中获取音频数据，每次获取1024*2字节
                                for chunk in response.iter_bytes(chunk_size=1024 * 2):
                                    # 如果停止事件被设置，则停止生成数据
                                    if self.stop_event.is_set():
                                        break
                                    # 生成音频数据
                                    yield chunk
                            except requests.exceptions.ChunkedEncodingError as e:
                                # 如果发生流数据中断，则记录错误日志
                                logger.error(f"流数据中断: {str(e)}")

                        # 创建一个线程，用于播放音频数据
                        play_thread = threading.Thread(
                            target=self._play_stream,
                            args=(audio_generator(),), #为目标函数传递参数​（元组形式）,args参数必须是一个元组，即使里面只有一个元素，后面也要加逗号，否则会被当作一个参数而不是元组。
                            daemon=True #将线程设置为守护线程。
                        )
                        # 启动线程
                        play_thread.start()
                        # 等待线程结束
                        play_thread.join()
            except Empty:
                time.sleep(0.05)  # 更细粒度的休眠减少延迟
            except Exception as e:
                # 如果发生其他异常，则记录错误日志，并等待1秒
                logger.error(f"TTS请求失败: {str(e)}")
                time.sleep(0.5)  # 错误冷却时间

if __name__ == "__main__":
    import queue
    q = queue.Queue()
    tts = TTSManager(q)
   
    q.put("你好呀！我很高兴和你交流~如果你有什么问题，随时告诉我哦。我在这里，准备为你提供帮助！希望能为你的每一天带来一些便利和乐趣。如果你需要了解任何信息，或者只是想聊聊天，我都在呢！")
    # q.put("恭喜你成为尊贵的蔚来车主！")
    q.put("[END]")
    
    tts_thread = threading.Thread(target=tts.start_tts)
    tts_thread.start()
    tts_thread.join()