import os
import threading
import pyaudio
from loguru import logger
from queue import Empty
import time
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
from pydub import AudioSegment
from pydub.utils import make_chunks
import warnings
import requests

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'
load_dotenv()

class TTSManager:
    def __init__(self, response_queue):
        self.stop_event = threading.Event()
        self.response_queue = response_queue
        self.current_stream = None
        
        # 修复音频设备初始化
        self.pyaudio_instance = pyaudio.PyAudio()  # 移除非法的参数
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
        # 修复音频流配置
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

            for mp3_data in audio_generator:
                if self.stop_event.is_set():
                    break
                
                # 增强解码稳定性
                try:
                    audio_buffer = BytesIO(mp3_data)
                    audio = AudioSegment.from_file(audio_buffer, format="mp3")
                    audio = audio.set_frame_rate(self.sample_rate)
                    audio = audio.set_channels(self.channels)
                    pcm_data = audio.raw_data
                except Exception as e:
                    logger.warning(f"音频解码失败: {str(e)}")
                    continue

                # 分块写入优化
                pos = 0
                while pos < len(pcm_data) and not self.stop_event.is_set():
                    end_pos = pos + self.chunk_size * 2  # 16bit=2bytes
                    stream.write(pcm_data[pos:end_pos])
                    pos = end_pos

        except OSError as e:
            logger.error(f"音频设备错误: {str(e)}")
        finally:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
    
    def stop_tts(self):
        logger.debug('Stopping TTS')
        self.stop_event.set()
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
                    chunk = self.response_queue.get(timeout=5)
                    if chunk == "[END]":
                        break
                    text_chunks.append(chunk)
                except Empty:
                    break

            full_text = "".join(text_chunks)
            if not full_text.strip():
                continue

            # 修复API调用参数
            try:
                with self.client.audio.speech.with_streaming_response.create(
                    model="FunAudioLLM/CosyVoice2-0.5B",
                    voice="speech:nomi:520j5ipxv4:tqpcoclegrdtuezsqprl",
                    input=full_text,
                    response_format="mp3",
                    # 移除无效的sample_rate参数
                    speed=1.0,  # 添加有效参数
                    # stream=True
                ) as response:
                    
                    def audio_generator():
                        try:
                            for chunk in response.iter_bytes(chunk_size=4096):
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
    # 测试时添加设备选择提示
    import queue
    q = queue.Queue()
    tts = TTSManager(q)
    
    # 显示可用音频设备
    print("可用音频设备：")
    for i in range(tts.pyaudio_instance.get_device_count()):
        dev = tts.pyaudio_instance.get_device_info_by_index(i)
        if dev["maxOutputChannels"] > 0:
            print(f"{dev['index']}: {dev['name']}")

    # 测试数据
    q.put("但是为什么我的声音很卡呢？")
    q.put("[END]")
    
    tts_thread = threading.Thread(target=tts.start_tts)
    tts_thread.start()
    tts_thread.join()