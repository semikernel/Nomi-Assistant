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


class TTSManager:
    def __init__(self, response_queue):
        self.stop_event = threading.Event()
        self.response_queue = response_queue
        self.current_stream = None
        # 优化音频参数
        self.sample_rate = 24000  # 必须与模型输出采样率严格一致
        self.chunk_size = 4096    # 增大缓冲块大小
        self.pre_buffer_size = 3  # 预缓冲块数
        
        # 优化音频设备配置
        self.pyaudio_instance = pyaudio.PyAudio()
        dev_info = self.pyaudio_instance.get_device_info_by_index(
            self.pyaudio_instance.get_default_output_device_info()["index"]
        )
        self.actual_rate = int(dev_info["defaultSampleRate"])  # 设备实际采样率
        self.client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API"),
            base_url="https://api.siliconflow.cn/v1",
            timeout=30  # 增加超时设置
        )

    def _play_stream(self, audio_generator):
        # 双缓冲队列
        from collections import deque
        buffer_queue = deque(maxlen=self.pre_buffer_size)
        
        def buffer_filler():
            """预填充缓冲"""
            for _ in range(self.pre_buffer_size):
                if data := next(audio_generator, None):
                    buffer_queue.append(data)
        
        # 启动预缓冲线程
        threading.Thread(target=buffer_filler, daemon=True).start()

        # 优化音频重采样
        def resample(audio_segment):
            return audio_segment.set_frame_rate(self.actual_rate).set_channels(1)

        # 使用低延迟音频流
        stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.actual_rate,  # 使用设备原生采样率
            output=True,
            frames_per_buffer=self.chunk_size,
            start=False,
            stream_callback=self._audio_callback  # 异步回调模式
        )

        try:
            stream.start_stream()
            while stream.is_active() and not self.stop_event.is_set():
                # 保持缓冲队列充足
                if len(buffer_queue) < self.pre_buffer_size:
                    buffer_filler()
                
                # 处理当前数据块
                if mp3_data := buffer_queue.popleft() if buffer_queue else None:
                    audio = resample(AudioSegment.from_file(BytesIO(mp3_data), format="mp3"))
                    stream.write(audio.raw_data)
                
                time.sleep(0.001)  # 精确控制线程调度
        finally:
            stream.stop_stream()
            stream.close()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """异步回调函数实现零延迟"""
        # 此处实现环形缓冲机制
        # [具体实现需配合硬件特性优化]
        return (None, pyaudio.paContinue)
    
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