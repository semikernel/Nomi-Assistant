import os
import threading
import asyncio
import numpy as np
import sounddevice as sd
from queue import Empty
from pydub import AudioSegment
from io import BytesIO
from loguru import logger
import edge_tts
import time

class TTSManager:
    def __init__(self, response_queue):
        self.stop_event = threading.Event()
        self.response_queue = response_queue
        self.sample_rate = 24000  # MP3的采样率

    async def _async_play(self, audio_generator):
        try:
            mp3_data = bytearray()
            try:
                async for chunk in audio_generator:
                    if self.stop_event.is_set():
                        break
                    if chunk["type"] == "audio":
                        mp3_data.extend(chunk["data"])
            except asyncio.TimeoutError:
                logger.error("音频流生成超时")
                return
            try:
                # MP3解码
                audio = AudioSegment.from_mp3(BytesIO(mp3_data))
                pcm_data = np.array(audio.get_array_of_samples())

                # 统一采样率（部分系统可能不支持24kHz）
                if audio.frame_rate != self.sample_rate:
                    audio = audio.set_frame_rate(self.sample_rate)
                    pcm_data = np.array(audio.get_array_of_samples())

                sd.play(pcm_data, self.sample_rate)
                sd.wait()
            except Exception as e:
                logger.error(f"音频解码失败: {repr(e)}")
        except Exception as e:
            logger.error(f"播放失败: {e}")

    def _play_stream(self, audio_generator):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_play(audio_generator))
        except Exception as e:
            logger.error(f"播放协程错误: {e}")
        finally:
            loop.close()
    def stop_tts(self):
        """停止TTS播放并结束线程"""
        logger.debug('Stopping TTS')
        self.stop_event.set()  # 触发停止事件
        sd.stop()  # 立即停止音频播放
        self.stop_event.clear()
    def start_tts(self):
        logger.info("TTS服务启动")
        while not self.stop_event.is_set():
            try:
                if self.response_queue.empty():
                    time.sleep(0.1)
                    continue
                text = self.response_queue.get(timeout=0.1)
                if text == "[END]" or not text.strip():
                    continue
                communicate = edge_tts.Communicate(text, "zh-CN-XiaoyiNeural")
                self._play_stream(communicate.stream())
            except Empty:  # 显式捕获空队列异常
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"TTS错误: {e}")

if __name__ == "__main__":
    import queue
    q = queue.Queue()
    q.put("你好呀！我很高兴和你交流~如果你有什么问题，随时告诉我哦。我在这里，准备为你提供帮助！希望能为你的每一天带来一些便利和乐趣。如果你需要了解任何信息，或者只是想聊聊天，我都在呢！")
    q.put("[END]")
    try:
        tts = TTSManager(q)
        tts_thread = threading.Thread(target=tts.start_tts)
        tts_thread.start()
        tts_thread.join()
    except KeyboardInterrupt:
            print()
            logger.info("对话结束。")