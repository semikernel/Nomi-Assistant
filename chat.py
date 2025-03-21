import sys
import os
import threading
import time
import queue
from typing import Optional
from pathlib import Path
import pyaudio
import logging
from tts_stream import TTSManager  # 假设已实现

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ConversationState:
    """线程安全的状态管理器"""
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False          # 是否在对话中
        self._interrupted = False     # 是否被中断
        self._recording = False       # 是否在录音
        self._shutdown = False        # 关闭标志

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @active.setter
    def active(self, value: bool):
        with self._lock:
            self._active = value

    @property
    def interrupted(self) -> bool:
        with self._lock:
            return self._interrupted

    def set_interrupt(self):
        with self._lock:
            self._interrupted = True

    def clear_interrupt(self):
        with self._lock:
            self._interrupted = False

    @property
    def shutdown(self) -> bool:
        with self._lock:
            return self._shutdown

    def request_shutdown(self):
        with self._lock:
            self._shutdown = True

class WakeWordDetector:
    """热词唤醒模块"""
    def __init__(self, callback):
        self.callback = callback
        self.detector = None
        self._init_snowboy()

    def _init_snowboy(self):
        if 'snowboy' not in sys.modules:
            try:
                from snowboy import snowboydecoder
                self.detector = snowboydecoder.HotwordDetector(
                    "resources/snowboy.umdl", 
                    sensitivity=0.5
                )
            except ImportError:
                logger.error("Snowboy not available")

    def start_listening(self):
        def hotword_callback():
            self.callback()
            return False  # 单次检测

        if self.detector:
            self.detector.start(
                detected_callback=hotword_callback,
                interrupt_check=lambda: False,
                sleep_time=0.03
            )

class SpeechRecognizer:
    """语音识别模块"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.stream = None

    def record_audio(self, filename: str, duration: int = 5):
        """录制音频到文件"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        frames = []
        for _ in range(0, int(self.rate / self.chunk * duration)):
            data = self.stream.read(self.chunk)
            frames.append(data)
        
        self.stream.stop_stream()
        with open(filename, 'wb') as f:
            f.write(b''.join(frames))

class VoiceAssistant:
    """语音助手主控模块"""
    def __init__(self):
        self.state = ConversationState()
        self.response_queue = queue.Queue()
        self.tts_manager = TTSManager(self.response_queue)
        
        # 初始化模块
        self.wake_detector = WakeWordDetector(self._on_wakeword)
        self.recognizer = SpeechRecognizer()
        
        # 音频提示文件
        self.alert_sound = str(Path(__file__).parent / "sounds/alert.wav")
        
        # 启动线程
        self.wake_thread = threading.Thread(target=self._run_wake_detection)
        self.main_thread = threading.Thread(target=self._run_main_loop)

    def _on_wakeword(self):
        """热词唤醒回调"""
        if self.state.active:
            logger.info("Interrupting current conversation")
            self.tts_manager.stop_event.set()
            self.state.set_interrupt()
        else:
            logger.info("Wake word detected")
            self.state.active = True

    def _play_alert(self):
        """播放提示音"""
        os.system(f"aplay {self.alert_sound}")

    def _process_query(self, text: str) -> str:
        """处理用户查询（示例）"""
        # 这里可以接入LLM
        if "time" in text:
            return f"现在时间是 {time.strftime('%H:%M')}"
        return "我已经收到你的请求"

    def _run_wake_detection(self):
        """运行热词检测线程"""
        self.wake_detector.start_listening()

    def _run_main_loop(self):
        """主事件循环"""
        while not self.state.shutdown:
            if self.state.active:
                try:
                    # 处理中断逻辑
                    if self.state.interrupted:
                        self._handle_interruption()
                        continue

                    # 开始对话流程
                    self._play_alert()
                    
                    # 录音并识别
                    audio_file = "temp.wav"
                    self.recognizer.record_audio(audio_file)
                    text = "测试识别文本"  # 这里接入ASR
                    
                    # 处理请求
                    response = self._process_query(text)
                    
                    # 发送到TTS
                    self.response_queue.put(response)
                    
                    # 等待TTS完成
                    while not self.tts_manager.stop_event.is_set():
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                finally:
                    self.state.active = False
            time.sleep(0.1)

    def _handle_interruption(self):
        """处理中断逻辑"""
        logger.info("Handling interruption")
        self.tts_manager.stop_event.set()
        self.state.clear_interrupt()
        self.state.active = True  # 立即开始新的会话

    def start(self):
        """启动助手"""
        self.wake_thread.start()
        self.main_thread.start()
        logger.info("Voice assistant started")

    def graceful_shutdown(self):
        """优雅关闭"""
        self.state.request_shutdown()
        self.tts_manager.stop_event.set()
        self.wake_thread.join(timeout=1)
        self.main_thread.join(timeout=1)
        logger.info("Shutdown complete")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        assistant.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        assistant.graceful_shutdown()