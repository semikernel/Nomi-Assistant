import json
import time
import queue
import threading
import webrtcvad
import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel
import re

SetLogLevel(level=-1)

# 音频参数配置
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000  # Vosk要求16kHz采样率 每秒16000个采样点
FRAME_DURATION = 20  # 每帧20ms，0.02秒,因为paInt16是两个字节的量
CHUNK = int(AUDIO_RATE * FRAME_DURATION * 2 / 1000) # 640个采样点
VAD_MODE = 3  # WebRTC VAD灵敏度(0-3)
SILENCE_TIMEOUT = 1.0  # 静音超时(秒)
VAD_WINDOW = 0.5  # VAD检测窗口（秒）
MIN_SPEECH_RATIO = 0.6  # 语音帧占比阈值

class RealtimeSTT:
    def __init__(self):
        # 初始化语音识别模型
        self.model = Model("vosk_reco/model_full")  # 替换为你的模型路径
        self.recognizer = KaldiRecognizer(self.model, AUDIO_RATE)
        
        # 初始化WebRTC VAD
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(VAD_MODE)
        
        # 音频输入队列
        self.audio_queue = queue.Queue()
        self.is_running = False

    def _vad_check(self, audio_data):
        """多帧统计检测语音活动"""
        frame_length = int(AUDIO_RATE * FRAME_DURATION / 1000) * 2  # 640字节/帧，1帧是20ms
        n_frames = len(audio_data) // frame_length
        if n_frames == 0:
            return False
        speech_frames = 0
        for i in range(n_frames):
            frame = audio_data[i * frame_length : (i + 1) * frame_length]
            if self.vad.is_speech(frame, AUDIO_RATE):
                speech_frames += 1
        return (speech_frames / n_frames) >= MIN_SPEECH_RATIO

    def audio_capture(self):
        """音频采集线程"""
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._audio_callback
        )
        stream.start_stream()
        while self.is_running:
            time.sleep(0.1)
        stream.stop_stream()
        stream.close()
        pa.terminate()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频输入回调"""
        # 将音频数据和当前时间放入队列中
        self.audio_queue.put((in_data, time.time()))
        # 这里的in_data是一个元组，包含音频数据和当前时间
        # 返回None和pyaudio.paContinue，表示继续录制
        return (None, pyaudio.paContinue)

    def process_audio(self):
        """语音处理线程"""
        speech_buffer = []
        vad_buffer = []
        last_voice_time = time.time()
        
        while self.is_running:
            try:
                data , timestamp = self.audio_queue.get(timeout=0.1)
                vad_buffer.append(data)
            except queue.Empty:
                data = None
                continue
            
            if len(vad_buffer) * FRAME_DURATION / 1000 >= VAD_WINDOW or (data is None and vad_buffer):
                audio_data = b''.join(vad_buffer)
                is_speech = self._vad_check(audio_data)
                
                if is_speech:
                    speech_buffer.extend(vad_buffer)
                    last_voice_time = time.time()
                else:
                    pass
                vad_buffer.clear()
            
            if time.time() - last_voice_time > SILENCE_TIMEOUT:
                if speech_buffer:
                    self._finalize_recognition(b''.join(speech_buffer))
                    speech_buffer.clear()
            
            # 流式识别处理
            if data and self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '')
                text = re.sub(r'\s+', '', text)
                if text:  # 仅当text非空时打印
                    print(f"最终结果: {text}")

    def _finalize_recognition(self, audio_data):
        """处理完整语音段"""
        print(f"检测到语音段，长度: {len(audio_data)/AUDIO_RATE:.2f}秒")

    def start(self):
        """启动服务"""
        self.is_running = True
        threading.Thread(target=self.audio_capture, daemon=True).start()
        threading.Thread(target=self.process_audio, daemon=True).start()

    def stop(self):
        """停止服务"""
        self.is_running = False

if __name__ == "__main__":
    stt_engine = RealtimeSTT()
    try:
        stt_engine.start()
        print("语音识别服务已启动，请开始说话...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stt_engine.stop()
        print("\n服务已停止")