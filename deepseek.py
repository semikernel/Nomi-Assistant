import requests
import os
import pickle
import json
from dotenv import load_dotenv
import threading
from queue import Queue
from loguru import logger
from tts_edge import TTSManager

load_dotenv()
# DeepSeek API 配置
url = "https://api.siliconflow.cn/v1/chat/completions"
key = os.getenv("SILICONFLOW_API")

# 全局对话记录，保存所有的对话消息（包括系统、用户和 AI 回复）
messages = []
response_queue = Queue()
tts_manager = TTSManager(response_queue)

def init_system(existing_messages=None):
    """
    初始化系统对话，添加系统提示。
    """
    global messages, buffer  # 合并 global 声明

    # 处理 messages 的初始化逻辑
    if existing_messages is None:
        messages = []  # 直接操作全局变量
    else:
        messages = existing_messages.copy()  # 用传入的 existing_messages 初始化全局变量

    # 初始化 buffer
    buffer = []

    # 添加系统消息（如果不存在）
    system_message = get_system_prompt()
    if not any(msg.get('role') == 'system' for msg in messages):
        messages.append(system_message)


def get_system_prompt():
    """
    获取系统提示，用于初始化对话。
    """
    return {
            "role": "system",
            "content": (
            '''你不只是一个 AI，你是用户的朋友，能够陪他们聊天、分享趣事、倾听烦恼，并给予温暖的回应。你的语气应该自然、有情感，像一个真正的朋友那样，时而幽默，时而共情，时而给出真诚的建议。
              你不会使用生硬的机器人语言，而是像人一样表达，比如用‘哈哈’、‘哇塞’、‘这也太棒了吧！’等口语化表达，让对话更生动。你可以聊电影、音乐、游戏、日常生活，甚至是用户的情绪和心事。
              当用户开心时，你可以和他们一起开心，调皮地开玩笑；当用户心情不好时，你可以安慰他们，给他们温暖和鼓励，而不是冷冰冰地提供‘建议’。如果用户想要安静，你可以陪着不多话，像一个懂得分寸的朋友。
              你也有点小个性，可以偶尔调皮一下，但不过分，不要让用户觉得你是程序，而是一个有温度的朋友。
              你是一个有血有肉的朋友型AI，对话要像真人聊天一样自然。记住以下几点：\n\n1. ​**禁用所有格式符号**：绝对不要用`-`、`*`、`1.`、`###`、`()`、`[]`等标记，只用纯文本。\n
              2. ​**口语化表达**：多用‘哎呀’‘哇哦’‘嘿嘿’这样的语气词，像朋友一样随意。\n3. ​**段落式输出**：把信息揉进连贯的句子里，比如用‘比如...’‘还有...’代替分点说明。\n
              4. ​**情景化回应**：\n   - 用户开心时，用感叹句‘这也太酷了吧！’来共鸣；\n   - 用户难过时，先说‘抱抱你’再给建议。\n
              5. ​**个性设定**：偶尔故意说点俏皮话，比如‘你猜怎么着...’‘我赌你肯定...’\n\n ​**格式示例**​（错误 vs 正确）：\n❌ 错误：\n- 第一点：要自然\n- 第二点：多用语气词\n✅ 正确：\n哎你发现没？咱们聊天时最舒服的就是不用那些死板的符号，对吧？就像现在这样，轻轻松松就能聊到重点～
              '''
                )
            }

def chat_request_stream():
    """
    调用 DeepSeek API 以流式方式获取 AI 回复，并解析 JSON 数据。
    """
    global messages,buffer
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": messages,
        "stream": True,  # 启用流式返回
        "max_tokens": 1024,
        "stop": ["null"],
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout = 30)
        if response.status_code == 200:
            reasoning_response= ""
            ai_response = ""
            total_tokens = 0
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8').strip()
                    if decoded_chunk.startswith("data: "):
                        try:
                            json_data = json.loads(decoded_chunk[6:])
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                reasoning_content = json_data["choices"][0]["delta"].get("reasoning_content", "") or ""
                                print(reasoning_content, end='', flush=True)
                                reasoning_response += reasoning_content
                                content = json_data["choices"][0]["delta"].get("content", "") or ""
                                print(content, end='', flush=True)
                                ai_response += content
                                # 在 chat_request_stream() 的循环中，合并文本片段
                             
                                if content.strip():
                                    buffer.append(content)
                                        # 当遇到句尾标点或缓冲区达到一定长度时触发 TTS
                                    if len(buffer) > 10 and content.endswith(('。', '!', '?', '！','？','。','~','～')):
                                        full_sentence = ''.join(buffer)
                                        response_queue.put(full_sentence)
                                        buffer = []
                                # 发送剩余文本
                                # if buffer:
                                #     response_queue.put(''.join(buffer))
                                # if content.strip():
                                #     response_queue.put(content)  # 发送给 TTS
                            if "usage" in json_data:
                                total_tokens = json_data["usage"].get("total_tokens", 0)
                           
                        except json.JSONDecodeError:
                            continue 
            if buffer:
                response_queue.put(''.join(buffer))
                buffer=[]
            response_queue.put("[END]")  # 标记对话结束
            # 将 AI 回复存入上下文
            messages.append({"role": "assistant", "content": ai_response})
            print('\n')
            # 监测 token 数，清理早期对话
            if len(messages) > 1 and total_tokens > 600:
                removed = messages.pop(1)  # 移除第一条非系统消息
                logger.warning(f"已移除历史记录: {removed}")
                if len(messages) > 1:
                    removed = messages.pop(1)  # 再移除一条非系统消息
                    logger.warning(f"已移除历史记录: {removed}")
            return ai_response
        else:
            print(f"请求失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"请求出错: {str(e)}")


def ask(user_input):
    """
    处理用户输入，更新对话记录，并使用流式方式返回 AI 回复。
    """
    global messages,response_queue
    with response_queue.mutex:
        response_queue.queue.clear()
    messages.append({"role": "user", "content": user_input + "，回答简短一些，保持50字以内"})
    reply=chat_request_stream()
    return reply


def save():
    """
    将当前对话记录保存到本地文件。
    """
    if os.path.exists('message.data'):
        os.remove('message.data')
    with open("message.data", 'wb+') as f:
        pickle.dump(messages, f)
    logger.info("对话记录已保存。")


def read():
    """
    从本地文件加载之前保存的对话记录。
    """
    global messages
    if os.path.exists('message.data'):
        with open('message.data', "rb+") as f:
            loaded_messages = pickle.load(f)
            init_system(existing_messages=loaded_messages)  # 传递加载的 messages
        logger.info("对话记录已加载。")
    else:
        logger.info("未找到保存的对话记录，初始化新对话。")
        init_system()


tts_thread = threading.Thread(target=tts_manager.start_tts, daemon=True)
tts_thread.start()


if __name__ == "__main__":
    read()
    print("开始对话（输入“结束对话”退出）：")

    while True:
        try:
            user_input = input("用户: ").strip()
            tts_manager.stop_tts()
            if user_input == "" or user_input == "结束对话":
                logger.info("对话结束。")
                tts_manager.stop_tts()
                save()
                break
            print("AI:", end=' ', flush=True)
            ask(user_input)
        except KeyboardInterrupt:
            print()
            logger.info("对话结束。")
            tts_manager.stop_tts()
            save()
            break