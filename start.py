from openai import OpenAI
import json

SYSTEM = """
你是xxx，一个聪明、热情、善良的人，后面的对话来自你的女朋友，你要认真地回答她。
"""


# ## 本地部署
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8888/v1"
# model_name = "model_path"
## 云端调用
openai_api_key="sk-xxx"
openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
model_name = "qwen-plus"

client = OpenAI(
    api_key = openai_api_key,
    base_url = openai_api_base
)

def single_chat(query):
    messages = []
    messages.append({"role": "system", "content": SYSTEM})
    messages.append({"role": "user", "content": query})
    chat_response = client.chat.completions.create(
        model = model_name,
        messages = messages,
        max_tokens = 4096, 
        temperature = 0.6, 
        top_p = 0.95, 
        extra_body = {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }, 
    )
    return chat_response.choices[0].message.content

def multi_chat():
    messages = []
    messages.append({"role": "system", "content": SYSTEM})
    history_len = 20
    print("输入回车可退出对话。")
    while True:
        if len(messages) >= history_len:
            messages.pop(1)
            messages.pop(1)
        query = input("请输入对话: ").strip()
        if query == "\n":
            break
        messages.append({"role": "user", "content": query})
        try:
            chat_response = client.chat.completions.create(
                model = model_name,
                messages = messages,
                max_tokens = 4096, 
                temperature = 0.6, 
                top_p = 0.95, 
                extra_body = {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, 
            )
            response = chat_response.choices[0].message.content
            print(response)
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"发生错误: {e}")
            continue

if __name__ == "__main__":
    multi_chat()