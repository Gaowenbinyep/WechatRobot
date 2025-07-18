import pandas as pd
import os
import json
from tqdm import tqdm
from openai import OpenAI
import re

os.environ["DASHSCOPE_API_KEY"] = "sk-xxx"

# ======================================
# 配置区
MODEL = "qwen-plus"  # 替换成可用的模型名
DEV_FILE = "xxx.json"
EVALUATION_FILE = "xxx.json"
# ======================================


def call_qwen(prompt):
    """ 调用 Qwen3-325B 返回结果 """
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=prompt,
        extra_body={"enable_thinking": False},
    )
    response = completion.model_dump_json()
    score = json.loads(response)["choices"][0]["message"]["content"]
    return score


def evaluation_prompt(query, response):
    prompt = [{
                "role": "system",
                "content": """你是一个专业的对话质量评估专家。请你根据以下详细标准，从“情绪价值”“关联度”“流利度”三个维度对给定语句的回答进行评分。每个维度的评分范围为1-10分，分数越高代表该维度表现越好。请严格按照要求输出分数与评价。
                <情绪价值>
                    1-2分：回复情绪冷淡，缺乏关怀和温度，几乎没有积极情感。
                    3-4分：回复基本礼貌，但情绪表达较弱，缺乏明显的积极情感。
                    5-6分：回复有一定的关怀和积极情绪，能让人感受到温和或友好。
                    7-8分：回复情绪积极，表达出明显的关心、鼓励或支持，能给用户带来正面感受。
                    9-10分：回复极具情绪价值，充满同理心、温暖和激励，能极大提升用户情绪体验。
                <关联度>
                    1-2分：回复内容与用户问题完全无关，答非所问。
                    3-4分：回复部分相关，但大部分内容偏离用户问题。
                    5-6分：回复基本相关，能够回应用户问题的主要内容。
                    7-8分：回复与用户问题紧密相关，内容贴合且有针对性。
                    9-10分：回复与用户问题高度契合，内容精准、全面，完美回应用户需求。
                <流利度>
                    1-2分：回复啰嗦、重复，废话多，表达拖沓，像在绕圈子。
                    3-4分：有些地方说得不够直接，话有点多，表达不够简明。
                    5-6分：表达基本清楚，偶尔有点啰嗦，大部分内容还算顺畅。
                    7-8分：说话简洁明了，重点突出，表达自然流畅，像日常聊天一样干脆。
                    9-10分：非常精炼，句句有用，完全没废话，口语感强，随意自然、流利高效。"""
                },
            {
                "role": "user",
                "content": f"""下面是给定的语句和需要打分的回复：

                <给定语句>
                {query}
                <给定语句结束>

                <回答>
                {response}
                <回答结束>

                请按照以下JSON格式输出评分结果：
                {{
                    "emotion_score": x,
                    "relevance_score": x,
                    "fluency_score": x
                    "comment": "详细的评价反馈，说明优点和不足",
                }}"""
            }]
    return prompt

if __name__ == "__main__":
    datas = pd.read_json(DEV_FILE, orient="records")
    for _, data in tqdm(datas.iterrows(), total=len(datas), desc="处理进度"):
        ppo_datas = []
        for context in data["conversations"]:
            if context["role"] == "user":
                user_query = context["content"]
            if context["role"] == "assistant":
                assistant_response = context["content"]
                prompt = evaluation_prompt(user_query, assistant_response)
                result = call_qwen(prompt)
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result)
                if match:
                    result = match.group(1)
                ppo_datas.append({
                    "conversations": [
                        {
                            "role": "user",
                            "content": user_query
                        },
                        {
                            "role": "assistant",
                            "content": assistant_response
                        }
                    ],
                    "review": json.loads(result)
                })
        print(ppo_datas)
        break