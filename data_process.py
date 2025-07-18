import pandas as pd
import os
import json
from tqdm import tqdm
from openai import OpenAI


os.environ["DASHSCOPE_API_KEY"] = "sk-xxx"

# ======================================
# 配置区
MODEL = "qwen-plus"  # 替换成可用的模型名
INPUT_FILE = "xxx.json"
OUTPUT_FILE = "xxx.json"
DEV_FILE = "xxx.json"
DEV_QUERY_FILE = "xxx.json"
GENERATION_QUERY_FILE = "xxx.json"
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


def score_prompt(conversations):
    prompt = [{
                "role": "system",
                "content": "你是一个专业的对话语料质量审校与评分助手，你的任务是：\n- 根据提供的多轮对话内容，判断这些对话内部的语义关联性是否紧密、是否连贯。\n- 只关注一条对话中assistant和user之间的各轮之间是否自然呼应、有连续上下文关系，不需要分析内容是否真实。\n- 请从 0 分到 5 分打分：\n  - 0 = 完全无关、毫无上下文关系\n  - 1 = 关联性极弱，内容跳跃、对话割裂\n  - 2 = 有一点点呼应，但大部分不连贯\n  - 3 = 一般，有明显上下文，但部分轮次脱节\n  - 4 = 关联性较强，基本完整且顺畅\n  - 5 = 关联性非常强，整条对话自然、完整、非常贴合\n请严格只输出一个整数分数，不需要解释。"
            },
            {
                "role": "user",
                "content": f"下面是需要打分的对话：\n<对话开始>\n{conversations}\n<对话结束>\n\n请直接给出分数："
            }]
    return prompt

if __name__ == "__main__":
    
    ### 对话数据打分
    # datas = pd.read_json(INPUT_FILE, orient="records")
    # with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    #     for _, data in tqdm(datas.iterrows(), total=len(datas), desc="处理进度"):
    #         conversations = data["conversations"]
    #         prompt = score_prompt(conversations)
    #         score = call_qwen(prompt)
    #         new_conversations = {
    #             "conversations": conversations,
    #             "score": score,
    #         }
    #         f.write(json.dumps(new_conversations, ensure_ascii=False) + "\n")

    ### 对话数据筛选
    # datas = pd.read_json(OUTPUT_FILE, lines=True, orient="records")
    # new_datas = []
    # for _, data in tqdm(datas.iterrows(), total=len(datas), desc="处理进度"):
    #     conversations = data["conversations"]
    #     score = data["score"]
    #     if score >= 3:
    #         new_datas.append(
    #             {
    #                 "conversations": conversations
    #             }
    #         )
    # new_datas = pd.DataFrame(new_datas).to_json(CLEAN_OUTPUT_FILE, orient="records", lines=True, force_ascii=False)

    ### 单轮对话测试数据提取
    datas = pd.read_json(DEV_FILE, orient="records")
    new_querys = []
    for _, data in tqdm(datas.iterrows(), total=len(datas), desc="处理进度"):
        querys = data["conversations"]
        for query in querys:
            if query["role"] == "user":
                new_querys.append(
                    {
                        "query": query["content"]
                    }
                )
    pd.DataFrame(new_querys).to_json(DEV_QUERY_FILE, orient="records", lines=True, force_ascii=False)