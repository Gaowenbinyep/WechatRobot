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


def build_prompt(query):
    prompt = [{
                "role": "system",
                "content": f"""你是一位语言风格模拟专家。现在我会提供10条某个人的真实语句，请你仔细分析这些语句的表达习惯、用词风格、语气特点和内容主题。
                    你的任务是：根据这10条语句的风格和内容，泛化生成3条新的语句。
                    要求如下：
                    新生成的每条语句都要尽量贴合原有说话人的表达风格和习惯，体现出相似的语气、用词和逻辑。
                    新语句的内容要与原有语句主题相关，但不能与原句重复或简单改写，要有一定创新和变化。
                    每条新语句要自然、通顺，符合日常交流习惯。
                    只输出3条新语句，每条单独成行，不要输出任何解释或多余内容。
                    请根据以上要求，完成泛化任务。"""
                },
            {
                "role": "user",
                "content": f"""下面是给定的10条语句：

                <给定语句>
                {query}
                <给定语句结束>

                请按照以下JSON格式输出评分结果：
                {{
                    "query": "新语句1",
                    "query": "新语句2",
                    "query": "新语句3"
                }}"""
            }]
    return prompt

if __name__ == "__main__":
    datas = pd.read_json(INPUT_FILE, lines=True, orient="records")
    ##TODO: 生成测试数据
    