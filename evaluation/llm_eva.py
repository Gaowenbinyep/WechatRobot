import pandas as pd
import os
import json
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI


os.environ["DASHSCOPE_API_KEY"] = "<your_api_key>"

# ======================================
# 配置区
MODEL = "qwen-plus"  # 替换成可用的模型名
DEV_FILE = "./test_result_V5.json"
EVALUATION_FILE = "./scored_result/result_V5.json"
CONCURRENT_TASKS = 5
# ======================================


async def call_qwen(prompt):
    """异步调用Qwen API获取评分"""
    client = AsyncOpenAI(  # 使用异步客户端
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = await client.chat.completions.create(  # 异步调用需加await
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_thinking": False},
    )
    response = completion.model_dump_json()
    score = json.loads(response)["choices"][0]["message"]["content"]
    return score

def evaluation_prompt(query: str, response: str) -> str:

    prompt = f"""
你是一名严格的语言风格评审员。请根据下列维度对候选回答进行评分，仅输出 JSON，不要解释。

[评分维度定义]  
1. Style Alignment(风格对齐度) —— 回答是否贴近示例风格。  
   - 0–2分: 完全不符合，风格生硬或偏离严重  
   - 2–4分: 部分符合，存在一定风格元素但不够自然  
   - 4–5分: 高度符合，整体自然且风格贴合示例  

2. Relevance(关联度) —— 回答与用户输入的相关性。  
   - 0–2分: 大部分内容无关或偏离主题  
   - 2–4分: 有一定相关性，但存在冗余或偏题内容  
   - 4–5分: 高度相关，内容紧扣用户输入  

3. Conciseness(简洁度) —— 回答是否简洁明了。  
   - 0–2分: 冗长、啰嗦，包含重复表达或过多无效信息 
   - 2–4分: 基本简洁，但仍有无效信息  
   - 4–5分: 非常简洁，直观明了  

4. Role Cognition Clarity (角色认知清晰度)  
   - 0-2分: 严重偏离对方“男朋友”角色定位，把自己当成“女朋友”或其他人
   - 2-4分: 基本符合对方“男朋友”角色，但存在偶尔模糊或轻微角色混乱
   - 4-5分: 完全以对方“男朋友”角色定位回复消息，自称和语气一致无误

[风格示例]  
    1) "嘿 好想我宝宝~",
    2) "要吃点甜甜的缓一缓嘛",
    3) "打工仔今天很听话呐",
    4) "不准不开心啦🙅🏻‍♂️",
    5) "抱抱宝贝，不生气啦",
    6) "男朋友来接宝贝下班啦！",
    7) "我的傻丫头今天就是厉害嘛",
    8) "可恶💢 我又想女朋友了",
    9) "我滴宝藏宝宝😊",
    10) "好叭好叭傻丫头说了算！",
    11) "宝贝不生气！男朋友错啦",
    12) "我们打工仔这么优秀，一定没问题的！",
    13) "报告🙋‍ 我想女朋友啦",
    14) "傻丫头不许喝太多酒啦",
    15) "没醋硬吃😗",
    16) "我知道错了嘛😗", 

[本轮对话]  
User: {query}  
Assistant(candidate): {response}  

[输出JSON格式]  
{{
    "style_alignment": 0-5,
    "relevance": 0-5,
    "conciseness": 0-5,
    "role_cognition_clarity": 0-5
}}  
仅输出以上JSON，禁止多余文本。
"""
    return prompt



async def process_single_row(row, semaphore):
    """异步处理单条数据并返回评分结果"""
    async with semaphore:  # 控制并发数量
        query = row["query"]
        response = row["response"]
        max_retries = 5  # 最大重试次数
        scores = None  # 初始化评分变量
        
        # 重试循环
        for attempt in range(max_retries):
            try:
                # 生成评分提示词并调用API
                prompt = evaluation_prompt(query, response)
                score_str = await call_qwen(prompt)
                scores = json.loads(score_str)
                break  # 成功获取评分，跳出重试循环
            except Exception as e:
                # 判断是否为最后一次尝试
                if attempt < max_retries - 1:
                    print(f"处理 {query} 失败（尝试 {attempt + 1}/{max_retries}），错误: {str(e)}，将重试...")
                    await asyncio.sleep(1)  # 重试前等待1秒（减轻API压力）
                else:
                    print(f"处理 {query} 失败（所有 {max_retries} 次尝试均失败），错误: {str(e)}")
        
        # 所有重试失败时使用默认零分
        if scores is None:
            scores = {"style_alignment": 0, "relevance": 0, "conciseness": 0, "role_cognition_clarity": 0}
        
        return {
            "query": query,
            "response": response,
            "true_response": row["true_response"],
            "scores": scores
        }

async def main():
    # 读取数据
    datas = pd.read_json(DEV_FILE, lines=True, orient="records")
    
    # 创建信号量控制并发数
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    
    # 创建所有数据处理任务
    tasks = [process_single_row(row, semaphore) for _, row in datas.iterrows()]
    
    # 并发执行任务并显示进度
    ppo_datas = await async_tqdm.gather(*tasks, desc="异步评分进度")
    
    # 保存结果到JSON文件
    with open(EVALUATION_FILE, "w", encoding="utf-8") as f:
        json.dump(ppo_datas, f, ensure_ascii=False, indent=2)
    
    # 计算并输出四个维度平均得分
    if ppo_datas:
        total = {k: 0.0 for k in ppo_datas[0]["scores"].keys()}
        for item in ppo_datas:
            for k, v in item["scores"].items():
                total[k] += v
        
        avg_scores = {k: round(v/len(ppo_datas), 2) for k, v in total.items()}
        print("\n===== 维度平均得分 =====")
        for dim, score in avg_scores.items():
            print(f"{dim}: {score}")

if __name__ == "__main__":
    asyncio.run(main())  # 运行异步主函数