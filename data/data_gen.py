import sys
import os
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm
from openai import OpenAI, AsyncOpenAI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 上层目录路径
sys.path.append(parent_dir)

from PROMPT import gen_promot, parse_gen_result


MODEL = "qwen-plus" 
os.environ["DASHSCOPE_API_KEY"] = "<your_api_key>"

async def async_call_qwen(prompt):
    """ 异步调用 Qwen3-325B """
    client = AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=prompt,
        temperature=0.0,
        top_p=1.0,
        extra_body={"enable_thinking": False},
    )
    return completion.choices[0].message.content

async def data_generation(source_path, target_path):
    datas = pd.read_json(source_path, lines=True)
    scored_datas = []
    semaphore = asyncio.Semaphore(10)  # 控制最大并发量
    
    # 异步处理单个对话
    async def process_conversation(idx, data):
        async with semaphore:
            conversations = data["conversations"]
            query = conversations[1]["content"]
            response = conversations[2]["content"]
            prompt = gen_promot(query, response)
            retry = 0
            gen_conversations = None
            while retry < 3 and gen_conversations is None:
                try:
                    # 调用异步API（注意传递消息对象列表）
                    gen_content = await async_call_qwen([{"role": "user", "content": prompt}])
                    gen_conversations = parse_gen_result(gen_content)
                except Exception as e:
                    print(f"对话 {idx} 处理失败: {str(e)}")
                
                retry += 1
                if gen_conversations is None and retry < 3:
                    await asyncio.sleep(1)
            
            return {
                "conversations": conversations,
                "gen_conversations": gen_conversations
            }
    
    # 创建任务列表
    tasks = [process_conversation(idx, data) for idx, data in datas.iterrows()]
    
    # 异步进度条：使用tqdm.asyncio.gather包装任务
    scored_datas = await tqdm.gather(
        *tasks,
        total=len(tasks),
        desc="异步处理对话评分"
    )
    
    # 过滤无效结果并保存
    valid_datas = [item for item in scored_datas if item is not None]
    pd.DataFrame(valid_datas).to_json(
        target_path,
        orient="records",
        lines=True,
        force_ascii=False
    )
    


if __name__ == "__main__":
    # source_path = "/media/a822/82403B14403B0E83/Gwb/WechatRobot/data/Gen/Single_train.json"
    # target_path = "/media/a822/82403B14403B0E83/Gwb/WechatRobot/data/Gen/Single_train_gen.json"
    # asyncio.run(data_generation(source_path, target_path))
    datas = pd.read_json("/media/a822/82403B14403B0E83/Gwb/WechatRobot/data/v1.0/LCCC_single_train.json", lines=True)
    new_datas = datas.sample(frac=1).reset_index(drop=True)
    # new_datas = []
    # for _, data in tqdm(datas.iterrows(), total=len(datas)):
    #     conversations = data["conversations"]
    #     system = conversations[0]["content"]
    #     user = conversations[1]["content"]
    #     assistant = conversations[2]["content"]
    #     new_datas.append({
    #             "conversations": [
    #                 {"role": "system", "content": system},
    #                 {"role": "user", "content": user},
    #                 {"role": "assistant", "content": assistant}
    #             ]
    #         })
        
    pd.DataFrame(new_datas).to_json(
        "/media/a822/82403B14403B0E83/Gwb/WechatRobot/data/PPO/Single_train_gen.json",
        orient="records",
        lines=True,
        force_ascii=False
    )
        
