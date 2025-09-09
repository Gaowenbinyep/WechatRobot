from langchain.tools import tool
from openai import OpenAI
import requests
import datetime

@tool
def get_weather(city: str = "北京") -> str:
    """查询城市实时天气"""
    url = f"https://wttr.in/{city}?format=2"
    response = requests.get(url)
    return response.text if response.status_code == 200 else "查询失败"

@tool
def web_search(query: str) -> str:
    """
    智能搜索工具，返回搜索结果摘要。
    query: 用户查询内容
    """
    client = OpenAI(
        api_key="<your_api_key>", # 千帆AppBuilder平台的ApiKey      
        base_url="https://qianfan.baidubce.com/v2/ai_search") # 智能搜索生成V2版本接口
    response = client.chat.completions.create(
        model="ernie-4.0-turbo-8k",
        messages=[
            {"role": "user", "content": query}
        ],
        stream=False
    )
    return response.choices[0].message.content

@tool
def get_current_time(input: str = None) -> str:
    """获取当前时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    print(get_weather("北京"))
    print(web_search("第二次世界大战的时间"))
    print(get_current_time("当前时间"))