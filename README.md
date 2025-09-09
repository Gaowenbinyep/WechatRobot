# WechatRobot

一个基于大语言模型（LLM）的智能微信机器人项目，支持工具调用（天气查询、网页搜索等）、自定义工作流和本地模型部署，可通过自然语言交互完成多种任务。
微调时，别忘了将<your_name>修改成你的名字~
---

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [核心功能](#核心功能)
- [环境配置](#环境配置)
- [使用指南](#使用指南)
- [工作流详解](#工作流详解)
- [模型微调与评估](#模型微调与评估)
- [常见问题](#常见问题)
- [依赖说明](#依赖说明)
- [致谢](#致谢)

---

## 项目简介

WechatRobot 是一个模块化的智能对话系统，核心功能包括：

- **工具集成**：通过 LangChain 框架整合天气查询、实时搜索、时间获取等工具。
- **工作流引擎**：基于 LangGraph 构建状态机工作流，支持用户输入、路由决策、工具调用和对话生成的自动化流程。
- **本地模型部署**：使用 vLLM 部署本地微调的 LLM 模型，支持高效推理。
- **数据处理与微调**：提供数据清洗、格式转换和基于 LLaMA-Factory 的模型微调脚本。

---

## 项目结构

WechatRobot/
├── Agent/                  
│   ├── graph_nodes/        
│   │   └── nodes.py        
│   ├── prompts/            
│   └── tools/              
│       ├── tools.py        
│       └── weather.py      
├── data/                   
│   ├── v0.0/               
│   └── LCCC/               
├── evaluation/             
│   ├── sim_eva.py          
│   └── test_result_*.json  
├── LLaMA-Factory/          
├── Saved_models/           
│   └── rlhf/4B_lora_PPO_V3/merged  
├── logs/                   
├── model_deploy.sh         
├── data_process.py         
└── requirements.txt        

---

## 核心功能

### 1. 工具调用能力

通过 LangChain 的 `@tool` 装饰器定义工具，支持自然语言触发：

- **天气查询**：调用 `get_weather(city)` 获取实时天气（基于 wttr.in API）。
- **网页搜索**：调用 `web_search(query)` 模拟搜索（百度 API）。
- **时间获取**：调用 `get_current_time()` 返回当前时间。

### 2. 工作流引擎

基于 LangGraph 构建状态机，核心节点包括：

- **输入节点**：接收用户输入。
- **路由节点**：通过 LLM 判断任务类型（直接对话 / 工具调用）。
- **工具节点**：调用 Agent 执行工具链（基于 ReAct 框架）。
- **对话节点**：生成自然语言响应（结合工具输出或直接对话）。

### 3. 本地模型部署

使用 vLLM 部署本地微调模型，支持高并发推理：

CUDA_VISIBLE_DEVICES=0 \
vllm serve ./Saved_models \
    --tensor-parallel-size 1 \
    --port 8888 \
    --max-model-len 3000

### 4. 模型微调与数据处理

- **数据处理**：`data_process.py` 支持对话数据格式转换（如 ShareGPT 格式）、清洗和增强。
- **微调框架**：集成 LLaMA-Factory，支持 LoRA 微调、RLHF 等训练策略，适配自定义对话数据。

---

## 环境配置

### 前置依赖

- Python 3.10+
- CUDA 11.7+（推荐，支持 GPU 推理）
- 依赖库：见 `requirements.txt`

### 安装步骤

cd /media/a822/82403B14403B0E83/Gwb/WechatRobot

pip install -r requirements.txt
pip install vllm  # 额外安装 vLLM

### 模型准备

- 将微调后的模型文件放入 `Saved_models`。
- 如需使用公开模型，可通过 LLaMA-Factory 下载并微调。

### 配置文件

修改 `Agent/graph_nodes/nodes.py` 中的模型路径和 API 配置：

model_name = "/media/a822/82403B14403B0E83/Gwb/WechatRobot/Saved_models"
openai_api_base = "http://localhost:8888/v1"  # vLLM 服务地址

---

## 使用指南

### 1. 启动模型服务

bash model_deploy.sh

服务默认运行在 `http://localhost:8888`，支持 OpenAI 兼容 API。

### 2. 运行智能体机器人

python Agent/main.py # 智能体入口文件
python start.py # 单纯对话入口文件

### 3. 交互示例

用户输入：北京天气怎么样？
机器人响应：
> 调用工具：get_weather(北京)
> 工具返回：北京：晴 25°C
> 最终回答：北京今天是晴天哦，像你一样温暖

---

## 工作流详解

### 核心状态与节点

class State(TypedDict):
    user_input: str       # 用户输入
    tools_output: str     # 工具调用结果
    route: str            # 路由决策 ("agent" 工具调用 / "chat" 直接对话)

#### 输入节点

def get_user_input_node(state: State) -> State:
    user_input = input()
    return {**state, "user_input": user_input}

#### 路由节点

def router_node(state: State) -> State:
    prompt_input = router_prompt.format(query=state["user_input"])
    route = router_llm.invoke(prompt_input).content
    return {**state, "route": route.strip().lower()}  # 返回 "agent" 或 "chat"

#### 工具节点

def tools_node(state: State) -> State:
    if state["route"] == "agent":
        tools_output = agent_executor.invoke({"input": state["user_input"]})
    return {**state, "tools_output": tools_output}

#### 对话节点

def chat_node(state: State) -> State:
    if state["route"] == "chat":
        prompt_input = chat_prompt.format(tools_output=state["tools_output"])
        answer = llm.invoke(prompt_input)
        print(answer.content)

---

## 模型微调与评估

### 数据准备

python data_process.py  # 转换数据至 ./data

### 微调流程

cd LLaMA-Factory
python train.py \
    --model_name_or_path /path/to/base_model \
    --lora_rank 16 \
    --data_path ../data/v1.0/Gen_single_train.json \
    --output_dir ../Saved_models/sft/8B_lora_V2

### 评估

python evaluation/sim_eva.py

---

## 依赖说明

- `langchain` / `langgraph`：工作流与工具链框架。
- `vllm`：高效 LLM 推理部署。
- `sentence-transformers`：文本相似度评估。
- `llama-factory`：模型微调工具。
- `requests`：网络请求（工具调用）。


