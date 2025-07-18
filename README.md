# 微信聊天机器人（基于Qwen3-1.7B）

## 项目简介

本项目基于 Qwen3-1.7B 大模型，构建了一个支持多轮对话的微信聊天机器人，具备数据生成、数据处理、自动评测等功能。项目集成了阿里云通义千问 API，支持本地和云端推理，适合个性化对话、数据标注、自动评测等多种场景。

---

## 目录
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [数据说明](#数据说明)
- [核心功能说明](#核心功能说明)
  - [对话机器人](#对话机器人)
  - [数据生成](#数据生成)
  - [数据处理与筛选](#数据处理与筛选)
  - [自动评测](#自动评测)
- [模型下载与部署](#模型下载与部署)
- [致谢与参考](#致谢与参考)

---

## 项目结构

```
WechatRobot/
├── Base_models/           # 预留基础模型存放目录
├── data/                  # 数据集目录（训练、验证、评分等）
├── LLaMA-Factory/         # LLaMA-Factory大模型微调与推理框架
├── Saved_models/          # 训练或微调后模型保存目录
├── logs/                  # 日志文件
├── data_generation.py     # 数据生成脚本
├── data_process.py        # 数据处理与筛选脚本
├── evaluation.py          # 自动评测脚本
├── start.py               # 聊天机器人主入口（命令行多轮对话）
├── model_deploy.sh        # 模型部署脚本
├── LLM_download.sh        # 预训练模型下载脚本
```

---

## 环境依赖

- Python >= 3.9
- openai
- pandas
- tqdm
- LLaMA-Factory 及其依赖（见 `LLaMA-Factory/requirements.txt`）

安装依赖（推荐虚拟环境）：

```bash
pip install -r LLaMA-Factory/requirements.txt
pip install openai pandas tqdm
```

---

## 快速开始

### 1. 模型准备
- 推荐使用阿里云通义千问 Qwen3-1.7B 及以上模型。
- 可通过 `LLM_download.sh` 脚本下载所需模型，或将模型权重放入 `Base_models/` 目录。

### 2. 数据准备
- 训练、验证、评分等数据均存放于 `data/` 目录，格式为 json 或 jsonl。
- 示例数据文件：
  - `train.json`：训练集
  - `dev.json`：验证集
  - `score_train.json`：带评分的训练集

### 3. 启动多轮对话机器人

```bash
python start.py
```
- 支持多轮对话，输入内容后回车即可获得回复，输入空行退出。

### 4. 数据生成与处理
- 数据生成：`python data_generation.py`
- 数据处理与筛选：`python data_process.py`

### 5. 自动评测
- 自动评测：`python evaluation.py`
- 支持对模型回复进行情感、相关性、流利度等多维度自动打分。

---

## 数据说明

- 所有数据文件均为标准 json 或 jsonl 格式，字段包括 `conversations`（多轮对话）、`score`（评分）、`review`（多维度评测结果）等。
- 具体格式可参考 `data/` 目录下样例文件。

---

## 核心功能说明

### 对话机器人（start.py）
- 支持单轮/多轮对话，调用 Qwen3-1.7B 模型 API。
- 支持本地模型和云端 API（默认使用openai）。
- 多轮对话历史自动维护，适合真实聊天场景。

### 数据生成（data_generation.py）
- 根据真实用户语料，自动生成风格一致的新对话数据，便于扩充训练集。
- 支持批量处理，输出格式与原始数据一致。

### 数据处理与筛选（data_process.py）
- 对原始对话数据进行质量评分与筛选，自动剔除低质量样本。
- 支持多轮对话相关性打分、单轮 query 抽取等功能。

### 自动评测（evaluation.py）
- 对模型回复进行情感价值、相关度、流利度等多维度自动打分。
- 支持批量评测，输出详细 JSON 格式评测结果，便于后续分析。

---

## 模型下载与部署

### 下载模型
- 可通过 `LLM_download.sh` 脚本自动下载 Qwen3-1.7B 等模型权重。
- 也可手动下载后放入 `Base_models/` 目录。

### 部署模型
- 推荐使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 进行本地微调与推理。
- 支持本地推理、API 调用、Docker 部署等多种方式。
- 详细用法请参考 `LLaMA-Factory/README_zh.md`。

---

## 致谢与参考

- 本项目部分功能基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 实现，感谢开源社区的支持。
- 通义千问 API 由阿里云提供。
- 参考数据与脚本请见本项目 `data/` 与 `LLaMA-Factory/examples/` 目录。

---

如有问题欢迎提 issue 或交流！
