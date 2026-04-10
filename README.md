## 空调维修诊断系统
基于Bert+Deepseek的空调维修诊断系统

## 项目说明
* 整个项目是针对已经给的空调数据，利用Bert+Deepseek最后生成诊断系统。输入故障描述，预测维修内容和预测故障元器件，最后生成维修建议。
* 准确率最后为维修内容84.74%，故障原因76.10%
* ![image](https://github.com/user-attachments/assets/11ebb73f-4090-4f3d-8ebe-d728ba5edc76)
* ![image](https://github.com/user-attachments/assets/5dc5fc08-9346-4621-a7b2-7715bda3b60d)
* ![image](https://github.com/user-attachments/assets/3b48158f-8f3a-4dca-b887-67b4a2b9b599)

> 这是一个结合了 **BERT (小模型)** 的精准分类能力与 **DeepSeek (大模型)** 的逻辑推理能力的垂直领域 AI 智能体。旨在解决空调维修场景中故障定位难、专家知识难以复用的问题。

## 项目背景 (Background)
在空调售后维修中，用户描述往往模糊（如“只吹风不凉”），而维修记录数据是非结构化的。本项目通过清洗上万条真实维修数据，构建了一套能够从自然语言描述中自动提取**故障元器件**与**维修方案**，并生成**标准化作业指导 (SOP)** 的辅助系统。

## 核心功能 (Key Features)
* **双模型精准定位:** 采用两个微调后的 `bert-base-chinese` 模型，分别预测：
    * **维修动作 (Repair Action):** 准确率优化中 (e.g., 更换电容, 加注制冷剂)
    * **故障根因 (Fault Component):** 基于历史数据的关联映射修正
* **专家级建议生成:** 集成 DeepSeek-V3 API，利用 Prompt Engineering 技术，根据 BERT 的分类结果生成带有人文关怀的操作步骤。
* **逻辑自洽性校验:** 引入 `Mapping` (映射表) 机制，利用历史数据中“故障-维修”的共现概率，修正模型预测的逻辑冲突。
* **可视化交互终端:** 基于 PyQt5 开发的桌面端应用，支持自然语言实时诊断。

## 技术架构 (Tech Stack)

### 1. 模型层 (Model Layer)
* **Backbone:** `bert-base-chinese` (HuggingFace Transformers)
* **LLM:** DeepSeek-Chat API (Context-Aware Prompting)
* **Optimization:** AdamW Optimizer, CrossEntropyLoss

### 2. 数据层 (Data Layer)
* **Preprocessing:** `pandas` 清洗, `LabelEncoder` 标签序列化
* **Constraint:** 基于统计学的故障-维修反向映射 (Reverse Mapping)

### 3. 应用层 (Application Layer)
* **GUI:** PyQt5 (Python Qt)
* **Network:** Requests (带重试机制的 HTTP Adapter)

## 文件说明
* 所需要的python库及对应版本在requirements.txt
* 所需要的空调维修数据在研究生作业数据.csv(课程老师提供的数据）
* 首先进行数据预处理对应代码yuchuli.py
* 接着进行模型训练对应代码train.py，需要bert-base-chinese模型，需要去官网下载，并且更改train.py里的文件路径
https://huggingface.co/google-bert/bert-base-chinese
* 随后进行后处理，建立映射表对应代码houchuli.py
* 结果展示预测准确率，分别是整体准确率和分类别准确率对应代码zhengti.py和ceshi.py
* 最后页面展示对应代码yemian.py，需要把deepseek的api替换成自己的api，api需要自己花钱在deepseek上购买
https://platform.deepseek.com/api_keys

## 硬件说明
代码没有用GPU跑，当时一直说GPU内存不够，用的CPU跑，大概跑了22小时，因为数据很多，如果资源够，可以把代码修改一下换成GPU，这样更快一些
  
##  目录结构 (Project Structure)

```text
project/
├── data/                  # 数据处理模块
│   └── yuchuli.py         # 数据清洗、分词、LabelEncoding、训练集划分
├── models/                # 模型训练与定义
│   ├── train.py           # BERT 模型微调 (Fine-tuning) 训练脚本
│   ├── bert-base-chinese/ # 预训练 BERT 权重目录
│   ├── fault_model/       # 训练好的故障元器件分类模型
│   └── repair_model/      # 训练好的维修内容分类模型
├── process/               # 后处理与逻辑优化
│   └── houchuli.py        # 建立故障与维修动作的映射表 (Mapping Construction)
├── test/                  # 模型评估
│   ├── ceshi.py           # 基础准确率测试脚本
│   └── zhengti.py         # 引入映射表修正后的整体准确率评估
└── page/                  # 客户端应用
    └── yemian.py          # PyQt5 主程序入口 (包含 BERT 推理 + DeepSeek 调用)



