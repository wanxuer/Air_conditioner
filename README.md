# 空调维修诊断系统
基于bert+deepseek的空调维修诊断系统

# 项目说明
* 整个项目是针对已经给的空调数据，利用bert+deepseek最后生成诊断系统。输入故障描述，预测维修内容和预测故障元器件，最后生成维修建议。
* 准确率最后为维修内容84.74%，故障原因76.10%
* ![image](https://github.com/user-attachments/assets/11ebb73f-4090-4f3d-8ebe-d728ba5edc76)
* ![image](https://github.com/user-attachments/assets/5dc5fc08-9346-4621-a7b2-7715bda3b60d)
* ![image](https://github.com/user-attachments/assets/3b48158f-8f3a-4dca-b887-67b4a2b9b599)

# 文件说明
* 所需要的python库及对应版本在requirements.txt
* 所需要的空调维修数据在研究生作业数据.csv
* 首先进行数据预处理对应代码yuchuli.py
* 接着进行模型训练对应代码train.py，需要bert-base-chinese模型，需要去官网下载，并且更改train.py里的文件路径
https://huggingface.co/google-bert/bert-base-chinese
* 随后进行后处理，建立映射表对应代码houchuli.py
* 结果展示预测准确率，分别是整体准确率和分类别准确率对应代码zhengti.py和ceshi.py
* 最后页面展示对应代码yemian.py，需要把deepseek的api替换成自己的api，api需要自己花钱在deepseek上购买
https://platform.deepseek.com/api_keys

# 硬件说明
代码没有用GPU跑，当时一直说GPU内存不够，用的CPU跑，大概跑了22小时，因为数据很多，如果资源够，可以把代码修改一下换成GPU，这样更快一些

# 结果展示
* 【基于Bert+DeepSeek的空调维修诊断系统-哔哩哔哩】 https://b23.tv/KBeWtf0
