# Air_conditioner
Air conditioner diagnostic system based on bert+deepseek

# 项目说明
整个项目是针对已经给的空调数据，利用bert+deepseek最后生成诊断系统。输入故障描述，预测维修内容和预测故障元器件，最后生成维修建议。
![image](https://github.com/user-attachments/assets/11ebb73f-4090-4f3d-8ebe-d728ba5edc76)

# 文件说明
所需要的python库及对应版本在requirements.txt
所需要的空调维修数据在研究生作业数据.csv
首先进行数据预处理对应代码yuchuli.py
接着进行模型训练对应代码train.py，需要bert-base-chinese模型，需要去官网下载，并且更改train.py里的文件路径
https://huggingface.co/google-bert/bert-base-chinese
随后进行后处理，建立映射表对应代码houchuli.py
结果展示预测准确率，分别是整体准确率和分类别准确率对应代码zhengti.py和ceshi.py
最后页面展示对应代码yemian.py，需要把deepseek的api替换成自己的api
