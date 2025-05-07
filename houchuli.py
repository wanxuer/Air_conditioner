import pandas as pd
df = pd.read_csv("repair_train.csv", encoding="utf-8-sig")
df_fault = pd.read_csv("fault_train.csv", encoding="utf-8-sig")
df_merged = df.merge(df_fault, on="输入文本")
mapping_stats = df_merged.groupby("维修内容标签")["故障原因标签"].value_counts().unstack(fill_value=0)
print(mapping_stats)
import pandas as pd
import pickle

# 加载标签编码器
with open("repair_encoder.pkl", "rb") as f:
    repair_encoder = pickle.load(f)
with open("fault_encoder.pkl", "rb") as f:
    fault_encoder = pickle.load(f)

# 加载训练数据并生成交叉表
repair_train_df = pd.read_csv("repair_train.csv", encoding="utf-8-sig")
fault_train_df = pd.read_csv("fault_train.csv", encoding="utf-8-sig")
df_merged = repair_train_df.merge(fault_train_df, on="输入文本")
mapping_stats = df_merged.groupby("维修内容标签")["故障原因标签"].value_counts().unstack(fill_value=0)

# 构建映射表，支持多标签
mapping = {}
threshold = 1  # 出现次数阈值，调整这个值以控制候选数量
for repair_label in mapping_stats.index:
    # 获取该维修内容标签对应的故障原因标签分布
    fault_counts = mapping_stats.loc[repair_label]
    # 选择出现次数大于阈值的故障原因标签
    fault_labels = fault_counts[fault_counts > threshold].index.tolist()
    if not fault_labels:  # 如果没有满足条件的标签，选择出现次数最多的
        max_count = fault_counts.max()
        if max_count > 0:
            fault_labels = [fault_counts.idxmax()]
        else:
            continue
    # 解码为实际类别名称
    repair_name = repair_encoder.inverse_transform([repair_label])[0]
    fault_names = fault_encoder.inverse_transform(fault_labels)
    mapping[repair_name] = list(fault_names)

# 保存映射表
with open("mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)

# 打印映射表以检查
print("生成的映射表：")
for repair_name, fault_names in mapping.items():
    print(f"{repair_name}: {fault_names}")