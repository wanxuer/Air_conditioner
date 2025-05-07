import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# 1. 加载数据
file_path = r"D:\c\桌面\nlp（4.28）\研究生作业数据.csv"
try:
    df = pd.read_csv(file_path, encoding="gbk")  # 先试gbk，失败再试utf-8
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="utf-8")  # 备选utf-8
print(df.head())  # 检查前几行数据是否正常

# 2. 删除缺失值
df = df.dropna(subset=["用户反映", "维修内容名称", "零部件故障原因"])

# 3. 清洗文本
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = text.replace(";", "").replace("。", "").replace("，", "")
    text = text.replace("故障描述：", "").replace(".", "")
    text = text.replace("不知热", "不制热").replace("不知郎也不制热", "不制热")
    return text

df["用户反映"] = df["用户反映"].apply(clean_text)
df["维修描述"] = df["维修描述"].apply(clean_text)
df["维修内容名称"] = df["维修内容名称"].apply(clean_text)
df["零部件故障原因"] = df["零部件故障原因"].apply(clean_text)

# 4. 合并输入特征
df["输入文本"] = df["用户反映"] + " " + df["维修描述"]

# 5. 标签编码
repair_encoder = LabelEncoder()
df["维修内容标签"] = repair_encoder.fit_transform(df["维修内容名称"])
fault_encoder = LabelEncoder()
df["故障原因标签"] = fault_encoder.fit_transform(df["零部件故障原因"])

with open("repair_encoder.pkl", "wb") as f:
    pickle.dump(repair_encoder, f)
with open("fault_encoder.pkl", "wb") as f:
    pickle.dump(fault_encoder, f)

# 6. 数据拆分
repair_train_df, repair_val_df = train_test_split(
    df[["输入文本", "维修内容标签"]], test_size=0.2, random_state=42
)
fault_train_df, fault_val_df = train_test_split(
    df[["输入文本", "故障原因标签"]], test_size=0.2, random_state=42
)

# 7. 保存结果（用utf-8-sig避免Windows乱码）
repair_train_df.to_csv("repair_train.csv", index=False, encoding="utf-8-sig")
repair_val_df.to_csv("repair_val.csv", index=False, encoding="utf-8-sig")
fault_train_df.to_csv("fault_train.csv", index=False, encoding="utf-8-sig")
fault_val_df.to_csv("fault_val.csv", index=False, encoding="utf-8-sig")

# 8. 验证输出
print("数据预处理完成！检查前几行数据：")
print(pd.read_csv("repair_train.csv", encoding="utf-8-sig").head())
