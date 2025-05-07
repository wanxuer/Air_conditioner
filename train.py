# import pandas as pd
#
# repair_train_df = pd.read_csv("repair_train.csv", encoding="utf-8-sig")
# fault_train_df = pd.read_csv("fault_train.csv", encoding="utf-8-sig")
#
# print("维修内容标签统计：")
# print(repair_train_df["维修内容标签"].describe())
# print("最小值:", repair_train_df["维修内容标签"].min())
# print("最大值:", repair_train_df["维修内容标签"].max())
# print("是否有NaN:", repair_train_df["维修内容标签"].isna().sum())
#
# print("\n故障原因标签统计：")
# print(fault_train_df["故障原因标签"].describe())
# print("最小值:", fault_train_df["故障原因标签"].min())
# print("最大值:", fault_train_df["故障原因标签"].max())
# print("是否有NaN:", fault_train_df["故障原因标签"].isna().sum())
#
# print("维修内容标签唯一值数量:", len(set(repair_train_df["维修内容标签"])))
# print("故障原因标签唯一值数量:", len(set(fault_train_df["故障原因标签"])))

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import os
from torch import optim

# 1. 加载预处理数据
repair_train_df = pd.read_csv("repair_train.csv", encoding="utf-8-sig")
repair_val_df = pd.read_csv("repair_val.csv", encoding="utf-8-sig")
fault_train_df = pd.read_csv("fault_train.csv", encoding="utf-8-sig")
fault_val_df = pd.read_csv("fault_val.csv", encoding="utf-8-sig")

# 2. 自定义数据集类
class ACDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained("D:/PycharmProjects/nlp/bert-base-chinese", local_files_only=True)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# 3. 创建数据集和DataLoader
repair_train_dataset = ACDataset(repair_train_df["输入文本"].tolist(), repair_train_df["维修内容标签"].tolist())
repair_val_dataset = ACDataset(repair_val_df["输入文本"].tolist(), repair_val_df["维修内容标签"].tolist())
fault_train_dataset = ACDataset(fault_train_df["输入文本"].tolist(), fault_train_df["故障原因标签"].tolist())
fault_val_dataset = ACDataset(fault_val_df["输入文本"].tolist(), fault_val_df["故障原因标签"].tolist())

repair_train_loader = DataLoader(repair_train_dataset, batch_size=1, shuffle=True)
repair_val_loader = DataLoader(repair_val_dataset, batch_size=1)
fault_train_loader = DataLoader(fault_train_dataset, batch_size=1, shuffle=True)
fault_val_loader = DataLoader(fault_val_dataset, batch_size=1)

# 4. 加载模型（修正num_labels）
repair_num_labels = repair_train_df["维修内容标签"].max() + 1  # 最大值+1
fault_num_labels = fault_train_df["故障原因标签"].max() + 1    # 最大值+1
print(f"维修内容分类数: {repair_num_labels}")
print(f"故障原因分类数: {fault_num_labels}")

repair_model = BertForSequenceClassification.from_pretrained(
    "D:/PycharmProjects/nlp/bert-base-chinese", num_labels=repair_num_labels, local_files_only=True
)
fault_model = BertForSequenceClassification.from_pretrained(
    "D:/PycharmProjects/nlp/bert-base-chinese", num_labels=fault_num_labels, local_files_only=True
)

# 5. 定义优化器
repair_optimizer = optim.AdamW(repair_model.parameters(), lr=2e-5, eps=1e-8)
fault_optimizer = optim.AdamW(fault_model.parameters(), lr=2e-5, eps=1e-8)

# 6. 训练函数
def train_model(model, train_loader, optimizer, num_epochs=3, accum_steps=4):
    device = torch.device("cpu")
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}, Loss: {loss.item() * accum_steps:.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
def evaluate(model, val_loader):
    device = torch.device("cpu")
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    return avg_loss, accuracy

# 在train_model后添加
print("训练维修内容模型...")
train_model(repair_model, repair_train_loader, repair_optimizer, num_epochs=3)
val_loss, val_acc = evaluate(repair_model, repair_val_loader)
print(f"维修内容验证集 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
