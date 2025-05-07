import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pickle


# 1. 数据集类（与原 zhengti.py 保持一致）
class ACDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained("D:/PycharmProjects/nlp/bert-base-chinese",
                                                       local_files_only=True)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=64,
                                   return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# 2. 加载验证集和训练集数据
repair_val_df = pd.read_csv("repair_val.csv", encoding="utf-8-sig")
fault_val_df = pd.read_csv("fault_val.csv", encoding="utf-8-sig")
repair_train_df = pd.read_csv("repair_train.csv", encoding="utf-8-sig")
fault_train_df = pd.read_csv("fault_train.csv", encoding="utf-8-sig")

repair_val_dataset = ACDataset(repair_val_df["输入文本"].tolist(), repair_val_df["维修内容标签"].tolist())
fault_val_dataset = ACDataset(fault_val_df["输入文本"].tolist(), fault_val_df["故障原因标签"].tolist())

repair_val_loader = DataLoader(repair_val_dataset, batch_size=1)
fault_val_loader = DataLoader(fault_val_dataset, batch_size=1)

# 3. 加载模型
repair_model = BertForSequenceClassification.from_pretrained("D:/PycharmProjects/nlp/repair_model")
fault_model = BertForSequenceClassification.from_pretrained("D:/PycharmProjects/nlp/fault_model")

device = torch.device("cpu")
repair_model.to(device)
fault_model.to(device)

# 4. 加载编码器和映射表
with open("repair_encoder.pkl", "rb") as f:
    repair_encoder = pickle.load(f)
with open("fault_encoder.pkl", "rb") as f:
    fault_encoder = pickle.load(f)
with open("mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

# 5. 构建反向映射表（故障元器件 → 维修内容）
df_merged = repair_train_df.merge(fault_train_df, on="输入文本")
reverse_mapping_stats = df_merged.groupby("故障原因标签")["维修内容标签"].value_counts().unstack(fill_value=0)

reverse_mapping = {}
threshold = 1
for fault_label in reverse_mapping_stats.index:
    repair_counts = reverse_mapping_stats.loc[fault_label]
    repair_labels = repair_counts[repair_counts > threshold].index.tolist()
    if not repair_labels:
        max_count = repair_counts.max()
        if max_count > 0:
            repair_labels = [repair_counts.idxmax()]
        else:
            continue
    fault_name = fault_encoder.inverse_transform([fault_label])[0]
    repair_names = repair_encoder.inverse_transform(repair_labels)
    reverse_mapping[fault_name] = list(repair_names)

# 保存反向映射表（可选）
with open("reverse_mapping.pkl", "wb") as f:
    pickle.dump(reverse_mapping, f)

print("反向映射表：")
for fault_name, repair_names in reverse_mapping.items():
    print(f"{fault_name}: {repair_names}")


# 6. 原有评估函数：计算损失和准确率
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


# 7. 新增评估函数：使用反向映射表预测维修内容，计算整体准确率
def evaluate_repair_with_mapping(fault_model, repair_val_loader, fault_val_loader, repair_encoder, fault_encoder,
                                 reverse_mapping):
    device = torch.device("cpu")
    fault_model.eval()
    correct = 0
    total = 0

    # 将 repair_val_loader 的标签按输入文本对齐
    repair_val_dict = {}
    for batch in repair_val_loader:
        input_ids = batch["input_ids"].to(device)
        text = batch["input_ids"][0].cpu().numpy().tolist()  # 简化，用 input_ids 表示文本
        label = batch["labels"].item()
        repair_val_dict[tuple(text)] = label

    # 使用 fault_model 预测故障元器件，并通过反向映射表预测维修内容
    with torch.no_grad():
        for batch in fault_val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            text = batch["input_ids"][0].cpu().numpy().tolist()  # 简化，用 input_ids 表示文本

            # 预测故障元器件
            outputs = fault_model(input_ids, attention_mask=attention_mask)
            fault_pred = torch.argmax(outputs.logits, dim=1).item()
            fault_label = fault_encoder.inverse_transform([fault_pred])[0]

            # 通过反向映射表查找维修内容
            if fault_label in reverse_mapping:
                predicted_repairs = reverse_mapping[fault_label]  # 反向映射表返回多个可能的维修内容
            else:
                predicted_repairs = []  # 如果反向映射表中没有，设为空

            # 获取真实维修内容标签
            true_label = repair_val_dict.get(tuple(text))
            if true_label is None:
                continue  # 如果找不到对应的真实标签，跳过

            true_repair = repair_encoder.inverse_transform([true_label])[0]

            # 统计总样本数
            total += 1

            # 检查预测的维修内容是否包含真实标签
            if true_repair in predicted_repairs:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


# 8. 新增评估函数：使用映射表预测故障原因，计算整体准确率
def evaluate_fault_with_mapping(repair_model, repair_val_loader, fault_val_loader, repair_encoder, fault_encoder,
                                mapping):
    device = torch.device("cpu")
    repair_model.eval()
    correct = 0
    total = 0

    # 将 fault_val_loader 的标签按输入文本对齐
    fault_val_dict = {}
    for batch in fault_val_loader:
        input_ids = batch["input_ids"].to(device)
        text = batch["input_ids"][0].cpu().numpy().tolist()  # 简化，用 input_ids 表示文本
        label = batch["labels"].item()
        fault_val_dict[tuple(text)] = label

    # 使用 repair_model 预测维修内容，并通过映射表预测故障元器件
    with torch.no_grad():
        for batch in repair_val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            text = batch["input_ids"][0].cpu().numpy().tolist()  # 简化，用 input_ids 表示文本

            # 预测维修内容
            outputs = repair_model(input_ids, attention_mask=attention_mask)
            repair_pred = torch.argmax(outputs.logits, dim=1).item()
            repair_label = repair_encoder.inverse_transform([repair_pred])[0]

            # 通过映射表查找故障元器件
            if repair_label in mapping:
                predicted_faults = mapping[repair_label]  # 映射表返回多个可能的故障元器件
            else:
                predicted_faults = []  # 如果映射表中没有，设为空

            # 获取真实故障元器件标签
            true_label = fault_val_dict.get(tuple(text))
            if true_label is None:
                continue  # 如果找不到对应的真实标签，跳过

            true_fault = fault_encoder.inverse_transform([true_label])[0]

            # 统计总样本数
            total += 1

            # 检查预测的故障元器件是否包含真实标签
            if true_fault in predicted_faults:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


# 9. 运行评估
# 评估维修内容模型（直接预测）
print("评估维修内容模型（直接预测）...")
repair_val_loss, repair_val_acc = evaluate(repair_model, repair_val_loader)
print(f"维修内容验证集（直接预测） - Loss: {repair_val_loss:.4f}, Accuracy: {repair_val_acc:.4f}")

# 评估维修内容模型（使用反向映射表）
print("\n评估维修内容模型（使用反向映射表）...")
mapping_repair_acc = evaluate_repair_with_mapping(fault_model, repair_val_loader, fault_val_loader, repair_encoder,
                                                  fault_encoder, reverse_mapping)
print(f"维修内容验证集（使用反向映射表） - Accuracy: {mapping_repair_acc:.4f}")

# 对比维修内容预测的准确率
print("\n维修内容准确率对比：")
print(f"直接预测（repair_model）: {repair_val_acc:.4f}")
print(f"使用反向映射表: {mapping_repair_acc:.4f}")
print(f"准确率变化: {(mapping_repair_acc - repair_val_acc):.4f}")
if mapping_repair_acc > repair_val_acc:
    print("使用反向映射表提高了维修内容预测的准确率！")
elif mapping_repair_acc < repair_val_acc:
    print("使用反向映射表降低了维修内容预测的准确率。")
else:
    print("维修内容预测的准确率没有变化。")

# 评估故障原因模型（直接预测）
print("\n评估故障原因模型（直接预测）...")
fault_val_loss, fault_val_acc = evaluate(fault_model, fault_val_loader)
print(f"故障原因验证集（直接预测） - Loss: {fault_val_loss:.4f}, Accuracy: {fault_val_acc:.4f}")

# 评估故障原因模型（使用映射表）
print("\n评估故障原因模型（使用映射表）...")
mapping_fault_acc = evaluate_fault_with_mapping(repair_model, repair_val_loader, fault_val_loader, repair_encoder,
                                                fault_encoder, mapping)
print(f"故障原因验证集（使用映射表） - Accuracy: {mapping_fault_acc:.4f}")

# 对比故障原因预测的准确率
print("\n故障原因准确率对比：")
print(f"直接预测（fault_model）: {fault_val_acc:.4f}")
print(f"使用映射表: {mapping_fault_acc:.4f}")
print(f"准确率变化: {(mapping_fault_acc - fault_val_acc):.4f}")
if mapping_fault_acc > fault_val_acc:
    print("使用映射表提高了故障原因预测的准确率！")
elif mapping_fault_acc < fault_val_acc:
    print("使用映射表降低了故障原因预测的准确率。")
else:
    print("故障原因预测的准确率没有变化。")