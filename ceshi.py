import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pickle  # 添加这一行

# 1. 数据集类
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

# 2. 加载验证集数据
repair_val_df = pd.read_csv("repair_val.csv", encoding="utf-8-sig")
fault_val_df = pd.read_csv("fault_val.csv", encoding="utf-8-sig")

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

def evaluate_by_label(model, val_loader, encoder):
    device = torch.device("cpu")
    model.eval()
    label_correct = {}
    label_total = {}
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            for pred, label in zip(preds, labels):
                label = label.item()
                label_correct[label] = label_correct.get(label, 0) + (pred == label).item()
                label_total[label] = label_total.get(label, 0) + 1
    for label in label_correct:
        acc = label_correct[label] / label_total[label]
        category = encoder.inverse_transform([label])[0]
        print(f"类别: {category}, 准确率: {acc:.4f}, 样本数: {label_total[label]}")

with open("repair_encoder.pkl", "rb") as f:
    repair_encoder = pickle.load(f)
with open("fault_encoder.pkl", "rb") as f:
    fault_encoder = pickle.load(f)

print("维修内容按类别准确率...")
evaluate_by_label(repair_model, repair_val_loader, repair_encoder)
print("\n故障原因按类别准确率...")
evaluate_by_label(fault_model, fault_val_loader, fault_encoder)


