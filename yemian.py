import sys
import torch
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QLineEdit
from PyQt5.QtCore import Qt

class FaultDiagnosisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("空调故障诊断系统")
        self.setGeometry(100, 100, 600, 500)

        # 加载 BERT 模型和编码器
        self.device = torch.device("cpu")
        self.repair_model = BertForSequenceClassification.from_pretrained("D:/PycharmProjects/nlp/repair_model").to(self.device)
        self.fault_model = BertForSequenceClassification.from_pretrained("D:/PycharmProjects/nlp/fault_model").to(self.device)
        self.bert_tokenizer = BertTokenizer.from_pretrained("D:/PycharmProjects/nlp/bert-base-chinese", local_files_only=True)

        with open("repair_encoder.pkl", "rb") as f:
            self.repair_encoder = pickle.load(f)
        with open("fault_encoder.pkl", "rb") as f:
            self.fault_encoder = pickle.load(f)

        # 加载映射表
        with open("mapping.pkl", "rb") as f:
            self.mapping = pickle.load(f)

        # DeepSeek 官方 API 配置
        self.deepseek_api_key = "输入申请的deepseek-api"
        self.deepseek_api_url = "https://api.deepseek.com/chat/completions"

        # 创建带重试机制的 Session
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        # 设置界面布局
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 故障描述输入
        layout.addWidget(QLabel("请输入故障描述："))
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("例如：空调不制冷")
        layout.addWidget(self.input_text)

        # 诊断按钮
        self.diagnose_button = QPushButton("诊断")
        self.diagnose_button.clicked.connect(self.diagnose)
        layout.addWidget(self.diagnose_button)

        # 维修内容输出（DeepSeek 预测）
        layout.addWidget(QLabel("预测的维修内容（由 DeepSeek 预测）："))
        self.repair_output = QLineEdit()
        self.repair_output.setReadOnly(True)
        layout.addWidget(self.repair_output)

        # 故障元器件输出（DeepSeek 预测）
        layout.addWidget(QLabel("预测的故障元器件（由 DeepSeek 预测）："))
        self.fault_output = QTextEdit()
        self.fault_output.setReadOnly(True)
        layout.addWidget(self.fault_output)

        # DeepSeek 生成的维修建议输出
        layout.addWidget(QLabel("详细维修建议（由 DeepSeek-V3 生成）："))
        self.deepseek_output = QTextEdit()
        self.deepseek_output.setReadOnly(True)
        layout.addWidget(self.deepseek_output)

        layout.addStretch()

    def predict_with_bert(self, text, repair_model, fault_model, repair_encoder, fault_encoder):
        repair_model.eval()
        fault_model.eval()
        encodings = self.bert_tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        with torch.no_grad():
            repair_outputs = repair_model(input_ids, attention_mask=attention_mask)
            fault_outputs = fault_model(input_ids, attention_mask=attention_mask)
            repair_pred = torch.argmax(repair_outputs.logits, dim=1).item()
            fault_probs = torch.softmax(fault_outputs.logits, dim=1)
            repair_confidence = torch.softmax(repair_outputs.logits, dim=1)[0, repair_pred].item()

        repair_label = repair_encoder.inverse_transform([repair_pred])[0]

        # 后处理：获取映射表中的所有可能故障元器件
        fault_labels = []
        fault_confidences = []
        if repair_label in self.mapping:
            fault_labels = self.mapping[repair_label]
            for fault_label in fault_labels:
                fault_idx = fault_encoder.transform([fault_label])[0]
                confidence = fault_probs[0, fault_idx].item()
                fault_confidences.append(confidence)
        else:
            fault_pred = torch.argmax(fault_outputs.logits, dim=1).item()
            fault_label = fault_encoder.inverse_transform([fault_pred])[0]
            fault_confidence = fault_probs[0, fault_pred].item()
            fault_labels = [fault_label]
            fault_confidences = [fault_confidence]

        return repair_label, fault_labels

    def predict_with_deepseek(self, fault_description, initial_repair_label, initial_fault_labels):
        # 构造提示，提供更多上下文并明确要求详细建议
        fault_str = "、".join(initial_fault_labels)
        prompt = (
            f"你是一个专业的空调维修专家。以下是用户输入的故障描述和初步预测结果，请重新预测‘维修内容’和‘故障元器件’，并提供详细的维修建议。\n"
            f"故障描述：{fault_description}\n"
            f"初步预测的维修内容（仅供参考）：{initial_repair_label}\n"
            f"初步预测的故障元器件（仅供参考）：{fault_str}\n"
            f"可能的维修内容类别：{', '.join(self.repair_encoder.classes_)}\n"
            f"可能的故障元器件类别：{', '.join(self.fault_encoder.classes_)}\n"
            f"请按以下格式输出，确保每个部分都有内容：\n"
            f"维修内容：<从可能的维修内容类别中选择一个最合适的维修内容>\n"
            f"故障元器件：<从可能的故障元器件类别中选择一个或多个最合适的故障元器件，用‘、’分隔>\n"
            f"详细维修建议：<提供具体的维修建议，至少 50 字，包含操作步骤和注意事项>\n"
        )

        # 调用 DeepSeek 官方 API
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",  # 使用 DeepSeek-V3 模型
            "messages": [
                {"role": "system", "content": "你是一个专业的空调维修专家，提供准确的故障诊断和详细的维修建议。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": False
        }
        try:
            # 使用带重试机制的 Session，增加超时时间
            response = self.session.post(self.deepseek_api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            print("DeepSeek API 响应：", result)  # 打印原始响应，方便调试

            # 解析 DeepSeek 的输出
            output = result["choices"][0]["message"]["content"].strip()
            print("DeepSeek 输出内容：", output)  # 打印 DeepSeek 的输出内容

            # 更鲁棒的解析逻辑
            repair_label = ""
            fault_labels = []
            advice = ""
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("维修内容："):
                    repair_label = line.replace("维修内容：", "").strip()
                elif line.startswith("故障元器件："):
                    fault_labels = line.replace("故障元器件：", "").strip().split("、")
                elif line.startswith("详细维修建议："):
                    # 提取建议部分，可能跨多行
                    advice_lines = []
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip():
                            advice_lines.append(lines[j].strip())
                        else:
                            break
                    advice = " ".join(advice_lines)

            # 如果未找到详细建议，尝试从剩余内容中提取
            if not advice:
                remaining_lines = []
                for line in lines:
                    if not (line.startswith("维修内容：") or line.startswith("故障元器件：") or line.startswith("详细维修建议：")):
                        remaining_lines.append(line.strip())
                advice = " ".join(remaining_lines).strip()

            # 确保所有字段都有内容
            if not repair_label:
                repair_label = "未预测到维修内容"
            if not fault_labels:
                fault_labels = ["未预测到故障元器件"]
            if not advice:
                advice = "未生成详细维修建议，请检查 DeepSeek API 返回内容或调整提示。"

            return repair_label, fault_labels, advice
        except requests.exceptions.Timeout:
            return "错误：API 请求超时", [], "请检查网络连接或稍后重试。"
        except requests.exceptions.RequestException as e:
            return "错误：API 调用失败", [], f"调用 DeepSeek API 失败：{str(e)}"
        except (KeyError, ValueError) as e:
            return "错误：解析响应失败", [], f"DeepSeek 返回的响应格式不正确：{str(e)}"

    def diagnose(self):
        fault_description = self.input_text.toPlainText().strip()
        if not fault_description:
            self.repair_output.setText("错误：请输入故障描述")
            self.fault_output.setText("")
            self.deepseek_output.setText("")
            return
        try:
            # 使用 BERT 提供初步预测
            initial_repair_label, initial_fault_labels = self.predict_with_bert(
                fault_description, self.repair_model, self.fault_model, self.repair_encoder, self.fault_encoder
            )

            # 使用 DeepSeek 重新预测并生成建议
            repair_label, fault_labels, advice = self.predict_with_deepseek(
                fault_description, initial_repair_label, initial_fault_labels
            )

            # 显示结果
            self.repair_output.setText(repair_label)
            self.fault_output.setText("\n".join(fault_labels))
            self.deepseek_output.setText(advice)
        except Exception as e:
            self.repair_output.setText(f"错误：{str(e)}")
            self.fault_output.setText("")
            self.deepseek_output.setText("")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaultDiagnosisWindow()
    window.show()
    sys.exit(app.exec_())