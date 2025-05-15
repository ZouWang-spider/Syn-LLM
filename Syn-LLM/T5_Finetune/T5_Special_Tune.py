# =================== 加载预训练好的 AdaLoRA_T5_Based_Full 模型 ====================
import torch

from SAGF.DateProcess.DataProcessV2 import Dataset_Process
from SAGF.BaseModel.PromptConstruct import Prompt_Construct


# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.is_available())


from peft import PeftConfig, PeftModel
from transformers import T5ForConditionalGeneration, AutoTokenizer

# 1. 加载配置
config = PeftConfig.from_pretrained("D:/Project/SAGF/Model_Path/AdaLoRA_T5_Based_Full")

# 2. 加载基础模型
base_model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)

# 3. 正确封装 PEFT 模型
finetuned_model = PeftModel.from_pretrained(
    model=base_model,  # 这里传模型对象，而不是字符串路径
    model_id="D:/Project/SAGF/Model_Path/AdaLoRA_T5_Based_Full",
    is_trainable=True  # 关键修复！
)

# # 4. 可选：切换 adapter（不一定需要调用这个，如果只有一个 adapter 默认就好）
# finetuned_model.set_active_adapter("default")  # 正确调用

# 5. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# 6. 放到设备上
finetuned_model.to(device)



# =================== 数据集路径设置 ====================
dataset_paths = {
    # "14lap": "D:/Project/SAGF/triplet_datav2/14lap/train_triplets.txt",
    # "14res": "D:/Project/SAGF/triplet_datav2/14res/train_triplets.txt",
    # "15res": "D:/Project/SAGF/triplet_datav2/15res/train_triplets.txt",
    "16res": "D:/Project/SAGF/triplet_datav2/16res/train_triplets.txt"
}


# 数据集处理和encoding
class TripletDataset():
    def __init__(self, file_path, tokenizer, max_input_len=512, max_target_len=128):
        self.data = Dataset_Process(file_path)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, triplet_labels, generative_label = self.data[idx]

        # 生成输入和目标文本
        input_text = Prompt_Construct(sentence)
        # print('输入prompt为', input_text)
        if isinstance(generative_label, list):
            target_text = " SSEP: ".join(generative_label)
        else:
            target_text = generative_label  # 已是字符串


        # 编码输入和目标
        input_encoding = self.tokenizer(input_text, padding="max_length", truncation=True,
                                        max_length=self.max_input_len, return_tensors="pt")
        target_encoding = self.tokenizer(target_text, padding="max_length", truncation=True,
                                         max_length=self.max_target_len, return_tensors="pt")

        #jiang输入转化为Token
        input_id = input_encoding["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_id[0])


        # 处理标签（忽略 PAD）
        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100  # 这个是为了在计算损失时忽略 PAD token



        return {
            'input_text': input_text,  # 真实目标文本字符串
            'token' : tokens,
            'input_ids': input_encoding["input_ids"][0],  # 变成 (seq_len,) 而非 (1, seq_len)
            'attention_mask': input_encoding["attention_mask"][0],
            'labels': labels[0],
            'target_text': target_text   # 真实目标文本字符串

        }




from sklearn.metrics import precision_score, recall_score, f1_score

def is_subsequence(target_words, pred_words):
    """
    判断 target_words 是否为 pred_words 的子序列
    """
    if not target_words:
        return False
    p_idx = 0
    for w in target_words:
        while p_idx < len(pred_words) and pred_words[p_idx] != w:
            p_idx += 1
        if p_idx == len(pred_words):
            return False
        p_idx += 1
    return True

#如果 targets 的所有词都按顺序出现在 predictions 中则预测正确
def compute_prf1(predictions, targets):
    # 如果传入的是单个字符串，转换为列表
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(targets, str):
        targets = [targets]

    y_true = []
    y_pred = []

    for pred, target in zip(predictions, targets):
        pred = pred.strip().lower()
        target = target.strip().lower()

        y_true.append(1 if target else 0)

        pred_words = pred.split()
        target_words = target.split()

        match = is_subsequence(target_words, pred_words)
        y_pred.append(1 if match else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1




# =================== 针对每个数据集单独精调 ====================
for dataset_name, path in dataset_paths.items():
    print(f"\n======== Finetuning on {dataset_name} ========")

    # 1. 加载数据
    train_dataset = TripletDataset(path, tokenizer)

    # 2. 重新初始化优化器
    optimizer = torch.optim.Adam(finetuned_model.parameters(), lr=3e-5)

    # 3. 微调循环
    for epoch in range(50):
        finetuned_model.train()
        total_loss = 0
        predictions, targets = [], []

        for idx in torch.randperm(len(train_dataset)):
            batch = train_dataset[idx]
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            labels = batch['labels'].unsqueeze(0).to(device)
            target_text = batch['target_text']

            outputs = finetuned_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # 预测解码
            predictions_text = tokenizer.decode(outputs.logits.argmax(dim=-1).squeeze(), skip_special_tokens=True)

            predictions.append(predictions_text)
            targets.append(target_text)
            print('predictions_text:', predictions_text)
            print('target_text:', target_text)


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 每个 epoch 打印 P/R/F1
        precision, recall, f1 = compute_prf1(predictions, targets)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

        # 保存结果
        with open(f"T5_Based_Full_16Res.txt", "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}\n")

    # 4. 保存该数据集专调后的模型
    output_dir = f"D:/Project/SAGF/Model_Path/AdaLoRA_T5_Based_Full_16Res"
    finetuned_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
