import torch
import torch.nn as nn
from peft import PeftModel, PeftConfig
from peft import AutoPeftModelForSeq2SeqLM
from peft import get_peft_model, AdaLoraConfig, TaskType
from transformers.modeling_outputs import BaseModelOutput
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import precision_recall_fscore_support

from SAGF.DateProcess.DataProcessV2 import Dataset_Process
from SAGF.BaseModel.PromptConstruct import Prompt_Construct
from SAGF.BaseModel.Dependency_POS import Get_Prompt_token
from SAGF.BaseModel.Sem_Dyn_Rout import SemanticDynamicRoutingModule
from SAGF.BaseModel.Triplet_optimization import Triplet_optimization


# 选择模型（使用 't5-small', 't5-based','t5-large', 但需更多显存 't5-3b', 't5-11b'）

# 加载模型和分词器，在四个联合数据集上进行微调
tokenizer = T5Tokenizer.from_pretrained("D:\T5_Based")
t5model = T5ForConditionalGeneration.from_pretrained("D:\T5_Based")


# 设置输入句子（T5 要求输入带任务提示）

# 2. 配置 AdaLoRA（你可以根据硬件和精度需求自定义）
adalo_config = AdaLoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,   # 对于 T5 是序列到序列
    r=8,                               # 降维维度（通常为 4~16）
    target_modules=["q", "v"],         # 针对注意力的 query 和 value 层
    lora_alpha=32,                     # 缩放因子
    lora_dropout=0.1,                  # Dropout 用于防止过拟合
    # init_r=6,
    # beta1=0.85,
    # beta2=0.85,
    # tinit=100,
    # tfinal=1000,
    # delta_t=10,
)

# 3. 将模型转换为 AdaLoRA 模式
finetuned_model = get_peft_model(t5model, adalo_config)
finetuned_model.print_trainable_parameters()


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


# # 准确率计算函数
# def compute_accuracy(predictions, targets):
#     correct = 0
#     total = 0
#     for pred, target in zip(predictions, targets):
#         # 这里你可以选择比较预测和目标的准确性
#         # 例如：如果预测完全匹配目标序列则认为是正确的
#         if pred == target:
#             correct += 1
#         total += 1
#     return correct / total if total > 0 else 0

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



# ---------- 微调训练开始 ----------
file_paths = [
    # r"D:\Project\SAGF\triplet_datav2\14lap\train_triplets.txt",
    r"D:\Project\SAGF\triplet_datav2\14res\train_triplets.txt",
    # r"D:\Project\SAGF\triplet_datav2\15res\train_triplets.txt",
    # r"D:\Project\SAGF\triplet_datav2\16res\train_triplets.txt"
]

# 加载并合并所有数据集
from torch.utils.data import ConcatDataset

datasets = [TripletDataset(path, tokenizer) for path in file_paths]
train_dataset = ConcatDataset(datasets)


# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finetuned_model.to(device)
print("GPU:", torch.cuda.is_available())

#语义动态路由模块初始化 T5 small:512   T5 based: 768   T5 lager: 1024
SDRmodel = SemanticDynamicRoutingModule(hidden_size=768, num_attention_heads=4).to(device)

# 训练循环
optimizer = torch.optim.Adam(finetuned_model.parameters(), lr=5e-5)  # 选择合适的学习率
num_epochs = 50

for epoch in range(num_epochs):
    total_loss = 0
    finetuned_model.train()
    predictions = []  # 用来存储所有预测文本
    targets = []  # 用来存储所有目标文本

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    for step, idx in enumerate(torch.randperm(len(train_dataset))):  # 打乱顺序
        batch = train_dataset[idx]

        input_text = batch['input_text']
        tokens = batch['token']
        input_ids = batch['input_ids'].unsqueeze(0).to(device)  # 添加 batch 维度
        attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
        labels = batch['labels'].unsqueeze(0).to(device)
        target_text = batch['target_text']


        ## Step 1: 编码器前向传播
        encoder_outputs = finetuned_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        #T5 编码器输出的隐藏向量
        encoder_hidden = encoder_outputs.last_hidden_state

        #######################  语义动态路由模块  #############################
        ##Step 2 获取 句子、名词prompt、pos prompt、 dep prompt的词向量
        sentence_hidden, noun_hidden, pos_hidden, dep_hidden = Get_Prompt_token(input_text, tokenizer, tokens, encoder_hidden)

        #Step 3 Semantic Dynamic Routing Module
        SDR_output = SDRmodel(sentence_hidden, noun_hidden, pos_hidden, dep_hidden)

        #zero padding
        pad = torch.zeros(encoder_hidden.size(1)-SDR_output.size(0), encoder_hidden.size(2), device=SDR_output.device)  # [pad_len, hidden_dim]
        SDR_output_full = torch.cat([SDR_output, pad], dim=0)  # [512, 512]
        SDR_output_expand = SDR_output_full.unsqueeze(0)  # [1, 512, 512]
        # print(SDR_output_expand.shape)

        # Step 4: 融合 encoder_hidden 和 SDR_output，例如拼接或加和
        fusion_hidden = encoder_hidden + SDR_output_expand  # (1, seq_len, hidden_size)
        # print(fusion_hidden.shape)

        # Step 5: 构造 encoder_outputs
        decoder_fusion = BaseModelOutput(last_hidden_state=fusion_hidden)

        # Step 6: 使用融合结果送入 decoder
        outputs = finetuned_model(
            encoder_outputs=decoder_fusion,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        ##################### Triplet Consistency Detecting ############################
        decoder_hidden = outputs.decoder_hidden_states
        #Step 2 使用三元组一致性检测模块
        # M_true = torch.randn(5, 5, 768)
        # triplet_loss = Triplet_optimization(decoder_hidden, sentence_hidden, M_true)

        # 使用T5编码器-解码器内部自动计算了交叉熵损失（CrossEntropyLoss）
        outputs = finetuned_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


        # 解码预测序列
        logits = outputs.logits
        predictions_text = tokenizer.decode(logits.argmax(dim=-1).squeeze(), skip_special_tokens=True)

        # 保存预测和目标文本
        predictions.append(predictions_text)
        targets.append(target_text)
        print('predictions_text:', predictions_text)
        print('target_text:', target_text)

        #T5编码器-解码器计算的损失
        t5_loss = outputs.loss

        # 联合损失返回
        # total_loss = 0.3 * t5_loss  + 0.3 * triplet_loss
        total_loss += t5_loss.item()

        print(f"Step {step}, Loss: {t5_loss.item():.4f}")

        t5_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    # 计算每个 Epoch 的准确率
    # accuracy = compute_accuracy(predictions, targets)

    # 计算每个 Epoch 的 P/R/F1
    precision, recall, f1 = compute_prf1(predictions, targets)

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    # print(f"Epoch {epoch + 1} Accuracy: {accuracy:.4f}")
    print(f"Epoch {epoch + 1} Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    with open("T5_based_14Res.txt", "a", encoding="utf-8") as f:
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}", file=f)


# 保存微调后的模型
output_dir = "D:/Project/SAGF/Model_Path/AdaLoRA_T5_Based_14Res"
finetuned_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
