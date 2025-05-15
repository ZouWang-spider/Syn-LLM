import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


# 移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载微调后的模型
model = T5ForConditionalGeneration.from_pretrained("./t5-aste-finetuned").to(device)
tokenizer = T5Tokenizer.from_pretrained("./t5-aste-finetuned")

# 输入新句子
test_sentence = "Task: Extract aspect, opinion, and sentiment triplets in the format <aspect> X <opinion> Y <sentiment> Z. The dishes were delicious but the waiters were rude."

# 编码
input_ids = tokenizer(test_sentence, return_tensors="pt").input_ids.to(device)

# 生成输出
outputs = model.generate(input_ids, max_length=128)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("生成的三元组输出：", generated_text)
