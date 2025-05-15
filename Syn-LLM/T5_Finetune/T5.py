import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 选择模型（也可用 't5-large', 't5-3b', 't5-11b'，但需更多显存）
# 加载模型和分词器
tokenizer = T5Tokenizer.from_pretrained("D:\T5_Large")
model = T5ForConditionalGeneration.from_pretrained("D:\T5_Large")
# 设置输入句子（T5 要求输入带任务提示）
input_text = """
 Extract all (aspect, opinion, sentiment) triplets from "Battery life is good but screen is faulty".
"""

# 编码输入
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print(input_ids.shape)

# 生成输出（解码器自动完成）
# 生成结果
output_ids = model.generate(
    input_ids,
    max_length=128,
    num_beams=4,
    early_stopping=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)
print(output_ids.shape)

# 解码输出
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("输入：", input_text)
print("输出：", generated_text)
