import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


# Semantic Dynamic Routing Module
class SemanticDynamicRoutingModule(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12):
        super(SemanticDynamicRoutingModule, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        # MultiheadAttention层（用于 Aspect-oriented Cross Attention 和 Aspect-Opinion Cross Attention）
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)

        # Add & Norm 层
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)

        # 前馈网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),  # 设定中间层维度为 4 倍
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # Dropout 层
        self.dropout = nn.Dropout(0.1)

    def forward(self, sentence_tensor, noun_prompt_tensor, pos_prompt_tensor, dep_prompt_tensor):
        # 拼接 noun_prompt_tensor 和 pos_prompt_tensor
        noun_pos_concat = torch.cat((noun_prompt_tensor, pos_prompt_tensor),dim=0)

        # Aspect-oriented Cross Attention (Q: sentence_tensor, K/V: noun_pos_concat)
        attn_output1, _ = self.cross_attn1(sentence_tensor, noun_pos_concat, noun_pos_concat)
        attn_output1 = self.layer_norm1(attn_output1 + sentence_tensor)  # Add & Norm

        # Aspect-Opinion Cross Attention (Q: attn_output1, K/V: dep_prompt_tensor)
        attn_output2, _ = self.cross_attn1(attn_output1, dep_prompt_tensor, dep_prompt_tensor)
        attn_output2 = self.layer_norm2(attn_output2 + attn_output1)  # Add & Norm

        # 前馈网络（FFN）
        ffn_output = self.ffn(attn_output2)
        ffn_output = self.layer_norm3(ffn_output + attn_output2)  # Add & Norm

        return ffn_output




############################  Test  ###########################################
# # 假设输入数据
# sentence_tensor = torch.rand(10, 768)
#
# noun_prompt_tensor = torch.rand(5, 768)
# pos_prompt_tensor = torch.rand(12, 768)
#
# dep_prompt_tensor = torch.rand(30, 768)
#
# # 创建模型
# SDRmodel = SemanticDynamicRoutingModule(hidden_size=768, num_attention_heads=12)
#
# #模型调用
# output = SDRmodel(sentence_tensor, noun_prompt_tensor, pos_prompt_tensor, dep_prompt_tensor)
#
# # 输出结果的形状
# print(output.shape)
