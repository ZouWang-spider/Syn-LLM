import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Cross-Attention: triplet 查询 sentence
def cross_attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, value)



# 定义余弦嵌入损失函数
cosine_loss = nn.CosineEmbeddingLoss()
# Triplet一致性检测模块
def Triplet_optimization(triplet_h, sentence_h, M_true):
    # 归一化向量（增强稳定性）
    triplet_h = F.normalize(triplet_h, p=2, dim=-1)
    sentence_h = F.normalize(sentence_h, p=2, dim=-1)

    # 自相关矩阵 M1
    M1 = torch.einsum("id,jd->ijd", triplet_h, triplet_h)  # [n, n, d]
    # Cross-Attention 得到 M2
    cross_out = cross_attention(triplet_h, sentence_h, sentence_h)
    cross_out = F.normalize(cross_out, p=2, dim=-1)
    M2 = torch.einsum("id,jd->ijd", cross_out, cross_out)

    # 拼接 → [n, n, 2d]
    M3_fusion = torch.cat([M1, M2], dim=-1)
    # 定义全连接层
    fc_layer = nn.Linear(M3_fusion.shape[-1], M2.shape[-1])
    # 应用全连接层：保持前两个维度不变，只对最后一个维度变换
    M3_predict = fc_layer(M3_fusion)  # shape: [5, 5, 768]

    # 假设 M3_predict 和 M_true 都是 [5, 5, 768]
    M3_flat = M3_predict.view(-1, M3_predict.shape[-1])  # [25, 768]
    Mtrue_flat = M_true.view(-1, M_true.shape[-1])  # [25, 768]
    # 创建标签，全为1，表示希望预测和真实方向一致
    target = torch.ones(M3_flat.size(0)).to(M3_flat.device)  # [25]
    # 计算损失
    triplet_loss = cosine_loss(M3_flat, Mtrue_flat, target)
    # print("Cosine Embedding Loss:", triplet_loss.item())

    return triplet_loss.item()


####################################  Test  ####################################
# # ht: T5解码器输出的三元组张量（[n, d]）
# triplet_h = torch.rand(5, 768)
# sentence_h = torch.rand(7, 768)
# M_true = torch.randn(5, 5, 768)
#
# triplet_loss = Triplet_optimization(triplet_h, sentence_h, M_true)
# print("Triplet Loss:", triplet_loss)

