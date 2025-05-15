from supar import Parser
import nltk
import dgl
import networkx as nx
import torch
import matplotlib.pyplot as plt
from stanfordcorenlp import StanfordCoreNLP

#使用BiAffine对句子进行处理得到arcs、rels、probs
text = nltk.word_tokenize('The food is good')
# 词性标注
pos_tags = nltk.pos_tag(text)
print(pos_tags)

sentence = "The food is good"                   #"Battery life is good but screen is faulty"
#Using Stanford Parsing to get Word and Part-of-speech
nlp = StanfordCoreNLP(r'D:\StanfordCoreNLP\stanford-corenlp-4.5.4', lang='en')
print(nlp.pos_tag(sentence))

# 加载本地模型
parser = Parser.load('D:\BiAffine\ptb.biaffine.dep.lstm.char')   #'biaffine-dep-roberta-en'解析结果更准确
dataset = parser.predict([text], prob=True, verbose=True)

print(dataset.sentences[0])
print(f"arcs:  {dataset.arcs[0]}\n"
      f"rels:  {dataset.rels[0]}\n"
      f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")

#构建句子的图，由弧-->节点
arcs = dataset.arcs[0]  # 边的信息
edges = [i + 1 for i in range(len(arcs))]
for i in range(len(arcs)):
      if arcs[i] == 0:
            arcs[i] = edges[i]

#将节点的序号减一，以便适应DGL graph从0序号开始
arcs = [arc - 1 for arc in arcs]
edges = [edge - 1 for edge in edges]
graph = (arcs,edges)
graph_line = '({}, {})\n'.format(graph[0], graph[1])  # 将图信息转为字符串
print("graph:", graph)
print(graph_line)




