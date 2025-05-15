import torch
from supar import Parser
from stanfordcorenlp import StanfordCoreNLP
from supar.utils.config import Config

# # 添加 Config 到可信类
# torch.serialization.add_safe_globals([Config])


#获取[(单词，词性),...]和依存关系[(单词1，依存，单词2),...]
nlp = StanfordCoreNLP(r'D:\StanfordCoreNLP\stanford-corenlp-4.5.4', lang='en')
parser = Parser.load('D:\BiAffine\ptb.biaffine.dep.lstm.char')
def Get_POS_DEP(sentence):
    text = nlp.word_tokenize(sentence)
    pos_tag = nlp.pos_tag(sentence)

    dataset = parser.predict([text], prob=True, verbose=True)

    # 获取arcs, rels以及probs
    arcs = dataset.arcs[0]  # 获取句法依赖关系中的arcs（依赖树的父节点索引）
    rels = dataset.rels[0]  # 获取句法关系类型（如det, nsubj等）
    words = text  # 获取句子中的词

    # 用来存储依赖关系的列表
    dependencies = []
    # 遍历arcs和rels，构建依赖关系
    for i, arc in enumerate(arcs):
        if arc != 0:  # 0表示该词没有父节点
            dep_relation = rels[i]  # 获取词语与父节点之间的关系
            word1 = words[i]  # 当前词语
            word2 = words[arc - 1]  # 父节点词语（arc是从1开始的）
            dependencies.append((word1, dep_relation, word2))

    # 构建句子的图，由弧-->节点
    arcs = dataset.arcs[0]  # 边的信息
    edges = [i + 1 for i in range(len(arcs))]
    for i in range(len(arcs)):
        if arcs[i] == 0:
            arcs[i] = edges[i]

    # 将节点的序号减一，以便适应DGL graph从0序号开始
    arcs = [arc - 1 for arc in arcs]
    edges = [edge - 1 for edge in edges]
    Dep_graph = (arcs, edges)
    # graph_line = '({}, {})\n'.format(graph[0], graph[1])  # 将图信息转为字符串
    # print("graph:", graph)
    # print(graph_line)

    return pos_tag, dependencies, Dep_graph




#获取序列的索引信息
def find_subsequence_index(full_tokens, sub_tokens):
    """
    在完整 token 列表中寻找子 token 列表的起止位置
    """
    for i in range(len(full_tokens) - len(sub_tokens) + 1):
        if full_tokens[i:i+len(sub_tokens)] == sub_tokens:
            return i, i + len(sub_tokens)
    return -1, -1

# 获取 anchor token 的索引
def extract_section_by_anchor_text(input_text: str, anchor1: str, anchor2: str, tokenizer):
    """
    通过字符串锚点在原始文本中找出对应段落的 token 和 hidden vector 区间。
    """
    # 找字符级位置
    idx1 = input_text.find(anchor1)
    idx2 = input_text.find(anchor2)

    if idx1 == -1 or idx2 == -1:
        raise ValueError("Anchor not found!")

    # 截取该段字符串
    section_text = input_text[idx1 + len(anchor1):idx2].strip()
    section_tokens = tokenizer.tokenize(section_text)
    section_ids = tokenizer(section_text, return_tensors="pt").input_ids[0][1:-1]  # 去掉 <pad> 和 </s>
    return section_text, section_tokens, section_ids


#获取sentence、task_prompt、noun_prompt、pos_prompt相对应的T5编码器Token
def Get_Prompt_token(input_text, tokenizer, tokens, hidden_states):

    # Anchor-based 分段提取
    sentence_text, sentence_tokens, sentence_ids = extract_section_by_anchor_text(input_text, "Sentence:", "Extract", tokenizer)
    noun_text, noun_tokens, noun_ids = extract_section_by_anchor_text(input_text, "Noun:", "Part-of-speech:", tokenizer)
    pos_text, pos_tokens, pos_ids = extract_section_by_anchor_text(input_text, "Part-of-speech:","Dependency relations:", tokenizer)
    dep_text, dep_tokens, dep_ids = extract_section_by_anchor_text(input_text, "Dependency relations:", "Note:", tokenizer)

    # 输出验证
    # print("Sentence:", sentence_text)
    # print("Noun tokens:", noun_text)
    # print("POS tokens:", pos_text)
    # print("DEP tokens:", dep_text)

    # 找出子串位置
    sent_start, sent_end = find_subsequence_index(tokens, sentence_tokens)
    noun_start, noun_end = find_subsequence_index(tokens, noun_tokens)
    pos_start, pos_end = find_subsequence_index(tokens, pos_tokens)
    dep_start, dep_end = find_subsequence_index(tokens, dep_tokens)

    # 提取隐藏向量
    sentence_hidden = hidden_states[0, sent_start:sent_end, :]
    noun_hidden = hidden_states[0, noun_start:noun_end, :]
    pos_hidden = hidden_states[0, pos_start:pos_end, :]
    dep_hidden = hidden_states[0, dep_start:dep_end, :]

    # 输出验证
    # print("Sentence hidden:", sentence_hidden.shape)
    # print("Noun hidden:", noun_hidden.shape)
    # print("POS hidden:", pos_hidden.shape)
    # print("DEP hidden:", dep_hidden.shape)

    return sentence_hidden, noun_hidden, pos_hidden, dep_hidden














############################# Test ###################################
# sentence = "Easy to start up and does not overheat as much as the laptops "
# pos_tag, dependency, Dep_graph = Get_POS_DEP(sentence)
#
# print(pos_tag)
# print(dependency)
# print(Dep_graph)








