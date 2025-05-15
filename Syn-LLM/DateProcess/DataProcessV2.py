import re
import ast


# extraction triplet
def extract_triplets(sentence, triplet_indices):
    tokens = sentence.split()
    triplets = []
    for aspect_indices, opinion_indices, sentiment in triplet_indices:
        aspect = " ".join(tokens[i] for i in aspect_indices)
        opinion = " ".join(tokens[i] for i in opinion_indices)
        triplets.append((aspect, opinion, sentiment))
    return triplets


def Dataset_Process(file_path):
    data = []
    # red dataset
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # get Sentence
            parts = line.split("####")
            sentences = parts[0].strip()
            triplet_information = eval(parts[-1].strip())  # 使用 eval 将字符串解析为 Python 对象

            #获取三元组标签
            sentiment_mapping = {'NEG': 'negative', 'POS': 'positive', 'NEU': 'neutral'}
            # 遍历并替换情感标签
            triplet_labels = [
                (aspect, opinion, sentiment_mapping[sentiment])
                for aspect, opinion, sentiment in triplet_information
            ]

            tokens = sentences.strip().split()
            Generative_label = []

            for aspect_idx, opinion_idx, sentiment in triplet_labels:
                aspect_words = [tokens[i] for i in aspect_idx]
                opinion_words = [tokens[i] for i in opinion_idx]

                aspect = " ".join(aspect_words)
                opinion = " ".join(opinion_words)

                triplet_text = f"aspect: {aspect} opinion: {opinion} sentiment: {sentiment}"
                Generative_label.append(triplet_text)



            data.append((sentences, triplet_labels, Generative_label))
    return data


# ###########################  test  #################################
# file_path = r"D:\Project\SAGF\triplet_datav2\14lap\train_triplets.txt"
# datasets = Dataset_Process(file_path)
# # print(datasets)
# for sentence, triplet_labels, Generative_label in datasets:
#     print(sentence)
#     print(triplet_labels)
#     print(Generative_label)
#
