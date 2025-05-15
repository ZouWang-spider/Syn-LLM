
##################  dependency Label  ########################
dependency_relation_dict = {
    "abbrev": "abbreviation modifier",
    "acomp": "adjectival complement",
    "advcl": "adverbial clause modifier",
    "advmod": "adverbial modifier",
    "agent": "agent",
    "amod": "adjectival modifier",
    "appos": "appositional modifier",
    "attr": "attributive",
    "aux": "auxiliary",
    "auxpass": "passive auxiliary",
    "cc": "coordination",
    "ccomp": "clausal complement",
    "complm": "complementizer",
    "conj": "conjunct",
    "cop": "copula",
    "csubj": "clausal subject",
    "csubjpass": "clausal passive subject",
    "dep": "dependent",
    "det": "determiner",
    "dobj": "direct object",
    "expl": "expletive",
    "infmod": "infinitival modifier",
    "iobj": "indirect object",
    "mark": "marker",
    "mwe": "multi-word expression",
    "neg": "negation modifier",
    "nn": "noun compound modifier",
    "npadvmod": "noun phrase as adverbial modifier",
    "nsubj": "nominal subject",
    "nsubjpass": "passive nominal subject",
    "num": "numeric modifier",
    "number": "element of compound number",
    "parataxis": "parataxis",
    "partmod": "participial modifier",
    "pcomp": "prepositional complement",
    "pobj": "object of a preposition",
    "poss": "possession modifier",
    "possessive": "possessive modifier",
    "preconj": "preconjunct",
    "predet": "predeterminer",
    "prep": "prepositional modifier",
    "prepc": "prepositional clausal modifier",
    "prt": "phrasal verb particle",
    "punct": "punctuation",
    "purpcl": "purpose clause modifier",
    "quantmod": "quantifier phrase modifier",
    "rcmod": "relative clause modifier",
    "ref": "referent",
    "rel": "relative",
    "root": "root",
    "tmod": "temporal modifier",
    "xcomp": "open clausal complement",
    "xsubj": "controlling subject",
    "subj": "subject",
    "top": "topic",
    "npsubj": "nominal passive subject",
    "obj": "object",
    "range": "grams",
    "lobj": "Prepositions of time",
    "comp": "Complement",
    "tcomp": "temporal complement",
    "lccomp": "localizer complement",
    "mod": "modifier",
    "pass": "passive",
    "numod": "numeric modifier",
    "ornmod": "numeric modifier",
    "clf": "classifier modifier",
    "nmod": "noun compound modifier",
    "vmod": "verb modifier, participle modifier",
    "prnmod": "parenthetical modifier",
    "possm": "possessive marker",
    "dvpmod": "dvp modifier",
    "dvpm": "dvp marker",
    "assm": "associative marker",
    "assmod": "associative modifier",
    "clmod": "clause modifier",
    "plmod": "prepositional localizer modifier",
    "asp": "aspect marker",
    "etc": "etc",
    "cordmod": "coordinated verb compound",
    "mmod": "modal verb",
    "ba": "Complement relations",
    "tclaus": "Temporal clauses",
    "cpm": "complementizer"
}

# #依存关系词典
# tag = 'conj'
# explanation = dependency_relation_dict.get(tag, 'Unknown')
# print(explanation)


################## Part-of-speech Label  ########################
pos_tag_dict = {
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'DT': 'Determiner',
    'EX': 'Existential there',
    'FW': 'Foreign word',
    'IN': 'Preposition',
    'JJ': 'Adjective',
    'JJR': 'Adjective comparative',
    'JJS': 'Adjective superlative',
    'LS': 'List item marker',
    'MD': 'Modal',
    'NN': 'Noun',
    'NNS': 'Noun',
    'NNP': 'Proper noun',
    'NNPS': 'Proper noun',
    'PDT': 'Predeterminer',
    'POS': 'Possessive ending',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb',
    'RBS': 'Adverb',
    'RP': 'Particle',
    'SYM': 'Symbol',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb',
    'VBD': 'Verb',
    'VBG': 'Verb',
    'VBN': 'Verb',
    'VBP': 'Verb',
    'VBZ': 'Verb',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive wh-pronoun',
    'WRB': 'Wh-adverb'
}

# #词性词典
# tag = 'CC'
# explanation = pos_tag_dict.get(tag, 'Unknown')
# print(explanation)


from SAGF.BaseModel.Dependency_POS import Get_POS_DEP

def Prompt_Construct(sentence):
    # 获取[(单词，词性),...]和依存关系[(单词1，依存，单词2),...]
    pos_tag, dependency, dep_graph = Get_POS_DEP(sentence)

    # 将词性标签转化为 Part-of-speech Prompt
    # 构建POS Prompt
    pos_prompt = "Part-of-speech: \n"
    # 遍历每个词及其标签
    for i, (word, tag) in enumerate(pos_tag):
        explanation = pos_tag_dict.get(tag, 'Unknown')  # 根据tag获取词性解释
        if i == len(pos_tag) - 1:  # 最后一个单词，不加逗号
            pos_prompt += f' "{word}"/{explanation}.'
        else:
            pos_prompt += f' "{word}"/{explanation}, \n'
    # print(pos_prompt)    #"I"/PRP

    # 将依存标签转化为 Dependency Prompt
    dep_prompt = "Dependency relations: \n"
    # 遍历每个依存关系
    for i, (word1, rel, word2) in enumerate(dependency):
        explanation = dependency_relation_dict.get(rel, 'Unknown')  # 根据rel获取依存关系解释
        if i == len(dependency) - 1:  # 最后一个关系，不加逗号
            dep_prompt += f' "{word1}"/{explanation}/"{word2}".'
        else:
            dep_prompt += f' "{word1}"/{explanation}/"{word2}", \n'
    # print(dep_prompt)     #but/coord/good

    Note = "Note: Aspect terms are usually Noun, and opinion terms are usually Adjective. The dependency relation between aspect and opinion terms is often nominal subject."

    # 任务提示
    task_prompt =  "Extract all (aspect, opinion, sentiment) triplets from the Sentence, with the output format aspect: A opinion: B sentiment: C SSEP:, where SSEP: use to separate multiple triplets.\n (For example: aspect: food opinion: good sentiment: positive SSEP: aspect: drink opinion: bad sentiment: negative)"

    # Noun Prompt
    # doc = noun(sentence)
    # noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    # noun_prompt = 'Noun: ' + ', '.join([f'"{noun}"' for noun in noun_chunks])
    # print(noun_prompt)


    nn_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
    nn_words = [word for word, tag in pos_tag if tag in nn_tags]
    noun_prompt = 'Noun: ' + ', '.join([f'"{noun}"' for noun in nn_words])
    # noun_prompt = "Noun: Battery life, screen"
    # print(noun_prompt)


    # 拼接这些提示
    # full_prompt = f"Sentence: {sentence}\n{noun_prompt}\n{pos_prompt}\n{dep_prompt}\n{Note}\n{task_prompt}"
    full_prompt = f"Sentence: {sentence}\n{task_prompt}\n{noun_prompt}\n{pos_prompt}\n{dep_prompt}\n{Note}"

    return full_prompt




########################  Test  ##############################

# sentence = "Battery life is good but screen is faulty"
# full_prompt = Prompt_Construct(sentence)
# print(full_prompt)



# from SAGF.DateProcess.DataProcessV2 import Dataset_Process
# file_path = r"D:\Project\SAGF\triplet_datav2\14lap\train_triplets.txt"
# datasets = Dataset_Process(file_path)
# for sentence, triplet_labels, Generative_label in datasets:
#     print(sentence)
#     print(triplet_labels)
#     print(Generative_label)
#
#     full_prompt = Prompt_Construct(sentence)
#     print(full_prompt)


##################### Output Style ###########################
# Sentence: Battery life is good but screen is faulty
# Extract all (aspect, opinion, sentiment) triplets from the Sentence, with the output format aspect: A opinion: B sentiment: C SSEP:, where SSEP: use to separate multiple triplets.
#  (For example: aspect: food opinion: good sentiment: positive SSEP: aspect: drink opinion: bad sentiment: negative)
# Noun: "Battery", "life", "screen"
# Part-of-speech:
#  "Battery"/Noun,
#  "life"/Noun,
#  "is"/Verb,
#  "good"/Adjective,
#  "but"/Coordinating conjunction,
#  "screen"/Noun,
#  "is"/Verb,
#  "faulty"/Adjective.
# Dependency relations:
#  "Battery"/noun compound modifier/"life",
#  "life"/nominal subject/"good",
#  "is"/copula/"good",
#  "but"/coordination/"good",
#  "screen"/nominal subject/"faulty",
#  "is"/copula/"faulty",
#  "faulty"/conjunct/"good".
# Note: Aspect terms are usually Noun, and opinion terms are usually Adjective. The dependency relation between aspect and opinion terms is often nominal subject.
