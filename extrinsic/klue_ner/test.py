import numpy as np
import torch
from typing import Dict, List, Optional, Union
from collections import namedtuple
import json
import os
import re
from pathlib import Path

InputExample = namedtuple("TextPairExample", ["guid", "text_a", "label"])

def create_examples_from_original_data(file_path, train=True)  -> List[InputExample]:
    examples = []
    ori_examples = []
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r"\n\t?\n", raw_text)
    cnt = 0
    for doc in raw_docs:
        original_clean_tokens = []  # clean tokens (bert clean func)
        original_clean_labels = []  # clean labels (bert clean func)
        sentence = ""
        for line in doc.split("\n"):
            if line[:2] == "##":
                guid = line.split("\t")[0].replace("##", "")
                continue
            token, tag = line.split("\t")
            sentence += token
            if token == " ":
                continue
            original_clean_tokens.append(token)
            original_clean_labels.append(tag)
        # sentence: "안녕 하세요.."
        # original_clean_labels: [안, 녕, 하, 세, 요, ., .]
        sent_words = sentence.split(" ")
        sent_words = [x for x in sent_words if x]

        assert sum([len(w) for w in sent_words])==len(original_clean_labels)

        modi_labels = []
        char_idx = 0
        for word in sent_words:
            correct_syllable_num = len(word)
            modi_labels.append(original_clean_labels[char_idx])
            char_idx += correct_syllable_num

        cnt += 1
        examples.append(InputExample(guid=guid, text_a=sent_words, label=modi_labels))
        ori_examples.append({"original_sentence": sentence, "original_clean_tokens": original_clean_tokens, 
                             "original_clean_labels": original_clean_labels})
    return examples, ori_examples

def label_transform(label):
    if label == '-':
        return 'O'
    else:
        label_ = label.split('_')
        return label_[1] + '-' + label_[0]

if __name__ == '__main__':
    file_path = "data/klue-ner-v1.1_train.tsv"
    examples, ori_examples = create_examples_from_original_data(file_path)
    for j in range(30):
    #     print(ori_examples[j])
    #     print(len(ori_examples[j]["original_clean_tokens"]))
    #     print(len(ori_examples[j]["original_clean_labels"]))

        # if '말론 브란도' in ori_examples[j]['original_sentence']:
        #     print(ori_examples[j])
        #     print(ori_examples[j]['original_sentence'].split(" "))

        print(examples[j])
        #print(examples[j].text_a)
        #print(ori_examples[j])