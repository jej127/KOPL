import numpy as np
import torch
from typing import Dict, List, Optional, Union
import json
from collections import namedtuple

InputExample = namedtuple("TextPairExample", ["guid", "text_a", "label"])

def create_examples(file_path: str) -> List[InputExample]:
    examples = []
    label_map = {label: i for i, label in enumerate(["정치", "경제", "사회", "생활문화", "세계", "IT과학", "스포츠"])}
    with open(file_path, "r", encoding="utf-8") as f:
        data_lst = json.load(f)

    for data in data_lst:
        guid, title, label = data["guid"], data["title"], data["label"]
        if isinstance(label, str):
            examples.append(InputExample(guid, title, label_map[label]))
        else:
            examples.append(InputExample(guid, title, label))
    return examples

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

class WordVocabulary(object):
    def __init__(self, train_path, dev_path, test_paths, number_normalized):
        self.number_normalized = number_normalized
        self._id_to_word = []
        self._word_to_id = {}
        self.index = 0

        self.read_vocab(train_path)
        self.read_vocab(dev_path)
        for test_path in test_paths:
            self.read_vocab(test_path)

    def read_vocab(self, data_path):
        examples = create_examples(data_path)
        for ex in examples:
            sentence = ex.text_a.strip().split()
            if self.number_normalized: 
                sentence = [normalize_word(word) for word in sentence]
            for word in sentence:
                if word not in self._word_to_id:
                    self._id_to_word.append(word)
                    self._word_to_id[word] = self.index
                    self.index += 1

    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        #return self.unk()

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def items(self):
        return self._word_to_id.items()


class WordVocabulary_count(object):
    def __init__(self, train_path, dev_path, test_paths, number_normalized):
        self.number_normalized = number_normalized
        self._id_to_word = []
        self._word_to_id = {}
        self.index = 0
        self.count = {}

        self.read_vocab(train_path)
        self.read_vocab(dev_path)
        for test_path in test_paths:
            self.read_vocab(test_path)

    def read_vocab(self, data_path):
        examples = create_examples(data_path)
        for ex in examples:
            sentence = ex.text_a.strip().split()
            if self.number_normalized: 
                sentence = [normalize_word(word) for word in sentence]
            for word in sentence:
                if word not in self._word_to_id:
                    self._id_to_word.append(word)
                    self._word_to_id[word] = self.index
                    self.index += 1
                    self.count[word] = 0
                else:
                    self.count[word] += 1

    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        #return self.unk()

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def items(self):
        return self._word_to_id.items()


if __name__ == '__main__':
    data_path = './data/ynat-v1.1_dev.json'
    examples = create_examples(data_path)
    print(examples[0].guid)
    print(examples[0].text_a)
    print(examples[0].label)

    guid, text_a, label = examples[0]
    print(guid)
    print(text_a)
    print(label)