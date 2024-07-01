import numpy as np
import torch
from typing import Dict, List, Optional, Union
import json

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
        with open(data_path) as f:
            examples = json.load(f)
        for ex in examples:
            sentence = ex['comment'].strip().split()
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
        with open(data_path) as f:
            examples = json.load(f)
        for ex in examples:
            sentence = ex['comment'].strip().split()
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
    test_path = './data/kold_v1_split_test.json'
    with open(test_path) as f:
        examples = json.load(f)
    print(examples[0]['guid'])
    print(examples[0]['comment'])
    print(examples[0]['OFF'])

    guid, text_a, label = examples[0]
    print(guid)
    print(text_a)
    print(label)