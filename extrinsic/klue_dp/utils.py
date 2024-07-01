import numpy as np
import torch
import json
from typing import Dict, List, Optional, Union
from collections import namedtuple

InputExample = namedtuple("TextPairExample", ["guid", "text", "sent_id", "token_id", "token", "pos", "head", "dep"])

dep_label_list = ["NP", "NP_AJT", "VP", "NP_SBJ", "VP_MOD", "NP_OBJ", "AP", "NP_CNJ", "NP_MOD", "VNP", "DP", "VP_AJT",
                  "VNP_MOD", "NP_CMP", "VP_SBJ", "VP_CMP", "VP_OBJ", "VNP_CMP", "AP_MOD", "X_AJT", "VP_CNJ", "VNP_AJT", "IP", "X", "X_SBJ",
                  "VNP_OBJ", "VNP_SBJ", "X_OBJ", "AP_AJT", "L", "X_MOD", "X_CNJ", "VNP_CNJ", "X_CMP", "AP_CMP", "AP_SBJ", "R", "NP_SVJ",]

pos_label_list = ["NNG", "NNP", "NNB", "NP", "NR", "VV", "VA", "VX", "VCP", "VCN", "MMA", "MMD", "MMN", "MAG", "MAJ", "JC", "IC", "JKS",
                  "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "EP", "EF", "EC", "ETN", "ETM", "XPN", "XSN", "XSV", "XSA", "XR", "SF",
                  "SP", "SS", "SE", "SO", "SL", "SH", "SW", "SN", "NA",]

def create_examples(file_path) -> List[InputExample]:
    sent_id = -1
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line == "\n" or line == "\t":
                continue
            if line.startswith("#"):
                parsed = line.strip().split("\t")
                if len(parsed) != 2:  # metadata line about dataset
                    continue
                else:
                    sent_id += 1
                    text = parsed[1].strip()
                    guid = parsed[0].replace("##", "").strip()
            else:
                token_list = [token.replace("\n", "") for token in line.split("\t")] + ["-", "-"]
                examples.append(
                    InputExample(
                        guid=guid,
                        text=text,
                        sent_id=sent_id,
                        token_id=int(token_list[0]),
                        token=token_list[1],
                        pos=token_list[3],
                        head=token_list[4],
                        dep=token_list[5],
                    )
                )
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
        for test_path in test_paths: self.read_vocab(test_path)

    def read_vocab(self, data_path):
        examples = create_examples(data_path)
        for ex in examples:
            word = ex.token.strip()
            if self.number_normalized: 
                word = normalize_word(word)
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
        for test_path in test_paths: self.read_vocab(test_path)

    def read_vocab(self, data_path):
        examples = create_examples(data_path)
        for ex in examples:
            word = ex.token.strip()
            if self.number_normalized: 
                word = normalize_word(word)
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
    pass