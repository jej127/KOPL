import numpy as np
from attacks import select_attack
import random
from utils import create_examples_from_original_data

def filter(word):
    min_len = 1
    if len(word) < min_len:return True
    return False

def load_dataset(path, DIM=300, lower=True):
    origin_words, origin_repre = list(), list()
    all_embs = dict()
    cnt = 0
    for line in open(path, encoding='utf8'):
        cnt += 1
        row = line.strip().split(' ')
        if len(row) != DIM + 1:continue
        word = row[0]
        if lower:
            word = str.lower(word)
        if filter(word): continue
        emb = [float(e) for e in row[1:]]
        origin_repre.append(emb)
        origin_words.append(word)
        all_embs[word] = emb

    print('loaded! Word num = {a}'.format(a=len(origin_words)))
    return {'origin_word': origin_words, 'origin_repre':origin_repre}, all_embs

def main():
    test_path = "data/klue-ner-v1.1_dev.tsv"
    examples, _ = create_examples_from_original_data(test_path)

    data_path='/mnt/oov/fasttext_jm.vec'
    dim=300
    lowercase=False
    dataset, emb = load_dataset(path=data_path, DIM=dim, lower=lowercase)

    with open(f"./data/test_split.txt", "w") as file:
        for ex in examples:
            for word, label in zip(ex.text_a, ex.label):
                file.write(word + " " + label + "\n")
            file.write("\n")

    for r in [0.3,0.5,0.7]:
        texts_oov = []
        examples_oov = []
        for ex in examples:
            text_oov = ex.text_a

            if len(text_oov) == 0 : continue
            oov_count = 0
            all_count = len(text_oov)
            for word in text_oov:
                if word not in emb:
                    oov_count += 1
            oov_rate = oov_count / all_count
            if oov_rate > r:
                texts_oov.append(text_oov)
                examples_oov.append(ex)

        with open(f"./data/test_split_natural_{int(r*100)}.txt", "w") as file:
            for text, ex in zip(texts_oov, examples_oov):
                for word, label in zip(text, ex.label):
                    file.write(word + " " + label + "\n")
                file.write("\n")

if __name__ == '__main__':
    main()