import numpy as np
import random
import json

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
    random.seed(123)
    np.random.seed(123)

    test_path = './data/kold_v1_split_test.json'
    with open(test_path) as f:
        examples = json.load(f)

    data_path='/home/user7/Main/LOVE_kor/embeddings/fasttext_jm.vec'
    dim=300
    lowercase=False
    dataset, emb = load_dataset(path=data_path, DIM=dim, lower=lowercase)

    for r in [0.3,0.5,0.7]:
        data = []
        for ex in examples:
            sentence_split = ex['comment'].strip().split()
            if len(sentence_split) == 0 : continue
            oov_count = 0
            all_count = len(sentence_split)
            for word in sentence_split:
                if word not in emb:
                    oov_count += 1
            oov_rate = oov_count / all_count
            if oov_rate > r:
                data.append({"guid": ex['guid'], "comment": ex['comment'], "OFF": ex['OFF']})

        with open(f"./data/kold_v1_split_test_natural_{int(r*100)}.json", "w") as f:
            json.dump(data, f, indent="\t", ensure_ascii=False)


if __name__ == '__main__':
    main()