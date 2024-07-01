import numpy as np
from attacks import select_attack
import random
from utils import create_examples
import json
import csv
from collections import Counter

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

    data_path='/mnt/oov/fasttext_jm.vec'
    dim=300
    lowercase=False
    dataset, emb = load_dataset(path=data_path, DIM=dim, lower=lowercase)

    test_path = './data/klue-dp-v1.1_dev.tsv'
    examples = create_examples(test_path)
    c = Counter([ex.sent_id for ex in examples])
    for r in [0.3,0.5,0.7]:
        data = []
        start_idx = 0
        for i in range(len(c)):
            token_ids, tokens, poses, heads, deps = [],[],[],[],[]
            word_length = c[i]
            guid, current_sent_id = examples[start_idx].guid, examples[start_idx].sent_id
            for j in range(start_idx, start_idx+word_length):
                word = examples[j].token.strip()
                token_ids.append(str(examples[j].token_id))
                tokens.append(word)
                poses.append(examples[j].pos)
                heads.append(examples[j].head)
                deps.append(examples[j].dep)

            oov_count = 0
            all_count = len(tokens)
            if all_count == 0 : continue
            for word in tokens:
                if word not in emb:
                    oov_count += 1
            oov_rate = oov_count / all_count
            if oov_rate > r:
                data.append({"guid": guid, "text": ' '.join(tokens), "sent_id": current_sent_id, "token_id": token_ids, "token": tokens,
                            "pos": poses, "head": heads, "dep": deps})
            start_idx += word_length

        with open(f'./data/klue-dp-v1.1_dev_natural_{int(r*100)}.tsv', 'w', encoding='utf-8', newline='') as f:
            for d in data:
                tw = csv.writer(f, delimiter='\t')
                tw.writerow(["## " + d["guid"], d["text"]])
                for token_id, token, pos, head, dep in zip(d["token_id"], d["token"], d["pos"], d["head"], d["dep"]):
                    tw.writerow([token_id, token, token, pos, head, dep])
                tw.writerow("")


if __name__ == '__main__':
    main()