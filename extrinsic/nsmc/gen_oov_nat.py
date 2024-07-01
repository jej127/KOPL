import numpy as np
import csv
from attacks import select_attack
import random

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
        #if cnt==64: break
        #print(word)

    # add <unk> token
    # emb = [0.0 for _ in range(DIM)]
    # origin_repre.append(emb)
    # origin_words.append('<unk>')
    # all_embs['<unk>'] = emb

    print('loaded! Word num = {a}'.format(a=len(origin_words)))
    return {'origin_word': origin_words, 'origin_repre':origin_repre}, all_embs


def main():
    random.seed(123)
    np.random.seed(123)

    csv_path = './data/nsmc_test_cleaned.csv'
    t = 0
    data = []
    with open(csv_path) as f:
        tr = csv.reader(f, delimiter=',')
        for row in tr:
            if t==0: 
                t += 1
                continue
            id_num, sentence, label = row[0], row[1], row[2]
            data.append((id_num, sentence, label))

    data_path='/mnt/oov/fasttext_jm.vec'
    dim=300
    lowercase=False
    dataset, emb = load_dataset(path=data_path, DIM=dim, lower=lowercase)
            
    for r in [0.5,0.7]:
        with open(f"./data/nsmc_test_natural_{int(r*100)}.csv", "w", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'doc', 'label'])
            for ex in data:
                id_num, sentence, label = ex
                sentence_split = sentence.strip().split()
                if len(sentence_split) == 0 : continue
                oov_count = 0
                all_count = len(sentence_split)
                for word in sentence_split:
                    if word not in emb:
                        oov_count += 1
                oov_rate = oov_count / all_count
                if oov_rate > r:
                    writer.writerow([id_num, sentence, label])

    # h = [0,0,0,0,0]
    # for ex in data:
    #     sentence = ex[1].strip().split()
    #     if len(sentence) == 0 : continue
    #     oov_count = 0
    #     all_count = len(sentence)
    #     for word in sentence:
    #         if word not in emb:
    #             oov_count += 1
    #     r = oov_count / all_count
    #     if r <= 0.2:
    #         h[0] += 1
    #     elif r <= 0.4:
    #         h[1] += 1
    #     elif r <= 0.6:
    #         h[2] += 1
    #     elif r <= 0.8:
    #         h[3] += 1
    #     else:
    #         h[4] += 1
    # print(h)        

if __name__ == '__main__':
    main()