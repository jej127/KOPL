import random
import numpy as np
import hgtk
import os

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
    emb = [0.0 for _ in range(DIM)]
    origin_repre.append(emb)
    origin_words.append('<unk>')
    all_embs['<unk>'] = emb

    print('loaded! Word num = {a}'.format(a=len(origin_words)))
    return {'origin_word': origin_words, 'origin_repre':origin_repre}, all_embs

if __name__ == '__main__':
    data_path = '/mnt/oov/fasttext_jm.vec'
    dim=300
    dataset, emb = load_dataset(path=data_path, DIM=dim)
    origin_words, origin_repre = dataset['origin_word'], dataset['origin_repre']
    print(origin_words[:10])

    vocab_path = './words'
    if not os.path.exists(vocab_path): os.makedirs(vocab_path)
    with open(os.path.join(vocab_path, 'words.txt'), 'w', encoding='utf8')as f:
        f.write('\n'.join(origin_words))