import numpy as np
import torch
import collections
from torch.utils.data import Dataset
from tqdm import tqdm
import hgtk
from decompose_dict import *


chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', '']
chosung_unicode = [u"\u1100", u"\u1101", u"\u1102", u"\u1103", u"\u1104", u"\u1105", u"\u1106", u"\u1107",
                   u"\u1108", u"\u1109", u"\u110A", u"\u110B", u"\u110C", u"\u110D", u"\u110E", u"\u110F",
                   u"\u1110", u"\u1111", u"\u1112", ""]
jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ', '']
jungsung_unicode = [u"\u1161", u"\u1162", u"\u1163", u"\u1164", u"\u1165", u"\u1166", u"\u1167", u"\u1168",
                    u"\u1169", u"\u116A", u"\u116B", u"\u116C", u"\u116D", u"\u116E", u"\u116F", u"\u1170",
                    u"\u1171", u"\u1172", u"\u1173", u"\u1174", u"\u1175", ""]
jongsung_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ' , 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 
                 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', '']
jongsung_unicode = [u"\u11A8", u"\u11A9", u"\u11AA", u"\u11AB", u"\u11AC", u"\u11AD", u"\u11AE", u"\u11AF",
                    u"\u11B0", u"\u11B1", u"\u11B2", u"\u11B3", u"\u11B4", u"\u11B5", u"\u11B6", u"\u11B7",
                    u"\u11B8", u"\u11B9", u"\u11BA", u"\u11BB", u"\u11BC", u"\u11BD", u"\u11BE", u"\u11BF",
                    u"\u11C0", u"\u11C1", u"\u11C2", ""]

ja_to_cho = {k:v for k,v in zip(chosung_list,chosung_unicode)}
mo_to_jung = {k:v for k,v in zip(jungsung_list,jungsung_unicode)}
ja_to_jong = {k:v for k,v in zip(jongsung_list,jongsung_unicode)}


ipa_list = ['tɕ', 'u','ju', 'o', 'ɛ', 'ɑ','jo','p*', 'h', 'l', 'ɯ', 'kʰ','i','wi','ɡ','wʌ','ɰi', 'n', 'b','tɕʰ', 
             'd', 'k','t*','pʰ', 't','jɛ','s*','k*','ja','tʰ','wɛ','tɕ*','ŋ','jʌ','p', 'ʌ', 's','wa','dʑ','m']
ipa_to_id = {'[PAD]':0, '[UNK]':1, '[CLS]':2, '[SEP]':3}
ipa_to_id.update({s:(idx + len(ipa_to_id)) for idx,s in enumerate(ipa_list)})

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
        #if cnt==256: break
        #print(word)

    # add <unk> token
    if 'bert' not in path:
        emb = [0.0 for _ in range(DIM)]
        origin_repre.append(emb)
        origin_words.append('<unk>')
        all_embs['<unk>'] = emb

    print('loaded! Word num = {a}'.format(a=len(origin_words)))
    return {'origin_word': origin_words, 'origin_repre':origin_repre}, all_embs


def load_predict_dataset(path):
    origin_words, origin_repre = list(), list()
    for line in open(path, encoding='utf8'):
        word = line.strip()
        origin_repre.append(word)
        origin_words.append(word)
    print('loaded! Word num = {a}'.format(a=len(origin_words)))
    return {'origin_word': origin_words, 'origin_repre':origin_repre}


class TextData(Dataset):
    def __init__(self, data):
        self.origin_word = data['origin_word']
        self.origin_repre = data['origin_repre']
        #self.repre_ids = data['repre_ids']

    def __len__(self):
        return len(self.origin_word)

    def __getitem__(self, idx):
        return self.origin_word[idx], self.origin_repre[idx]


def collate_fn(batch_data, TOKENIZER, pad=0):
    batch_words, batch_oririn_repre = list(zip(*batch_data))

    aug_words, aug_repre, aug_ids = list(), list(), list()
    for index in range(len(batch_words)):
        #aug_word = get_random_attack(batch_words[index])
        aug_word = batch_words[index]
        repre, repre_ids = repre_word(aug_word, TOKENIZER, id_mapping=None)
        aug_words.append(aug_word)
        aug_repre.append(repre)
        aug_ids.append(repre_ids)

    batch_words = list(batch_words) + aug_words
    batch_oririn_repre = torch.FloatTensor(batch_oririn_repre)

    x_lens = [len(x) for x in aug_ids]
    max_len = max([len(seq) for seq in aug_ids])
    batch_aug_repre_ids = [char + [pad]*(max_len - len(char)) for char in aug_ids]
    batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)

    return batch_words, batch_oririn_repre, batch_aug_repre_ids, x_lens

def collate_fn_predict(batch_data, TOKENIZER, rtype='mixed', pad=0):
    batch_words, batch_oririn_repre = list(zip(*batch_data))

    batch_repre_ids = list()
    for word in batch_words:
        repre, repre_id = repre_word(word, TOKENIZER, id_mapping=None, rtype=rtype)
        batch_repre_ids.append(repre_id)

    max_len = max([len(seq) for seq in batch_repre_ids])
    batch_repre_ids = [char + [pad]*(max_len - len(char)) for char in batch_repre_ids]
    batch_repre_ids = torch.LongTensor(batch_repre_ids)
    mask = torch.ne(batch_repre_ids, pad).unsqueeze(2)
    return batch_words, batch_oririn_repre, batch_repre_ids, mask

def collate_fn_predict_(batch_data, args, TOKENIZER, word_to_ipa, pad=0):
    batch_words, batch_oririn_repre = list(zip(*batch_data))

    batch_repre_ids, batch_repre_ids_ipa = list(), list()
    for word in batch_words:
        _, repre_id = repre_word(word, TOKENIZER, args.input_type)
        if args.use_ipa:
            ipa = word_to_ipa[word]
            _, repre_ids_ipa = repre_ipas(ipa, ipa_to_id)
            batch_repre_ids_ipa.append(repre_ids_ipa)

        batch_repre_ids.append(repre_id)

    max_len = max([len(seq) for seq in batch_repre_ids])
    batch_repre_ids = [char + [pad]*(max_len - len(char)) for char in batch_repre_ids]
    batch_repre_ids = torch.LongTensor(batch_repre_ids)
    mask = torch.ne(batch_repre_ids, pad).unsqueeze(2)

    if args.use_ipa:
        max_len_ipa = max([len(seq) for seq in batch_repre_ids_ipa])
        batch_repre_ids_ipa = [char + [pad] * (max_len_ipa - len(char)) for char in batch_repre_ids_ipa]
        batch_repre_ids_ipa = torch.LongTensor(batch_repre_ids_ipa)
        mask_ipa = torch.ne(batch_repre_ids_ipa, pad).unsqueeze(2)
    else:
        batch_repre_ids_ipa, mask_ipa = None, None
    return batch_words, batch_oririn_repre, batch_repre_ids, batch_repre_ids_ipa, mask, mask_ipa


def collate_fn_predict_bert(batch_data, args, tokenizer, vocab, word_to_ipa, pad=0):
    batch_words, batch_oririn_repre = list(zip(*batch_data))

    batch_repre_ids, batch_repre_ids_ipa = list(), list()
    for word in batch_words:
        _, repre_id = repre_word_bert(word, tokenizer, vocab)
        if args.use_ipa:
            ipa = word_to_ipa[word]
            _, repre_ids_ipa = repre_ipas(ipa, ipa_to_id)
            batch_repre_ids_ipa.append(repre_ids_ipa)

        batch_repre_ids.append(repre_id)

    max_len = max([len(seq) for seq in batch_repre_ids])
    batch_repre_ids = [char + [pad]*(max_len - len(char)) for char in batch_repre_ids]
    batch_repre_ids = torch.LongTensor(batch_repre_ids)
    mask = torch.ne(batch_repre_ids, pad).unsqueeze(2)

    if args.use_ipa:
        max_len_ipa = max([len(seq) for seq in batch_repre_ids_ipa])
        batch_repre_ids_ipa = [char + [pad] * (max_len_ipa - len(char)) for char in batch_repre_ids_ipa]
        batch_repre_ids_ipa = torch.LongTensor(batch_repre_ids_ipa)
        mask_ipa = torch.ne(batch_repre_ids_ipa, pad).unsqueeze(2)
    else:
        batch_repre_ids_ipa, mask_ipa = None, None
    return batch_words, batch_oririn_repre, batch_repre_ids, batch_repre_ids_ipa, mask, mask_ipa


def collate_fn_predict_bert_(batch_data, args, word_dict, word_to_ipa, pad=0):
    batch_words, batch_oririn_repre = list(zip(*batch_data))

    batch_repre_ids, batch_repre_ids_ipa = list(), list()
    for word in batch_words:
        _, repre_id = repre_word_bert(word, word_dict)
        if args.use_ipa:
            ipa = word_to_ipa[word]
            _, repre_ids_ipa = repre_ipas(ipa, ipa_to_id)
            batch_repre_ids_ipa.append(repre_ids_ipa)

        batch_repre_ids.append(repre_id)

    max_len = max([len(seq) for seq in batch_repre_ids])
    batch_repre_ids = [char + [pad]*(max_len - len(char)) for char in batch_repre_ids]
    batch_repre_ids = torch.LongTensor(batch_repre_ids)
    mask = torch.ne(batch_repre_ids, pad).unsqueeze(2)

    if args.use_ipa:
        max_len_ipa = max([len(seq) for seq in batch_repre_ids_ipa])
        batch_repre_ids_ipa = [char + [pad] * (max_len_ipa - len(char)) for char in batch_repre_ids_ipa]
        batch_repre_ids_ipa = torch.LongTensor(batch_repre_ids_ipa)
        mask_ipa = torch.ne(batch_repre_ids_ipa, pad).unsqueeze(2)
    else:
        batch_repre_ids_ipa, mask_ipa = None, None
    return batch_words, batch_oririn_repre, batch_repre_ids, batch_repre_ids_ipa, mask, mask_ipa

def filter(word):
    min_len = 1
    if len(word) < min_len:return True
    return False


def tokenize_and_getid(word, tokenizer):
    tokens = tokenizer.tokenize(tokenizer.convert_to_unicode(word))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids


def hash_sub_word(total, bucket):
    bucket -= 1
    id_mapping = collections.OrderedDict()
    for id in range(total):
        hashing = ((id % bucket) ^ 2) + 1
        #print(id, hashing)
        id_mapping[id] = hashing
    id_mapping[0] = 0
    id_mapping[100] = bucket + 2
    id_mapping[101] = bucket + 3
    id_mapping[102] = bucket + 4
    id_mapping[103] = bucket + 5
    id_mapping[104] = bucket + 6
    return id_mapping

def repre_word(word, tokenizer, rtype='mixed'):
    start = '[CLS]'
    sub = '[SUB]'
    end = '[SEP]'
    char_seq = list(word)
    tokens = tokenizer.tokenize(word)

    if 'bts' in rtype:
        bts_seq = []
        for i in char_seq:
            try:
                temp = []
                cho, joong, jong = hgtk.letter.decompose(i)
                cho = decompose_dict[cho]
                for unit in cho:
                    temp.append(unit)
                joong = decompose_dict[joong]
                for unit in joong:
                    temp.append(unit)
                jong = decompose_dict[jong]
                if jong == '':
                    temp.append("*")
                else:
                    for unit in jong:
                        temp.append(unit)
                bts_seq.extend(temp)
            except:
                bts_seq.extend(i)

    else:
        jamo_seq = []
        for i in char_seq:
            try:
                cho, joong, jong = hgtk.letter.decompose(i)
                jamo_seq.extend(cho)
                jamo_seq.extend(joong)
                if len(cho+joong+jong) == 2:
                    jamo_seq.extend("*")
                else:
                    jamo_seq.extend(jong)
            except:
                jamo_seq.extend(i)

    if rtype == 'mixed':
        repre = [start] + jamo_seq + [sub] + tokens + [end]
    elif rtype  == 'jamo':
        repre = [start] + jamo_seq + [end]
    elif 'bts' in rtype:
        repre = [start] + bts_seq + [sub] + tokens + [end]
    repre_ids = tokenizer.convert_tokens_to_ids(repre)
    return repre, repre_ids

def repre_ipas(ipa, ipa_to_id):
    start,end = '[CLS]','[SEP]'
    repre_ipa = [start] + ipa + [end]
    repre_ids_ipa = [ipa_to_id[s] for s in repre_ipa]
    return repre_ipa, repre_ids_ipa


def repre_word_bert_(word, word_dict):
    start = '[CLS]'
    sub = '[SUB]'
    end = '[SEP]'
    char_seq = list(word)
    tokens = [word]

    jamo_seq = []
    for i in char_seq:
        try:
            cho, joong, jong = hgtk.letter.decompose(i)
            jamo_seq.extend(cho)
            jamo_seq.extend(joong)
            if len(cho+joong+jong) == 2:
                jamo_seq.extend("*")
            else:
                jamo_seq.extend(jong)
        except:
            jamo_seq.extend(i)

    repre = [start] + jamo_seq + [sub] + tokens + [end]
    repre_ids = [word_dict[w] for w in repre]
    return repre, repre_ids

def repre_word_bert(word, tokenizer, vocab, rtype='mixed'):
    start = '[CLS]'
    sub = '[SUB]'
    end = '[SEP]'
    char_seq = list(word.replace('▁',''))
    tokens = tokenizer.tokenize(word.replace('▁',''))
    jamo_seq = []
    for i in char_seq:
        try:
            cho, joong, jong = hgtk.letter.decompose(i)
            jamo_seq.extend(cho)
            jamo_seq.extend(joong)
            if len(cho+joong+jong) == 2:
                jamo_seq.extend("*")
            else:
                jamo_seq.extend(jong)
        except:
            jamo_seq.extend(i)
    if rtype == 'mixed':
        repre = [start] + jamo_seq + [sub] + tokens + [end]
    elif rtype  == 'jamo':
        repre = [start] + jamo_seq + [end]
    repre_ids = vocab.convert_tokens_to_ids(repre)
    return repre, repre_ids

def add_tokens(tokenizer):
    add_list = ['[SUB]', 'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㅊ', 'ㅍ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 
                'ㅝ', 'ㅞ', 'ㅟ', 'ㅢ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']
    tokenizer.add_tokens(add_list)
    return tokenizer

def load_ipa(ipa_path):
    with open(ipa_path, "r") as file:
        lines = file.readlines()
    ipa_set, word_to_ipa = set(), {}
    for line in lines:
        line = line.strip().split("\t")
        word = line[1]
        if len(line)==4:
            ipa_split = line[3].strip().split()
            ipa_set = ipa_set.union(set(ipa_split))
        else:
            ipa_split = ['[UNK]']
        word_to_ipa[word] = ipa_split
    ipa_set = list(ipa_set)
    return word_to_ipa, ipa_set

if __name__ == '__main__':
    pass