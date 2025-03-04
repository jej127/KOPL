from registry import register
from functools import partial
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from utils import load_dataset, TextData, repre_word, repre_ipas, repre_word_bert
from attacks import get_random_attack, get_random_attack_ipa
registry = {}
register = partial(register, registry=registry)


@register('simple')
class SimpleLoader():
    def __init__(self, args, TOKENIZER):
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.dim = args.emb_dim
        self.input_type = args.input_type
        self.lowercase = args.lowercase
        self.tokenizer = TOKENIZER

    def collate_fn(self, batch_data, pad=0):
        batch_words, batch_oririn_repre = list(zip(*batch_data))
        aug_words, aug_repre, aug_ids = list(), list(), list()
        for index in range(len(batch_words)):
            aug_word = batch_words[index]
            repre, repre_ids = repre_word(aug_word, self.tokenizer, rtype=self.input_type)
            aug_words.append(aug_word)
            aug_repre.append(repre)
            aug_ids.append(repre_ids)

        batch_words = list(batch_words) + aug_words
        batch_oririn_repre = torch.FloatTensor(batch_oririn_repre)

        max_len = max([len(seq) for seq in aug_ids])
        batch_aug_repre_ids = [char + [pad] * (max_len - len(char)) for char in aug_ids]
        batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)
        mask = torch.ne(batch_aug_repre_ids, pad).unsqueeze(2)
        return batch_words, batch_oririn_repre, batch_aug_repre_ids, mask

    def __call__(self, data_path, neg_sample_path=''):
        dataset, _ = load_dataset(path=data_path, DIM=self.dim, lower=self.lowercase)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // 2, shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)
        return train_iterator

@register('aug_ipa')
class SimpleLoader_IPA():
    def __init__(self, args, TOKENIZER, word_to_ipa=None):
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.dim = args.emb_dim
        self.tokenizer = TOKENIZER
        self.word_to_ipa = word_to_ipa
        self.probs = args.probs
        self.probs_ipa = args.probs_ipa
        self.input_type = args.input_type
        self.use_ipa = args.use_ipa
        if self.use_ipa: 
            ipa_list = ['tɕ', 'u','ju', 'o', 'ɛ', 'ɑ','jo','p*', 'h', 'l', 'ɯ', 'kʰ','i','wi','ɡ','wʌ','ɰi', 'n', 'b','tɕʰ', 
                         'd', 'k','t*','pʰ', 't','jɛ','s*','k*','ja','tʰ','wɛ','tɕ*','ŋ','jʌ','p', 'ʌ', 's','wa','dʑ','m']
            self.ipa_to_id = {'[PAD]':0, '[UNK]':1, '[CLS]':2, '[SEP]':3}
            self.ipa_to_id.update({s:(idx + len(self.ipa_to_id)) for idx,s in enumerate(ipa_list)})
        
    def collate_fn(self, batch_data, pad=0):
        batch_words, batch_oririn_repre = list(zip(*batch_data))
        aug_words, aug_repre, aug_ids = list(), list(), list()
        aug_ipas, aug_repre_ipa, aug_ids_ipa = list(), list(), list()
        batch_ipas = list()
        for index in range(len(batch_words)):
            aug_word = get_random_attack(batch_words[index], probs=self.probs)
            repre, repre_ids = repre_word(aug_word,  self.tokenizer, self.input_type)
            if self.use_ipa:
                ipa = self.word_to_ipa[batch_words[index]]
                aug_ipa = get_random_attack_ipa(ipa, probs=self.probs_ipa)
                repre_ipa, repre_ids_ipa = repre_ipas(aug_ipa, self.ipa_to_id) ###
                batch_ipas.append(ipa)
                aug_ipas.append(aug_ipa)
                aug_repre_ipa.append(repre_ipa)
                aug_ids_ipa.append(repre_ids_ipa)
            aug_words.append(aug_word)
            aug_repre.append(repre)
            aug_ids.append(repre_ids)

        batch_words_pooled = list(batch_words) + aug_words
        batch_oririn_repre = torch.FloatTensor(batch_oririn_repre)

        max_len = max([len(seq) for seq in aug_ids])
        batch_aug_repre_ids = [char + [pad] * (max_len - len(char)) for char in aug_ids]
        batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)
        mask = torch.ne(batch_aug_repre_ids, pad).unsqueeze(2)

        if self.use_ipa:
            batch_ipa_pooled = [''.join(i) for i in batch_ipas + aug_ipas]
            max_len_ipa = max([len(seq) for seq in aug_ids_ipa])
            batch_aug_repre_ids_ipa = [char + [pad] * (max_len_ipa - len(char)) for char in aug_ids_ipa]
            batch_aug_repre_ids_ipa = torch.LongTensor(batch_aug_repre_ids_ipa)
            mask_ipa = torch.ne(batch_aug_repre_ids_ipa, pad).unsqueeze(2)
        else:
            batch_ipa_pooled, batch_aug_repre_ids_ipa, mask_ipa = None, None, None
        return batch_words_pooled, batch_ipa_pooled, batch_oririn_repre, batch_aug_repre_ids, batch_aug_repre_ids_ipa, mask, mask_ipa

    def __call__(self, data_path):
        dataset, _ = load_dataset(path=data_path, DIM=self.dim)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // 2, shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)
        return train_iterator

@register('aug_ipa_bert')
class SimpleLoader_IPA_BERT():
    def __init__(self, args, tokenizer, vocab, word_to_ipa=None):
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.dim = args.emb_dim
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.word_to_ipa = word_to_ipa
        self.probs = args.probs
        self.probs_ipa = args.probs_ipa
        self.input_type = args.input_type
        self.use_ipa = args.use_ipa
        if self.use_ipa: 
            ipa_list = ['tɕ', 'u','ju', 'o', 'ɛ', 'ɑ','jo','p*', 'h', 'l', 'ɯ', 'kʰ','i','wi','ɡ','wʌ','ɰi', 'n', 'b','tɕʰ', 
                         'd', 'k','t*','pʰ', 't','jɛ','s*','k*','ja','tʰ','wɛ','tɕ*','ŋ','jʌ','p', 'ʌ', 's','wa','dʑ','m']
            self.ipa_to_id = {'[PAD]':0, '[UNK]':1, '[CLS]':2, '[SEP]':3}
            self.ipa_to_id.update({s:(idx + len(self.ipa_to_id)) for idx,s in enumerate(ipa_list)})
        
    def collate_fn(self, batch_data, pad=0):
        batch_words, batch_oririn_repre = list(zip(*batch_data))
        aug_words, aug_repre, aug_ids = list(), list(), list()
        aug_ipas, aug_repre_ipa, aug_ids_ipa = list(), list(), list()
        batch_ipas = list()
        for index in range(len(batch_words)):
            aug_word = get_random_attack(batch_words[index], probs=self.probs)
            repre, repre_ids = repre_word_bert(aug_word, self.tokenizer, self.vocab)
            if self.use_ipa:
                ipa = self.word_to_ipa[batch_words[index]]
                aug_ipa = get_random_attack_ipa(ipa, probs=self.probs_ipa)
                repre_ipa, repre_ids_ipa = repre_ipas(aug_ipa, self.ipa_to_id) ###
                batch_ipas.append(ipa)
                aug_ipas.append(aug_ipa)
                aug_repre_ipa.append(repre_ipa)
                aug_ids_ipa.append(repre_ids_ipa)
            aug_words.append(aug_word)
            aug_repre.append(repre)
            aug_ids.append(repre_ids)

        batch_words_pooled = list(batch_words) + aug_words
        batch_oririn_repre = torch.FloatTensor(batch_oririn_repre)

        max_len = max([len(seq) for seq in aug_ids])
        batch_aug_repre_ids = [char + [pad] * (max_len - len(char)) for char in aug_ids]
        batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)
        mask = torch.ne(batch_aug_repre_ids, pad).unsqueeze(2)

        if self.use_ipa:
            batch_ipa_pooled = [''.join(i) for i in batch_ipas + aug_ipas]
            max_len_ipa = max([len(seq) for seq in aug_ids_ipa])
            batch_aug_repre_ids_ipa = [char + [pad] * (max_len_ipa - len(char)) for char in aug_ids_ipa]
            batch_aug_repre_ids_ipa = torch.LongTensor(batch_aug_repre_ids_ipa)
            mask_ipa = torch.ne(batch_aug_repre_ids_ipa, pad).unsqueeze(2)
        else:
            batch_ipa_pooled, batch_aug_repre_ids_ipa, mask_ipa = None, None, None
        return batch_words_pooled, batch_ipa_pooled, batch_oririn_repre, batch_aug_repre_ids, batch_aug_repre_ids_ipa, mask, mask_ipa

    def __call__(self, data_path):
        dataset, _ = load_dataset(path=data_path, DIM=self.dim, lower=False)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // 2, shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)
        return train_iterator