
import pandas as pd
import re
import math
import os
from hanja_tools import intToHan, hanja_cleaner
from ipa_converter import applyRulesToHangul
from tqdm import tqdm

ipa_list = ['tɕ', 'u','ju', 'o', 'ɛ', 'ɑ','jo','p*', 'h', 'l', 'ɯ', 'kʰ','i','wi','ɡ','wʌ','ɰi', 'n', 'b','tɕʰ', 
             'd', 'k','t*','pʰ', 't','jɛ','s*','k*','ja','tʰ','wɛ','tɕ*','ŋ','jʌ','p', 'ʌ', 's','wa','dʑ','m','',' ']

if __name__ == '__main__':
    # Create IPAs for fine-tuning downstream models
    task = 'kold' # Choose one of 'kold','klue_tc','nsmc','klue_dp','klue_ner'
    vocab_path = f'./extrinsic/{task}/data/words.txt'

    words = []
    with open(vocab_path, "r") as f:
        for line in f:
            words.append(line.strip())

    with open(f'./words/ipas_{task}.txt', 'w') as f:
        for i, w in enumerate(tqdm(words)):
            try:
                output, output_sparse = applyRulesToHangul(w, rules="pastcnv")
            except ValueError:
                output, output_sparse = '',[]
            try:
                assert set(output_sparse).issubset(set(ipa_list))
            except:
                print(w, output_sparse)
                exit()
            output_sparse = ' '.join(output_sparse)
            f.write(str(i) + '\t' + w + '\t' + output + '\t' + output_sparse + '\n')
