import random
import numpy as np
import csv
import pandas as pd
import re
import math
import os

GA_CODE = 44032 # The unicode representation of the Korean syllabic orthography starts with GA_CODE
G_CODE = 12593 # The unicode representation of the Korean phonetic (jamo) orthography starts with G_CODE
ONSET = 588
CODA = 28

# ONSET LIST. 00 -- 18 (자음)
ONSET_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# VOWEL LIST. 00 -- 20 (모음)
VOWEL_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
               'ㅣ']

# CODA LIST. 00 -- 27 + 1 (1 for open syllable)
CODA_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
              'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

highV_diphthongs = ["ㅑ","ㅕ","ㅖ","ㅛ","ㅠ","ㅣ"]

def read_tsv_resource(file_name):
    base_path = "./ipa_convert/csv"
    path = os.path.join(base_path, file_name)
    return pd.read_csv(path, delimiter='\t', names=['CN','KO'], header=None)

def intToHan(han_int):
    sep_loc = han_int.index('+')
    han_int = han_int[sep_loc+1:].lower()
    han_int = eval("0x" + han_int)
    return chr(han_int)

def hanja_to_hangul(data, syllable):
    try:
        idx = data['CN'].tolist().index(syllable)
        res = data.iloc[idx,1]
        return res
    except ValueError:
        return syllable

def convertHangulStringToJamos(word):
    split_word = list(word)
    output = []
    
    for letter in split_word:
        syllable = ""
        if re.match("[가-힣]", letter):  # run this only for a Korean character
            chr_code = ord(letter)  # returns the Unicode code point of a character
            chr_code = chr_code - GA_CODE
            
            if chr_code < 0:
                syllable = letter
            
            onset = math.floor(chr_code / ONSET)
            vowel = math.floor((chr_code - (ONSET * onset)) / CODA)
            coda = math.floor(chr_code - (ONSET * onset) - (CODA * vowel))
            syllable = ONSET_LIST[onset] + VOWEL_LIST[vowel] + CODA_LIST[coda] # (자음, 모음, 받침)으로 분해하기
            
        else:
            syllable = letter
        
        output.append(syllable)
        
    return output

def assembleJamo(syllable):
    # only accepts one syllable!
    if len(syllable) > 1:
        onset = ONSET_LIST.index(syllable[0])
        vowel = VOWEL_LIST.index(syllable[1])
        coda = 0
        
        if len(syllable) == 3:
            coda = CODA_LIST.index(syllable[2])
        
        syllable = chr((((onset * 21) + vowel) * 28) + coda + GA_CODE)
    
    return syllable

def hanja_cleaner(data, syllables, hanja_loc):
    syllables = list(syllables)
    if len(syllables) > 1:
        for j in range(len(syllables)-1,0,-1):
            if hanja_loc[j]:
                if syllables[j]=="實" and (syllables[j-1]=="不" or syllables[j-1]=="不"):
                    syllables[j-1] = '부'
                    hanja_loc[j-1] = False
                    syllables[j] = '실'
                    hanja_loc[j] = False
                else:
                    syllables[j] = hanja_to_hangul(data, syllables[j])
                    new_onset = convertHangulStringToJamos(syllables[j])[0][0]
                    if new_onset in ['ㄷ','ㅈ'] and (syllables[j-1]=="不" or syllables[j-1]=="不"):
                        syllables[j-1] = '부'
                        hanja_loc[j-1] = False
    if hanja_loc[0]:
        converted_hangul = hanja_to_hangul(data, syllables[0])
        if converted_hangul == syllables[0]: return 'A' # 한글 변환 안 되면 그냥 ''를 ipa로 뽑도록
        syllables[0] = hanja_to_hangul(data, syllables[0])
        first_syllable = list(convertHangulStringToJamos(syllables[0])[0])
        if first_syllable[0] == "ㄹ": first_syllable[0] = "ㄴ"
        if first_syllable[0] == "ㄴ" and first_syllable[1] in highV_diphthongs: first_syllable[0] = "ㅇ"
        syllables[0] = assembleJamo(first_syllable)

    return ''.join(syllables)

if __name__ == '__main__':
    data = read_tsv_resource("hanja.tsv")

    #syllables = '육甲㒚오㔡'
    #syllables = '今을子'
    #syllables = '不實'
    syllables = '不動산'
    hanja_loc = [bool(re.search(r"[\u4e00-\u9fff|\u3400-\u4DBF|\U00020000-\U0002A6DF|一-鿕|㐀-䶵|豈-龎]+", s, flags=re.UNICODE)) for s in syllables] 
    # find chinese characters
    data['CN'] = data['CN'].map(intToHan)
    #print(data[-20:])
    #print(data['CN'].tolist()[-20:])

    print(hanja_to_hangul(data, '相'))

    print(hanja_cleaner(data, syllables, hanja_loc))

