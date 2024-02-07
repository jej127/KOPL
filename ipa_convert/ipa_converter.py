'''
MIT License

Copyright (c) 2022 stannam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# translated R code into Python by DevTae@2023

import pandas as pd
import re
import math
import os
from hanja_tools import intToHan, hanja_cleaner

def read_csv_resource(file_name):
    base_path = "./ipa_convert/csv"
    path = os.path.join(base_path, file_name)
    return pd.read_csv(path)

def read_tsv_resource(file_name):
    base_path = "./ipa_convert/csv"
    path = os.path.join(base_path, file_name)
    return pd.read_csv(path, delimiter='\t', names=['CN','KO'], header=None)

# load csv files
criteria_DoubleCoda = read_csv_resource("double_coda.csv")
roman_ipa = read_csv_resource("ipa.csv")
roman_yale = read_csv_resource("yale.csv")
neutral = read_csv_resource("neutralization.csv") # 중화
criteria_Tensification = read_csv_resource("tensification.csv") # 경음화
criteria_Assimilation = read_csv_resource("assimilation.csv") # 동화
criteria_Aspiration = read_csv_resource("aspiration.csv") # 기음화 (숨소리)
hanja_data = read_tsv_resource("hanja.tsv") # 한자
hanja_data['CN'] = hanja_data['CN'].map(intToHan)

# hangul unicode interface

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

#print(convertHangulStringToJamos("안녕하세요")) # 안녕하세요 -> "ㅇㅏㄴ", "ㄴㅕㅇ", "ㅎㅏ", "ㅅㅔ", "ㅇㅛ"
#print(convertHangulStringToJamos("저는양념치킨을더좋아해요"))
#print(convertHangulStringToJamos("앇녋핛셃욯!"))

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

#print(assembleJamo(["ㄱ", "ㅏ"])) # 가
#print(assembleJamo(["ㄱ", "ㅏ", "ㅇ"])) # 강
#print(assembleJamo(["ㄱ", "ㅏ", "ㄻ"])) # 갊

def sanitize(word):
    if len(word) < 1:  # if empty input, no sanitize
        return word
    
    syllables = list(word)
    while syllables[0] == " ":
        syllables = syllables[1:]
    
    #hanja_loc = [bool(re.search(r"[\u4e00-\u9fff]", s, flags=re.UNICODE)) for s in syllables] # find chinese characters
    hanja_loc = [bool(re.search(r"[\u4e00-\u9fff|\u3400-\u4DBF|\U00020000-\U0002A6DF|一-鿕|㐀-䶵|豈-龎]+", s, flags=re.UNICODE)) for s in syllables]
    if any(hanja_loc):
        #return None # 한자 문자가 발견 시 None 반환
        return hanja_cleaner(hanja_data, syllables, hanja_loc)
    else:
        word = ''.join(syllables)
    
    return word

#print(sanitize("안녕하세오..")) # 안녕하시오 ..
#print(sanitize("  안녕하시오..")) # 안녕하시오..
#print(sanitize("안녕하시오.. 無")) # None

def toJamo(data, removeEmptyOnset=True, sboundary=False):
    # Hangul forms to Jamo
    jamo = str()
    
    syllable = convertHangulStringToJamos(data) # 안녕 -> "ㅇㅏㄴ", "ㄴㅕㅇ"
    for j in range(len(syllable)):
        if len(syllable[j]) > 2: # 받침이 있는 경우
            DC = (criteria_DoubleCoda.loc[criteria_DoubleCoda['double'] == syllable[j][2]])
            if not DC.empty:  # 겹받침을 둘로 나눔 (e.g. "ㄳ" -> "ㄱㅅ")
                syllable[j] = syllable[j][:2] + str(DC['separated'].values[0])
        
        if removeEmptyOnset: # 'syllable'의 j번째 element를 각 자모단위로 분리해서 새로운 list 'phonemic'에 넣습니다.
            if syllable[j][0] == "ㅇ":  # 첫번째 자모(즉, 초성)가 'ㅇ'이면, 그것을 제거합니다.
                jamo += syllable[j][1:] # 'phonemic'을 결합해서 다시 음절단위로 만듭니다. 그러나 초성의 ㅇ은 제거된 상태입니다.
            else:
                jamo += syllable[j]
        else: # 그 결과를 jamo에 저장합니다.
            jamo += syllable[j]
    
    return jamo

#print(toJamo("앇안녕하세요")) # ㅇㅏㄱㅅㅏㄴㄴㅕㅇㅎㅏㅅㅔㅛ
#print(toJamo("가나다라마바사감남담람맘밤삼")) # ㄱㅏㄴㅏㄷㅏㄹㅏㅁㅏㅂㅏㅅㅏㄱㅏㅁㄴㅏㅁㄷㅏㅁㄹㅏㅁㅁㅏㅁㅂㅏㅁㅅㅏㅁ

def CV_mark(input):
    # This function is for identifying a Jamo as either consonant or vowel.
    output = []
    phoneme = list(input)
    for j in range(len(phoneme)):
        if phoneme[j] not in roman_ipa['C'].values:
            phoneme[j] = "V"
        else:
            phoneme[j] = "C"
    
    output = ''.join(phoneme)
    return output

#print(CV_mark(toJamo("안녕하세요"))) # ㅏㄴㄴㅕㅇㅎㅏㅅㅔㅛ -> VCCVCCVCVV
#print(CV_mark(toJamo("저기저비둘기"))) # ㅈㅓㄱㅣㅈㅓㅂㅣㄷㅜㄹㄱㅣ -> CVCVCVCVCVCCV

def CodaClusterSimplify(cv, jamo):
    # coda cluster simplication.
    # input:
    #   cv: char. vector
    #   jamo: char. vector
    #   main: bool (False if only need to treat a word-final cluster.
    # output: cv, jamo
    CCC_location = [m.start() for m in re.finditer("VCCC", cv)] # V"CC"C 패턴을 가지는 시작점 인덱스
    if len(CCC_location) > 0:
        for l in reversed(CCC_location):
            CCC_part = jamo[l+1:l+3]
            for i in range(len(criteria_DoubleCoda)): # ex. ㄱㅅ -> ㄱ, ㄹㄱ -> ㄱ
                if CCC_part == criteria_DoubleCoda.loc[i, 'separated']: # 자음군단순화에 속하는 부분 확인
                    jamo = jamo[:l+1] + criteria_DoubleCoda.loc[i, 'to'] + jamo[l+3:]
                    cv = cv[:l+1] + 'C' + cv[l+3:] # CC -> C
                    
    # 이상 CCC -> CC 해결
    # 아래 부분은 단어 끝에 나오는 자음연쇄(겹받침)의 음가를, 마치 뒤에 자음이 이어지는 것처럼 정해줌
    if re.search("CC$", cv):
        for l in range(len(criteria_DoubleCoda)):
            if re.search(criteria_DoubleCoda.loc[l, "separated"] + "$", jamo):
                jamo = re.sub(criteria_DoubleCoda.loc[l, "separated"], criteria_DoubleCoda.loc[l, "to"], jamo)
                cv = re.sub("CC$", "C", cv)
    return cv, jamo

def applyRulesToHangul(data, entry="entry", rules="pastcnhvr", convention="ipa"):
    # 규칙의 종류와 순서
    # (P)alatalization: 구개음화 (맏이 -> 마지)
    # (A)spiration: 격음화 (북한 -> 부칸)
    # a(S)similation: 음운동화
    # (T)ensification: 표준발음법 제23항(예외없는 경음화) 적용
    # (C)omplex coda simplification: 자음군단순화 (닭도 -> 닥도, 닭 -> 닥)
    # coda (N)eutralization: 음절말 장애음 중화 (빛/빚/빗 -> 빝)
    # intersonorant (H)-deletion: 공명음 사이 'ㅎ' 삭제
    # intersonorant Obstruent (V)oicing: 공명음 사이 장애음 유성음화
    
    if type(data) != str: # string 계열이 아닌 경우, 각 요청에 대하여 반환값을 정리하여 반환해줌.
        if any(isinstance(d, pd.DataFrame) for d in data):
            if data[entry] is None:
                #raise ValueError("Please specify a value for 'entry' parameter (a column name for wordforms). Default = 'entry'")
                return None
            list_data = list(data[entry])
            surface = [applyRulesToHangul(d, entry=entry, rules=rules, convention=convention) for d in list_data]
            surface = [item for sublist in surface for item in sublist]
            data["output"] = surface
            result = data
            return result
        else:
            #raise ValueError("Please input a character, data.frame, or tbl object.")
            return None

    if len(data) < 1: # if no content, then return no content
        return ""

    # 빈칸 혹은 특수기호가 있는 경우
    # to be implemented

    rules = rules.lower()

    data = sanitize(data)  # the function 'sanitize' converts all Hanja into hangul and removes string initial spaces
    jamo = toJamo(data, removeEmptyOnset=True)

    

    # cv = CV_mark(jamo)

    # # resolve word-final consonant clusters right off the bat
    # res_pack = CodaClusterSimplify(cv, jamo)
    # cv, jamo = res_pack
    # del res_pack

    # (P)alatalization: 구개음화 (맏이 -> 마지)
    if 'p' in rules and ('ㄷㅣ' in jamo or 'ㅌㅣ' in jamo):
        # syllable = convertHangulStringToJamos(data)
        # for i, s in enumerate(syllable):

        #     if len(syllable[i]) > 2: # 받침이 있는 경우
        #         DC = (criteria_DoubleCoda.loc[criteria_DoubleCoda['double'] == syllable[i][2]])
        #         if not DC.empty:  # 겹받침을 둘로 나눔 (e.g. "ㄳ" -> "ㄱㅅ")
        #             syllable[i] = syllable[i][:2] + DC['separated']
        separated_by_char = list(data)
        syllable = separated_by_char

        for i, s in enumerate(syllable):
            syllable[i] = toJamo(separated_by_char[i])
            phonemic = list(syllable[i]) # 'syllable'의 j번째 element를 각 자모단위로 분리해서 새로운 vector 'phonemic'에 넣습니다.
            # if len(phonemic) >= 3 and phonemic[2] == 'ㄷ':
            #     phonemic[2] = 'x'
            # if len(phonemic) >= 3 and phonemic[2] == 'ㅌ':
            #     phonemic[2] = 'X'
            # if phonemic[0] == 'ㅇ': # 첫 번째 자모(즉, 초성)가 'ㅇ'이면, 그것을 제거합니다.
            #     phonemic[0] = ''
            p_len = len(phonemic)
            if p_len > 1 and phonemic[p_len-1] == 'ㄷ': phonemic[p_len-1] = 'x'
            if p_len > 1 and phonemic[p_len-1] == 'ㅌ': phonemic[p_len-1] = 'X'
            syllable[i] = ''.join(phonemic) # 'phonemic'을 결합해서 다시 음절 단위로 만듭니다.
            
        jamo = ''.join(syllable) # 그 결과를 jamo 로.
        jamo = jamo.replace('xㅣ', 'ㅈㅣ')  # 구개음화 처리
        jamo = jamo.replace('Xㅣ', 'ㅊㅣ')
        jamo = jamo.replace('x', 'ㄷ')
        jamo = jamo.replace('X', 'ㅌ')

    

    # (A)spiration: 격음화 (북한 -> 부칸)
    if 'a' in rules and 'ㅎ' in jamo:
        for i in range(len(criteria_Aspiration)):
            if criteria_Aspiration['from'][i] in jamo:
                jamo = jamo.replace(criteria_Aspiration['from'][i], criteria_Aspiration['to'][i])

    if len(jamo) < 1:
        return ""

    cv = CV_mark(jamo) # 자음 및 모음에 대한 분류 진행 (ex. VCCVVCCV)
    
    # a(S)similation: 음운동화
    if re.search("s", rules):
        for l in range(len(criteria_Assimilation)):
            while re.search(criteria_Assimilation.loc[l, "from"], jamo):
                jamo = re.sub(criteria_Assimilation.loc[l, "from"], criteria_Assimilation.loc[l, "to"], jamo)

    # (T)ensification: 표준발음법 제23항(예외없는 경음화) 적용
    if re.search("t", rules):
        for l in range(len(criteria_Tensification)):
            while re.search(criteria_Tensification.loc[l, "from"], jamo):
                jamo = re.sub(criteria_Tensification.loc[l, "from"], criteria_Tensification.loc[l, "to"], jamo)

    # (C)omplex coda simplification: 자음군단순화 (닭도 -> 닥도, 닭 -> 닥) (닮았어 vs 담았어. 닮다 vs 담다)
    if 'c' in rules:
        CCC_location = [m.start() for m in re.finditer("VCCC", cv)] # V"CC"C 패턴을 가지는 시작점 인덱스
        if len(CCC_location) > 0:
            for l in reversed(CCC_location):
                CCC_part = jamo[l+1:l+3]
                for i in range(len(criteria_DoubleCoda)): # ex. ㄱㅅ -> ㄱ, ㄹㄱ -> ㄱ
                    if CCC_part == criteria_DoubleCoda.loc[i, 'separated']: # 자음군단순화에 속하는 부분 확인
                        jamo = jamo[:l+1] + criteria_DoubleCoda.loc[i, 'to'] + jamo[l+3:]
                        cv = cv[:l+1] + 'C' + cv[l+3:] # CC -> C
                        
        # 이상 CCC -> CC 해결
        # 아래 부분은 단어 끝에 나오는 자음연쇄(겹받침)의 음가를, 마치 뒤에 자음이 이어지는 것처럼 정해줌
        if re.search("CC$", cv):
            for l in range(len(criteria_DoubleCoda)):
                if re.search(criteria_DoubleCoda.loc[l, "separated"] + "$", jamo):
                    jamo = re.sub(criteria_DoubleCoda.loc[l, "separated"], criteria_DoubleCoda.loc[l, "to"], jamo)
                    cv = re.sub("CC$", "C", cv)

        # res_pack = CodaClusterSimplify(cv, jamo)
        # cv, jamo = res_pack
        # del res_pack

    # coda (N)eutralization: 음절말 장애음 중화 (빛/빚/빗 -> 빝)
    if re.search("n", rules):
        phoneme = list(jamo)
        for l in range(len(phoneme)):
            if phoneme[l] in list(neutral["from"]):
                if l == len(phoneme) - 1 or list(cv)[l+1] == "C":
                    phoneme[l] = str(neutral.loc[neutral["from"] == phoneme[l], "to"].values[0])
        jamo = "".join(phoneme)

    # intervocalic (H)-deletion: 공명음 사이 'ㅎ' 삭제
    if "h" in rules:
        phoneme = list(jamo)
        split_cv = list(cv)
        h_location = [i + 1 for i, ph in enumerate(phoneme[1:]) if ph == "ㅎ"]
        h_deletion_criteria = ["V", "C", "V"] # 'ㅎ' (= i) 이 consonant 인 경우

        for i in reversed(h_location):
            if i < len(phoneme):
                # check if /ㅎ/ comes after a sonorant
                if phoneme[i-1] in ['ㄴ', 'ㄹ', 'ㅇ', 'ㅁ']:
                    split_cv = split_cv[0:i] + split_cv[i+1:]
                    phoneme = phoneme[0:i] + phoneme[i+1:]
                else:
                    # check if /ㅎ/ comes in between two vowels
                    check_h_deletion = split_cv[i-1:i+1]
                    if check_h_deletion == h_deletion_criteria:
                        split_cv = split_cv[0:i] + split_cv[i+1:]
                        phoneme = phoneme[0:i] + phoneme[i+1:]

        cv = "".join(split_cv)
        jamo = "".join(phoneme)
    
    # 규칙적용 완료. 이하, IPA나 Yale로 변환
    romanization = None
    if convention == "ipa":
        romanization = roman_ipa
    elif convention == "yale":
        romanization = roman_yale
        
    jamo = list(jamo)

    for l in range(len(jamo)):
        if jamo[l] not in list(romanization['C']):
            if jamo[l] not in list(romanization['V']):
                if jamo[l] != ' ':
                    jamo[l] = ''
            else:
                jamo[l] = str(romanization.loc[romanization['V'] == jamo[l], 'VKlattese'].values[0])
        else:
            jamo[l] = str(romanization.loc[romanization['C'] == jamo[l], 'CKlattese'].values[0])

    # Yale convention 에서 bilabial 뒤 high mid/back vowel merger 적용하기
    # intervocalic Obstruent (V)oicing: 공명음 사이 장애음 유성음화
    if "u" in rules and convention == "yale":
        bilabials = ["p", "pp", "ph", "m"] # 양순음 bilabial: ㅂㅃㅍㅁ
        for j in range(len(jamo)):
            if jamo[j] in bilabials:
                if j+1 < len(jamo) and jamo[j+1] == "wu":
                    jamo[j+1] = "u"

    # 음성작용인 intervocalic voicing 적용 (공명음 사이 장애음 유성음화)
    if re.search("v", rules) and convention == "ipa":
        sonorants = ["n", "l", "ŋ", "m"] # 공명음
        cv_split = list(cv)

        for j in range(len(jamo)):
            if jamo[j] in sonorants or cv_split[j] == "V":
                if j+2 < len(jamo) and (jamo[j+2] in sonorants or cv_split[j+2] == "V"):
                    if jamo[j+1] == "p":
                        jamo[j+1] = "b"
                    if jamo[j+1] == "t":
                        jamo[j+1] = "d"
                    if jamo[j+1] == "k":
                        jamo[j+1] = "ɡ"
                    if jamo[j+1] == "tɕ":
                        jamo[j+1] = "dʑ"

    # 음성작용인 liquid alternation 적용
    if re.search("r", rules) and convention == "ipa" and 'l' in jamo:
        cv_split = list(cv)
        liquid_location = [i for i, x in enumerate(jamo) if x == "l"]
        liquid_location = [l for l in liquid_location if l < len(jamo)]

        if any(l > 0 for l in liquid_location):
            for l in [l for l in liquid_location if l > 1]:
                if l-1 >= 0 and l+1 < len(cv_split) and cv_split[l-1] == "V" and cv_split[l+1] == "V":
                    jamo[l] = "ɾ"

    # 수의작용인 non-coronalization 적용
    if re.search("o", rules) and convention == "ipa":
        velars = ["ɡ", "k", "k*", "kʰ"]
        bilabials = ["b", "p", "p*", "pʰ", "m"]
        non_velar_nasals = ['m', 'n']

        for j in range(len(jamo)):
            if jamo[j] in non_velar_nasals and j+1 < len(jamo):
                if jamo[j+1] in velars:
                    jamo[j] = "ŋ"
                elif jamo[j+1] in bilabials:
                    jamo[j] = "m"
                    
    # if '#' in jamo: jamo.remove('#')
    # if '$' in jamo: jamo.remove('$')
    jamo = [j for j in jamo if j not in ['$','#','',' ']]
    output = "".join(jamo)
    output_sparse = jamo

    return output, output_sparse

# 긴 문장도 한꺼번에 처리해주는 함수
def applyRulesToHangulTotal(text):
    # 특수문자 인덱스 찾기
    pattern = r'[\s?,.!]'
    matches = re.finditer(pattern, text)
    indices = [match.start() for match in matches]

    # 만약 split 할 데이터가 없다면 그대로 함수 호출 후 반환
    if len(indices) == 0:
        return applyRulesToHangul(text) # call function

    # 만약 split 할 데이터가 있다면 분할 후 처리
    result = []
    prev = int()
    
    for n, index in enumerate(indices):
        if n == 0: # 처음 문자열인 경우
            t = str(text[:index])
            if t != "":
                result.append(applyRulesToHangul(t)) # call function
        else: # 이전 문자열이 있는 경우
            t = str(text[prev+1:index])
            if t != "":
                result.append(applyRulesToHangul(t)) # call function
        result.append(text[index])
        prev = index # 이전 split 구간 기억

    # 마지막 텍스트 데이터에 대한 처리
    if prev + 1 < len(text):
        t = text[prev+1:]
        result.append(applyRulesToHangul(t)) # call function
        
    return ''.join(result)

if __name__ == '__main__':
    pass
    # print(applyRulesToHangul("SM엔터테인먼트", rules="pastcnv")) # kɑpt
    # print(applyRulesToHangul("ABC마트", rules="pastcnv")) # kɑpt
    #print(applyRulesToHangul("|", rules="pastcnv"))
    #print(applyRulesToHangul("A", rules="pastcnv"))
    #print(applyRulesToHangul("특히", rules="pastcnv"))
    # print(applyRulesToHangul("캉아지", rules="pastcnv")) 
    # print(applyRulesToHangul("돼다", rules="pastcnv")) 
    # print(applyRulesToHangul("$마트", rules="pastcnv")) # kɑpt
    # print(applyRulesToHangul("값", rules="pastcnv")) # kɑpt
    #print(applyRulesToHangul("육甲오㔡", rules="pastcnv")) # jukk*ɑbobjʌl
    #print(applyRulesToHangul("女子", rules="pastcnv")) # jʌdʑɑ
    print(applyRulesToHangul("굿", rules="pastcnv")) # jʌdʑɑ
    # print(applyRulesToHangul("金은房", rules="pastcnv")) # kɯmɯnbɑŋ
    # print(applyRulesToHangul("넓다", rules="pastcnv")) # nʌlt*ɑ
    # print(applyRulesToHangul("읽는다", rules="pastcnv")) # iŋnɯndɑ
    # print(applyRulesToHangul("안녕하세요", rules="pastcnv")) # ɑnnjʌŋɑsɛjo

    # print(applyRulesToHangul("닭도리탕", rules="pastcnv")) # tɑkt*olitʰɑŋ
    # print(applyRulesToHangul("읊지마", rules="pastcnv")) # ɯptɕ*imɑ

    # print(applyRulesToHangul("닭", rules="pastcnv")) # tɑk
    # print(applyRulesToHangul("닭도", rules="pastcnv")) # tɑkt*o

    # print(applyRulesToHangul("각막", rules="pastcnv")) # kɑŋmɑk
    # print(applyRulesToHangul("법률", rules="pastcnv")) # pʌmnjul

    # print(applyRulesToHangul("한라산", rules="pastcnv")) # hɑllɑsɑn
    # print(applyRulesToHangul("발냄새", rules="pastcnv")) # pɑllɛmsɛ

    # print(applyRulesToHangul("굳이?", rules="pastcnv")) # kudʑi?

    # print(applyRulesToHangul("툭하면", rules="pastcnv")) # tʰukʰɑmjʌn

    # print(applyRulesToHangul("박수", rules="pastcnv")) # pɑks*u

    # print(applyRulesToHangul("만화", rules="pastcnv")) # mɑnwa

    # print(applyRulesToHangul("한글", rules="pastcnv")) # hɑŋɡɯl

    # print(applyRulesToHangul("빛도", rules="pastcnv")) # pitt*o
    # print(applyRulesToHangul("빝도", rules="pastcnv")) # pitt*o
    # print(applyRulesToHangul("빗도", rules="pastcnv")) # pitt*o

    # print(applyRulesToHangul("안녕 나는 태현이야.", rules="pastcnv")) # ɑnnjʌŋ nɑnɯn tʰɛjʌnija.
