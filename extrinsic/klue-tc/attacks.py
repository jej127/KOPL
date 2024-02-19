import random
import jamotools
import hgtk
from Levenshtein import distance

consonant_group = [['ㄱ','ㄲ','ㅋ'],
                   ['ㄴ','ㄷ','ㅌ','ㄹ','ㄸ'],
                   ['ㅁ','ㅂ','ㅍ','ㅃ'],
                   ['ㅅ','ㅈ','ㅊ','ㅆ','ㅉ'],
                   ['ㅇ','ㅎ']]
consonant_dict = {}
for group in consonant_group:
    for letter in group:
        consonant_dict[letter] = [s for s in group if s!=letter]

mo_to_cji = {
    'ㅏ':'ㅣ'+'ㆍ', 
    'ㅐ':'ㅣ'+'ㆍ'+'ㅣ', 
    'ㅑ':'ㅣ'+'ㆍ'+'ㆍ', 
    'ㅒ':'ㅣ'+'ㆍ'+'ㆍ'+'ㅣ', 
    'ㅓ':'ㆍ'+'ㅣ', 
    'ㅔ':'ㆍ'+'ㅣ'+'ㅣ', 
    'ㅕ':'ㆍ'+'ㆍ'+'ㅣ', 
    'ㅖ':'ㆍ'+'ㆍ'+'ㅣ'+'ㅣ', 
    'ㅗ':'ㆍ'+'ㅡ', 
    'ㅘ':'ㆍ'+'ㅡ'+'ㅣ'+'ㆍ', 
    'ㅙ':'ㆍ'+'ㅡ'+'ㅣ'+'ㆍ'+'ㅣ', 
    'ㅚ':'ㆍ'+'ㅡ'+'ㅣ', 
    'ㅛ':'ㆍ'+'ㆍ'+'ㅡ', 
    'ㅜ':'ㅡ'+'ㆍ', 
    'ㅝ':'ㅡ'+'ㆍ'+'ㆍ'+'ㅣ', 
    'ㅞ':'ㅡ'+'ㆍ'+'ㆍ'+'ㅣ'+'ㅣ', 
    'ㅟ':'ㅡ'+'ㆍ'+'ㅣ', 
    'ㅠ':'ㅡ'+'ㆍ'+'ㆍ', 
    'ㅡ':'ㅡ', 
    'ㅢ':'ㅡ'+'ㅣ', 
    'ㅣ':'ㅣ'
    }

vowel_dict = {}
for v in mo_to_cji:
    distances = {v_:distance(mo_to_cji[v], mo_to_cji[v_]) for v_ in mo_to_cji}
    vowel_dict[v] = [v_ for v_ in mo_to_cji if distances[v_]==1]
     
def replace_bts(word):
    for _ in range(30):
        try:
            rand_word_idx = random.randint(0, len(word)-1)
            cho_joong_jong = hgtk.letter.decompose(word[rand_word_idx])
            rand_char_idx = random.randint(0, len(''.join(cho_joong_jong))-1)
            cho, joong, jong = cho_joong_jong[0], cho_joong_jong[1], cho_joong_jong[2]
            jamo = cho_joong_jong[rand_char_idx]
            if rand_char_idx == 0:
                changed_jamo = random.choice(consonant_dict[jamo])
                #compose = jamotools.join_jamos(changed_jamo + joong + jong)
                compose = hgtk.letter.compose(changed_jamo, joong, jong)
            elif rand_char_idx == 1:
                changed_jamo = random.choice(vowel_dict[jamo])
                #compose = jamotools.join_jamos(cho + changed_jamo + jong)
                compose = hgtk.letter.compose(cho, changed_jamo, jong)
            else:
                changed_jamo = random.choice(consonant_dict[jamo])
                #compose = jamotools.join_jamos(cho + joong + changed_jamo)
                compose = hgtk.letter.compose(cho, joong, changed_jamo)
            typo_word = word[:rand_word_idx] + compose + word[rand_word_idx+1:]
            return typo_word
        except:
            continue
    return word

def select_attack(word, attack_prob):
    if random.random() <= attack_prob:
        return replace_bts(word)
    else:
        return word

def main():
    print(consonant_dict)
    print(vowel_dict)

    for _ in range(10):
        print(replace_bts('거제도'))

if __name__ == '__main__':
    main()