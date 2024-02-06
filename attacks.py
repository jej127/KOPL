import random
import numpy as np
import hgtk
import jamotools

keyboard_replace =  {
    'ㅂ': ['ㅈ', 'ㅁ', 'ㄴ'],
    'ㅃ': ['ㅉ', 'ㅁ', 'ㄴ'],
    'ㅈ': ['ㅂ', 'ㄷ', 'ㅁ', 'ㄴ', 'ㅇ'],
    'ㅉ': ['ㅃ', 'ㄸ', 'ㅁ', 'ㄴ', 'ㅇ'],
    'ㄷ': ['ㅈ', 'ㄱ', 'ㄴ', 'ㅇ', 'ㄹ'],
    'ㄸ': ['ㅉ', 'ㄲ', 'ㄴ', 'ㅇ', 'ㄹ'],
    'ㄱ':['ㄷ', 'ㅅ', 'ㅇ', 'ㄹ', 'ㅎ'],
    'ㄲ':['ㄸ', 'ㅆ', 'ㅇ', 'ㄹ', 'ㅎ'],
    'ㅅ': ['ㄱ', 'ㄹ', 'ㅎ'],
    'ㅆ': ['ㄲ', 'ㄹ', 'ㅎ'],
    'ㅛ': ['ㅕ', 'ㅗ', 'ㅓ'],
    'ㅕ':['ㅛ', 'ㅑ', 'ㅗ', 'ㅓ', 'ㅏ'],
    'ㅑ':['ㅕ', 'ㅐ', 'ㅓ', 'ㅏ', 'ㅣ'],
    'ㅐ':['ㅑ', 'ㅔ', 'ㅏ', 'ㅣ'],
    'ㅒ':['ㅑ', 'ㅖ', 'ㅏ', 'ㅣ'],
    'ㅔ':['ㅐ', 'ㅣ'],
    'ㅖ':['ㅒ', 'ㅣ'],
    'ㅁ': ['ㅂ', 'ㅈ', 'ㄴ', 'ㅋ', 'ㅌ'],
    'ㄴ': ['ㅂ', 'ㅈ', 'ㄷ', 'ㅁ', 'ㅇ', 'ㅋ', 'ㅌ', 'ㅊ'],
    'ㅇ': ['ㅈ', 'ㄷ', 'ㄱ', 'ㄴ', 'ㄹ', 'ㅌ', 'ㅊ', 'ㅍ'],
    'ㄹ': ['ㄷ', 'ㄱ', 'ㅅ', 'ㅇ', 'ㅎ', 'ㅊ', 'ㅍ', 'ㅠ'],
    'ㅎ': ['ㄱ', 'ㅅ', 'ㅛ', 'ㄹ', 'ㅗ', 'ㅍ'],
    'ㅗ': ['ㅛ', 'ㅕ', 'ㅓ', 'ㅠ', 'ㅜ'],
    'ㅓ': ['ㅛ', 'ㅕ', 'ㅑ', 'ㅗ', 'ㅏ', 'ㅜ', 'ㅡ'],
    'ㅏ': ['ㅕ', 'ㅑ', 'ㅐ', 'ㅓ', 'ㅣ', 'ㅡ'],
    'ㅣ': ['ㅐ', 'ㅔ', 'ㅏ', 'ㅑ'],
    'ㅋ': ['ㅁ', 'ㄴ', 'ㅌ'],
    'ㅌ': ['ㅁ', 'ㄴ', 'ㅇ', 'ㅋ', 'ㅊ'],
    'ㅊ': ['ㄴ', 'ㅇ', 'ㄹ', 'ㅌ', 'ㅍ'],
    'ㅍ': ['ㅇ', 'ㄹ', 'ㅎ', 'ㅊ'],
    'ㅠ': ['ㅗ', 'ㅜ'],
    'ㅜ': ['ㅗ', 'ㅓ', 'ㅠ', 'ㅡ'],
    'ㅡ': ['ㅜ', 'ㅓ', 'ㅏ']}
chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
jongsung_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ' , 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 
                 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
ipa_list = ['tɕ', 'u','ju', 'o', 'ɛ', 'ɑ','jo','p*', 'h', 'l', 'ɯ', 'kʰ','i','wi','ɡ','wʌ','ɰi', 'n', 'b','tɕʰ', 
             'd', 'k','t*','pʰ', 't','jɛ','s*','k*','ja','tʰ','wɛ','tɕ*','ŋ','jʌ','p', 'ʌ', 's','wa','dʑ','m']

def add_jamo(word):
    new_jamo = random.choice(chosung_list + jungsung_list + jongsung_list)
    rand_idx = random.randint(0, len(word)-1)
    typo_word = word[:rand_idx] + new_jamo + word[rand_idx:]
    return typo_word

def drop_jamo(word):
    for _ in range(30):
        try:
            rand_idx = random.randint(0, len(word)-1)
            chars = hgtk.letter.decompose(word[rand_idx])
            idx = random.randint(0,len(''.join(chars))-1)
            trimmed_chars = [chars[i] for i in range(3) if i!=idx]
            typo_word = word[:rand_idx] + ''.join(trimmed_chars) + word[rand_idx+1:]
            return typo_word
        except:
            return word

def reverse_jamo(word):
    for _ in range(30):
        decompose_word = []
        # 자모 분리
        for ch in word:
            try:
                decompose_character = list(hgtk.letter.decompose(ch))
                decompose_word += decompose_character
            except hgtk.exception.NotHangulException: # 한글 아니면 그냥 word를 return
                return word
        rand_idx = random.randint(0, len(decompose_word)-2)
        try:
            assert decompose_word[rand_idx+1] != ''
            temp = decompose_word[rand_idx]
            decompose_word[rand_idx] = decompose_word[rand_idx+1]
            decompose_word[rand_idx+1] = temp
        except:
            continue
        cnt, jamo_word, typo_word = 0,"",""
        for jm in decompose_word:
            cnt += 1
            jamo_word += jm
            if cnt % 3 == 0:
                jamo_word += "/"
                typo_word += hgtk.text.compose(jamo_word, compose_code="/")
                jamo_word = ""
        return typo_word
    return word

def replace_keyboard(word):
    for _ in range(30):
        try:
            rand_word_idx = random.randint(0, len(word)-1)
            cho_joong_jong = hgtk.letter.decompose(word[rand_word_idx])
            rand_char_idx = random.randint(0, len(''.join(cho_joong_jong))-1)
            cho, joong, jong = cho_joong_jong[0], cho_joong_jong[1], cho_joong_jong[2]
            jamo = cho_joong_jong[rand_char_idx]
            changed_jamo = random.choice(keyboard_replace[jamo])
            if rand_char_idx == 0:
                #compose = hgtk.letter.compose(changed_jamo, joong, jong)
                compose = jamotools.join_jamos(changed_jamo + joong + jong)
            elif rand_char_idx == 1:
                #compose = hgtk.letter.compose(cho, changed_jamo, jong)
                compose = jamotools.join_jamos(cho + changed_jamo + jong)
            else:
                #compose = hgtk.letter.compose(cho, joong, changed_jamo)
                compose = jamotools.join_jamos(cho + joong + changed_jamo)
            typo_word = word[:rand_word_idx] + compose + word[rand_word_idx+1:]
            return typo_word
        except:
            continue
    return word

def get_random_attack(word, probs=[0.12, 0.12, 0.12, 0.12, 0.52]): # kops aug
    attack_type = ["add", "drop", "reverse", "keyboard", "unchange"]
    attack_probs = np.array(probs)
    attack_probs = attack_probs / sum(attack_probs)
    attack = np.random.choice(attack_type, 1, p=attack_probs)[0]

    num_typos = 1
    for _ in range(num_typos):
        if attack == "add":
            return add_jamo(word)
        elif attack == "drop":
            return drop_jamo(word)
        elif attack == "reverse":
            return reverse_jamo(word)
        elif attack == "keyboard":
            return replace_keyboard(word)
        elif attack == "unchange":
            return word

def add_ipa(ipa):
    new_ipa = random.choice(ipa_list)
    rand_idx = random.randint(0, len(ipa))
    typo_ipa = ipa[:rand_idx] + [new_ipa] + ipa[rand_idx:]
    return typo_ipa

def drop_ipa(ipa):
    rand_idx = random.randint(0, len(ipa)-1)
    typo_ipa = ipa[:rand_idx] + ipa[rand_idx+1:]
    return typo_ipa

def reverse_ipa(ipa):
    if len(ipa)==1: return ipa
    rand_idx = random.randint(0, len(ipa)-2)
    typo_ipa = ipa.copy()
    temp = typo_ipa[rand_idx]
    typo_ipa[rand_idx] = typo_ipa[rand_idx+1]
    typo_ipa[rand_idx+1] = temp
    return typo_ipa

def get_random_attack_ipa(ipa, probs=[0.16, 0.16, 0.16, 0.52]): # kops aug
    if '[UNK]' in ipa: return ipa
    attack_type = ["add", "drop", "reverse", "unchange"]
    attack_probs = np.array(probs)
    attack_probs = attack_probs / sum(attack_probs)
    attack = np.random.choice(attack_type, 1, p=attack_probs)[0]
    if attack == "add":
        return add_ipa(ipa)
    elif attack == "drop":
        return drop_ipa(ipa)
    elif attack == "reverse":
        return reverse_ipa(ipa)
    elif attack == "unchange":
        return ipa

if __name__ == '__main__':
    pass