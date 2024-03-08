import torch
import numpy as np
from scipy import stats
import math
from model import registry as Producer
from torch.utils.data import DataLoader
from utils import load_predict_dataset, TextData, collate_fn_predict_, add_tokens, load_ipa
from transformers import AutoTokenizer

rare_words = ['이복누이', '마키아밸리', '미얀아', '간자미', '풀치', '능소니', '꺼병이', '장사니', '학배기', '초고리', '발강이', '꽝다리', '개호주', '시허연', 
              '허여스름한', '꺼무스름한', '개굴', '개미굴', '영국령과', '미국산은', '항산화제와', '부대찌개로', '신경과를', '마이클잭슨이', '신경과가', '행정고시와', 
              '오소리가', '국악과는', '자갈치로', '마이클잭슨을', '동서양은', '이스라엘으로', '새으로', '신경과와', '양상추가', '자갈치를', '제트스키는', 
              '대추나무와', '중환자실으로', '티머니로', '자갈치와', '신경과로', '영국령을', '카페인으로', '알코올으로', '스크린샷으로', '과일으로', '가발으로', 
              '국민대와', '원석과', '스크린샷과', '팝콘으로', '연속극과', '마이클잭슨은', '영국령은', '까마귀로', '이불으로', '수도승으로', '아이비리그는', 
              '정오는', '국민대를', '마라도나로', '키즈까페', '키즈까페를', '동서양과', '행정고시가', '새이', '국악과로', '홈런포와', '뉴발란스를', '부대찌개가', 
              '달걀으로', '새과', '렌트카와', '마이클잭슨으로', '자갈치가', '아이비리그가', '아이비리그를', '동서양으로', '제트스키가', '렌트카가', '국민대로', 
              '쌍용차로', '상도가', '새은', '재규어로', '국악과가', '아이비리그와', '아이비리그로', '수도승은', '뉴발란스로', '오소리로', '키즈까페로', '박진영으로', 
              '한국말으로', '단축키와', '올레길으로', '홈런포는', '마이클잭슨과', '수도승을', '항산화제는', '자갈치는', '제트스키로', '상도는', '혈액형으로', 
              '단축키가', '물리로', '양상추로', '나물으로', '결승점과', '제트스키와', '벽시계와', '키즈까페와', '수도승과', '테이블으로', '렌트카는', '스크린샷은', 
              '국악과와', '겨울철으로', '국민대가', '새을', '렌트카로', '키즈까페가', '대추나무로', '벽시계로', '행정고시는', '키즈까페는', '도착해요', '소환됩니다', 
              '소환됐습니다', '향합니다', '뜯지만', '놓음', '뜯음', '인정함', '배우셨다', '이끄시고', '주무셨다']

def produce(args, model_path, tokenizer, batch_size=32, vocab_path='data/word_sim/evaluate_words.txt', ipa_path="data/word_sim_kor/evaluate_ipas.txt"):
    dataset = load_predict_dataset(path=vocab_path)
    if args.use_ipa:
        word_to_ipa, _ = load_ipa(ipa_path)
    else:
        word_to_ipa = None
    dataset = TextData(dataset)
    train_iterator = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=lambda x: collate_fn_predict_(x, args, tokenizer, word_to_ipa))
    model = Producer[args.model_type](args)
    model.load_state_dict(torch.load(model_path))
    total_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('in total, K-LOVE has {a} parameters'.format(a=total_num))
    model.eval()
    model.cuda()

    embeddings = dict()
    for words, _,  batch_repre_ids, batch_repre_ids_ipa, mask, mask_ipa in train_iterator:
        batch_repre_ids = batch_repre_ids.cuda()
        mask = mask.cuda()
        if mask_ipa is not None: 
            batch_repre_ids_ipa = batch_repre_ids_ipa.cuda()
            mask_ipa = mask_ipa.cuda()
        emb,_ = model(batch_repre_ids, mask, batch_repre_ids_ipa, mask_ipa)
        emb = emb.cpu().detach().numpy()
        embeddings.update(dict(zip(words, emb)))
    return embeddings

def cosine_sim(vec1, vec2):
    vec1 = np.add(vec1, 1e-8 * np.ones_like(vec1))
    vec2 = np.add(vec2, 1e-8 * np.ones_like(vec2))
    return np.dot(vec1, vec2) / ((np.linalg.norm(vec1) + 1e-8) * (np.linalg.norm(vec2) + 1e-8))

def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (n1 * n2 + 1e-7)

def analogy(embeddings):
    distance = 0
    category = "sem1_capital-conturies"
    score_list = []
    distance_oov = 0
    len_oov = 0
    with open("data/word_sim_kor/word_analogy_korean.txt", "r") as g:
        for line in g.readlines():
            try:
                a, b, c, d = line.split()
                a, b, c, d = a, b, c, d
                a_vec = np.array(embeddings[a]) / np.linalg.norm(np.array(embeddings[a]) + 1e-8)
                b_vec = np.array(embeddings[b]) / np.linalg.norm(np.array(embeddings[b]) + 1e-8)
                c_vec = np.array(embeddings[c]) / np.linalg.norm(np.array(embeddings[c]) + 1e-8)
                d_vec = np.array(embeddings[d]) / np.linalg.norm(np.array(embeddings[d]) + 1e-8)
                score = 1 - similarity(-a_vec + b_vec + c_vec, d_vec)

                distance += score
                if sum([int(w in rare_words) for w in [a,b,c,d]]) > 1:
                    distance_oov += score
                    len_oov += 1
            except:
                if distance > 0:
                    print(f"{category} -> {distance / 1000}")
                    #score_list.append(f"{category} -> {distance / 1000}")
                    score_list.append("{:.4f}".format(distance / 1000))
                    distance = 0
                    category = line.split()[1]

    print(f"{category} -> {distance / 1000}")
    print(f"OOV -> {distance_oov / len_oov}")
    score_list.append("{:.4f}".format(distance / 1000))
    score_list.append("{:.4f}".format(distance_oov / len_oov))
    return score_list


def word_sim(embeddings):
    with open("data/word_sim_kor/WS353_korean.csv", "r") as g:
        sims, cosines = [], []
        for line in g.readlines()[1:]:
            word1, word2, sim = line.strip().split(",")
            word1, word2 = word1, word2
            sims.append(eval(sim))
            cos_sim = similarity(embeddings[word1], embeddings[word2])
            cosines.append(cos_sim)
    print("Word Sim: {:.4f}".format(stats.spearmanr(sims, cosines).correlation))
    return ["{:.4f}".format(stats.spearmanr(sims, cosines).correlation)]

def overall(args, model_path, tokenizer):
    all_score = list()
    # embeddings = produce(args, model_path=model_path, tokenizer=tokenizer)
    embeddings = produce(args, model_path, tokenizer, batch_size=32,
                         vocab_path="data/word_sim_kor/evaluate_words.txt",
                         ipa_path="data/word_sim_kor/evaluate_ipas.txt")
    sim = word_sim(embeddings)
    anal = analogy(embeddings)
    return anal + sim


if __name__ == '__main__':
    import os
    from train import args
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    TOKENIZER = AutoTokenizer.from_pretrained("klue/bert-base")
    #args.input_type = 'mixed'
    if args.use_ipa:
        word_to_ipa, ipa_set = load_ipa(args.ipa_path)
    else:
        word_to_ipa, ipa_set = None, []
    TOKENIZER = add_tokens(TOKENIZER)
    vocab_size = len(TOKENIZER)

    args.vocab_size = vocab_size
    args.ipa_vocab_size = len(ipa_set) + 4
    print(args)
    results = overall(args, model_path='./output/kops/model_7_122.pt', tokenizer=TOKENIZER)
    #results = overall(args, model_path='./output/love/model_7_444.pt', tokenizer=TOKENIZER)
    for s in results: print(s)
