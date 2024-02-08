import torch
import numpy as np
from scipy import stats
import math
from model import registry as Producer
from torch.utils.data import DataLoader
from utils import load_predict_dataset, TextData, collate_fn_predict_, add_tokens, load_ipa
from transformers import AutoTokenizer


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
            except:
                if distance > 0:
                    print(f"{category} -> {distance / 1000}")
                    #score_list.append(f"{category} -> {distance / 1000}")
                    score_list.append("{:.4f}".format(distance / 1000))
                    distance = 0
                    category = line.split()[1]

    print(f"{category} -> {distance / 1000}")
    #score_list.append(f"{category} -> {distance / 1000}")
    score_list.append("{:.4f}".format(distance / 1000))
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
    #results = overall(args, model_path='./output/k_love3/model_7_111.pt', tokenizer=TOKENIZER)
    #results = overall(args, model_path='./output/k_love4/model_3_222_0.2.pt', tokenizer=TOKENIZER)
    #results = overall(args, model_path='./output/k_love4/model_9_102_0.1.pt', tokenizer=TOKENIZER)
    results = overall(args, model_path='./output/k_love2/model_7_122.pt', tokenizer=TOKENIZER)
    #results = overall(args, model_path='./output/k_love/model_7_222.pt', tokenizer=TOKENIZER)
    #results = overall(args, model_path='./output/k_love/model_7_444.pt', tokenizer=TOKENIZER)
    #results = overall(args, model_path='./output/k_love2_bts/model_8_122_0.098839020547.pt', tokenizer=TOKENIZER)
    for s in results: print(s)
