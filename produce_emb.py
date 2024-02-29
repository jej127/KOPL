import torch
from model import registry as Producer
from torch.utils.data import DataLoader
from utils import TextData, collate_fn_predict_, add_tokens, load_ipa
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import os

from train import args
#model_path = 'output/model_ merge.pt'

TOKENIZER = AutoTokenizer.from_pretrained("klue/bert-base")
TOKENIZER = add_tokens(TOKENIZER)
vocab_size = len(TOKENIZER)
args.vocab_size = vocab_size
# args.att_head_num=2
# args.encoder_layer=2

def produce(word, batch_size=1):
    dataset = {'origin_word': [word], 'origin_repre':[None]}
    dataset = TextData(dataset)
    train_iterator = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn_predict_(x, TOKENIZER))
    model = Producer[args.model_type](args)
    model.load_state_dict(torch.load(args.ex_model_path))
    model.eval()
    model.cuda()

    embeddings = dict()
    for words, _, batch_repre_ids, mask in train_iterator:
        batch_repre_ids = batch_repre_ids.cuda()
        mask = mask.cuda()
        emb,_ = model(batch_repre_ids, mask)
        emb = emb.cpu().detach().numpy()
        embeddings.update(dict(zip(words, emb)))
    return embeddings

def gen_embeddings_for_vocab(vocab_path, emb_path, ipa_path=None, batch_size=32):
    vocab = [line.strip() for line in open(vocab_path, encoding='utf8')]
    if ipa_path is not None:
        assert args.use_ipa
        word_to_ipa, ipa_set = load_ipa(ipa_path)
        args.ipa_vocab_size = len(ipa_set) + 4
        args.ipa_vocab_size = 44
        if 'bert' in args.ipa_path: args.ipa_vocab_size += 1
        if args.random_search:
            args.lamda = 0.1*np.random.rand() + 0.1
            print('A sampled lambda is %f' % (args.lamda))
    else:
        assert not args.use_ipa
        word_to_ipa, _ = None, []
    print(args.input_type)
    print('use ipa: ', args.use_ipa)
    dataset = {'origin_word': vocab, 'origin_repre': [None for _ in range(len(vocab))]}
    dataset = TextData(dataset)
    train_iterator = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda x: collate_fn_predict_(x, args, TOKENIZER, word_to_ipa))
    model = Producer[args.model_type](args)
    model.load_state_dict(torch.load(args.ex_model_path))
    model.eval()
    model.cuda()

    embeddings = dict()
    for words, _, batch_repre_ids, batch_repre_ids_ipa, mask, mask_ipa in tqdm(train_iterator, desc='Generating Embeddings..'):
        batch_repre_ids = batch_repre_ids.cuda()
        mask = mask.cuda()
        if mask_ipa is not None: 
            batch_repre_ids_ipa = batch_repre_ids_ipa.cuda()
            mask_ipa = mask_ipa.cuda()
        emb,_ = model(batch_repre_ids, mask, batch_repre_ids_ipa, mask_ipa)
        emb = emb.cpu().detach().numpy()
        embeddings.update(dict(zip(words, emb)))

    save_path = '/'.join(emb_path.split('/')[:-1])
    if not os.path.exists(save_path): os.makedirs(save_path)
    wl = open(emb_path, 'w', encoding='utf8')
    for word, embedding in embeddings.items():
        emb_str = ' '.join([str(e) for e in list(embedding)])
        wl.write(word + ' ' + emb_str + '\n')
    wl.close()

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #gen_embeddings_for_vocab(vocab_path='extrinsic/rnn_ner/output/words.txt', emb_path='extrinsic/rnn_ner/output/love.emb')
    # gen_embeddings_for_vocab(vocab_path='extrinsic/cnn_text_classification/output/words.txt',
    #                          emb_path='extrinsic/cnn_text_classification/output/love.emb')
    emb = produce('mispelling')
    print(emb)

