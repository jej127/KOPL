# reference repo
# https://github.com/cswangjiawei/pytorch-NER
# https://github.com/jiesutd/NCRFpp

import random
import torch
import numpy as np
import argparse
import os
from utils import WordVocabulary, LabelVocabulary, Alphabet, build_pretrain_embedding, my_collate_fn, lr_decay
import time
from dataset import MyDataset_woChar
from torch.utils.data import DataLoader
from model import NamedEntityRecog_woChar
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from train import train_model_wochar, evaluate_wochar




parser = argparse.ArgumentParser(description='Named Entity Recognition Model')
parser.add_argument('--word_embed_dim', type=int, default=300)
parser.add_argument('--word_hidden_dim', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.33)
parser.add_argument('--pretrain_embed_path', default='/mnt/oov/klue-ner/love_kor.emb')
parser.add_argument('--output_path', default='./extrinsic/klue-ner/output/')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--feature_extractor', choices=['lstm', 'cnn', 'linear'], default='lstm')
parser.add_argument('--train_path', default='./extrinsic/klue-ner/data/train_split.txt')
parser.add_argument('--dev_path', default='./extrinsic/klue-ner/data/dev_split.txt')
parser.add_argument('--test_path', default='./extrinsic/klue-ner/data/test_split.txt')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--number_normalized', type=bool, default=False)
parser.add_argument('--use_crf', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--postfix', type=str, default='')



def single_run(args, word_vocab, pretrain_word_embedding):

    use_gpu = torch.cuda.is_available()
    print('use_crf:', args.use_crf)
    print('emb dim:', args.word_embed_dim)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    eval_path = "./extrinsic/klue-ner/evaluation"
    eval_temp = os.path.join(eval_path, "temp")
    eval_script = os.path.join(eval_path, "conlleval")

    if not os.path.isfile(eval_script):
        raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)

    pred_file = eval_temp + '/pred.txt'
    score_file = eval_temp + '/score.txt'

    model_name = args.output_path + '/' + args.feature_extractor + str(args.use_crf)
    #model_name = args.output_path + args.feature_extractor + str(args.use_char) + str(args.use_crf)
    #word_vocab = WordVocabulary(args.train_path, args.dev_path, args.test_path, args.number_normalized)
    label_vocab = LabelVocabulary(args.train_path)

    # emb_begin = time.time()
    # pretrain_word_embedding = build_pretrain_embedding(args.pretrain_embed_path, word_vocab, args.word_embed_dim)
    # emb_end = time.time()
    # emb_min = (emb_end - emb_begin) % 3600 // 60
    # print('build pretrain embed cost {}m'.format(emb_min))

    train_dataset = MyDataset_woChar(args.train_path, word_vocab, label_vocab, args.number_normalized)
    dev_dataset = MyDataset_woChar(args.dev_path, word_vocab, label_vocab, args.number_normalized)
    test_paths = [args.test_path]
    test_paths += [f"./extrinsic/klue-ner/data/test_split_natural_{r}.txt" for r in [30]]
    test_datasets = [MyDataset_woChar(test_path, word_vocab, label_vocab, args.number_normalized) for test_path in test_paths]
    #test_dataset = MyDataset_woChar(args.test_path, word_vocab, label_vocab, args.number_normalized)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    test_dataloaders = [DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn) for test_dataset in test_datasets]

    model = NamedEntityRecog_woChar(word_vocab.size(), args.word_embed_dim, args.word_hidden_dim, args.num_layers,
                                    args.feature_extractor, label_vocab.size(), args.dropout,
                                    pretrain_embed=pretrain_word_embedding, use_crf=args.use_crf, use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_begin = time.time()
    print('train begin', '-' * 50)
    print()

    writer = SummaryWriter('log')
    batch_num = -1
    best_f1 = -1
    early_stop = 0

    for epoch in range(args.epochs):
        epoch_begin = time.time()
        print('train {}/{} epoch'.format(epoch + 1, args.epochs))
        optimizer = lr_decay(optimizer, epoch, 0.05, args.lr)
        batch_num = train_model_wochar(train_dataloader, model, optimizer, batch_num, writer, use_gpu)
        new_f1 = evaluate_wochar(dev_dataloader, model, word_vocab, label_vocab, pred_file, score_file, eval_script, use_gpu)
        print('f1 is {} at {}th epoch on dev set'.format(new_f1, epoch + 1))
        if new_f1 > best_f1:
            best_f1 = new_f1
            print('new best f1 on dev set:', best_f1)
            early_stop = 0
            torch.save(model.state_dict(), model_name)
        else:
            early_stop += 1

        epoch_end = time.time()
        cost_time = epoch_end - epoch_begin
        print('train {}th epoch cost {}m {}s'.format(epoch + 1, int(cost_time / 60), int(cost_time % 60)))
        print()

        if early_stop > args.patience:
            print('early stop')
            break

    train_end = time.time()
    train_cost = train_end - train_begin
    hour = int(train_cost / 3600)
    min = int((train_cost % 3600) / 60)
    second = int(train_cost % 3600 % 60)
    print()
    print()
    print('train end', '-' * 50)
    print('train total cost {}h {}m {}s'.format(hour, min, second))
    print('-' * 50)

    
    model.load_state_dict(torch.load(model_name))
    ratios = ['0%','30_%']
    f1_list = []
    for ratio, test_dataloader in zip(ratios, test_dataloaders):
        r = ratio[:-1]
        pred_file = eval_temp + f'/pred_{r}.txt'
        score_file = eval_temp + f'/score_{r}.txt'
        test_acc = evaluate_wochar(test_dataloader, model, word_vocab, label_vocab, pred_file, score_file, eval_script, use_gpu)
        print(f'test F1 on test set (OOV: {ratio}):', test_acc)
        f1_list.append(test_acc)

    output_all_eval_file = os.path.join(args.output_path, "eval_all_results.txt")
    with open(output_all_eval_file, "a") as all_writer:
        #all_writer.write("seed: %d | test acc: %.2f | lr: %.4f\n" % (args.seed, test_acc, args.lr))
        all_writer.write("eval results:\n")
        all_writer.write("%s\n" % (args.postfix))
        for f1 in f1_list:
            all_writer.write("%s\n" % (f"{f1:.2f}"))
        all_writer.write("-"*25+"\n")
    return test_acc

if __name__ == '__main__':
    args = parser.parse_args()
    seed_num = args.seed
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)

    test_paths = [args.test_path]
    word_vocab = WordVocabulary(args.train_path, args.dev_path, test_paths, args.number_normalized)

    #args.pretrain_embed_path = f'output/love{oov}.emb'
    # args.word_embed_dim = 300
    # print(args.pretrain_embed_path)

    pretrain_word_embedding = build_pretrain_embedding(args.pretrain_embed_path, word_vocab, args.word_embed_dim)
    single_run(args, word_vocab, pretrain_word_embedding)


