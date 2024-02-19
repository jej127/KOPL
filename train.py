import torch
import torch.optim as optim
import sys
import os
import argparse
from transformers import AutoTokenizer
from torch.optim import lr_scheduler
from loss import registry as loss_f
from loader import registry as loader
from model import registry as Producer
from evaluate import overall
from utils import add_tokens, load_ipa
import random
import numpy as np
from time import time

#hyper-parameters
parser = argparse.ArgumentParser(description='contrastive learning framework for word vector')
parser.add_argument('-dataset', help='the file of target vectors', type=str, default='data/wiki_100.vec')
parser.add_argument('-batch_size', help='the number of samples in one batch', type=int, default=64)
parser.add_argument('-epochs', help='the number of epochs to train the model', type=int, default=10)
parser.add_argument('-min_epoch_to_save', help='minimum number of epochs to save the model', type=int, default=6)
parser.add_argument('-shuffle', help='whether shuffle the samples', type=bool, default=True)
#parser.add_argument('-lowercase', help='if only use lower case', type=bool, default=True)
parser.add_argument('-model_type', help='sum, rnn, cnn, attention, pam', type=str, default='self_attention_2')
parser.add_argument('-encoder_layer', help='the number of layer of the encoder', type=int, default=2)
parser.add_argument('-encoder_layer_ipa', help='the number of layer of the encoder', type=int, default=2)
parser.add_argument('-merge', help='merge pam and attention layer', type=bool, default=True)
parser.add_argument('-att_head_num', help='the number of attentional head for the pam encoder', type=int, default=2)
parser.add_argument('-loader_type', help='simple, aug', type=str, default='aug_ipa')
parser.add_argument('-loss_type', help='mse, ntx, align_uniform', type=str, default='ntx')
parser.add_argument('-input_type', help='mixed, char, sub, jamo', type=str, default='mixed')
parser.add_argument("-use_ipa", help="Whether to use IPA", action='store_true')
parser.add_argument('-learning_rate', help='learning rate for training', type=float, default=2e-3)
parser.add_argument('-drop_rate', help='the rate for dropout', type=float, default=0.1)
parser.add_argument('-gamma', help='decay rate', type=float, default=0.97)
parser.add_argument('-input_dim', help='the dimension of input embeddings', type=int, default=300)
parser.add_argument('-emb_dim', help='the dimension of target embeddings (FastText:300; BERT:768)', type=int, default=300)
parser.add_argument('-hidden_dim', help='the dimension of hidden layer', type=int, default=None)
parser.add_argument('-vocab_path', help='the vocabulary used for training and inference', type=str, default='./data/bert_vocab.txt')
parser.add_argument('-ipa_path', help='the ipa used for training and inference', type=str, default='./words/ipas.txt')
parser.add_argument('-output_path', help='the path to save models', type=str, default='./output')
parser.add_argument('-vocab_size', help='the size of the vocabulart', type=int, default=0)
parser.add_argument("-seed", help="Seed for randomized elements in the training", type=int, default=None)
parser.add_argument("-probs", help='probability of sampling', type=float, default=[0.16,0.16,0.16,0.16,0.36], nargs='+')
parser.add_argument("-probs_ipa", help='probability of sampling in ipa', type=float, default=[0.16,0.16,0.16,0.52], nargs='+')
parser.add_argument("-alpha", help="Used in computing loss", type=float, default=0.0)
parser.add_argument("-lamda", help="Used in interpolation", type=float, default=0.1)
parser.add_argument("-random_search", help="Whether to run random hyperparameter searching", action='store_true')
parser.add_argument("-ex_model_path", help="the path to save models", type=str, default=None)
parser.add_argument("-ex_ipa_path", help="the path to save ipas", type=str, default=None)
parser.add_argument("-ex_task", help="extrinsic task", type=str, default='nsmc', choices=['nsmc','klue_tc','klue_re','klue_dp','klue_ner','naver_ner'])
parser.add_argument("-ex_emb_path", help="the path to save embeddings", type=str, default=None)
                        

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

def main():
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.random_search:
        args.lamda = 0.06*np.random.rand() + 0.09
        print('A sampled lambda is %f' % (args.lamda))

    TOKENIZER = AutoTokenizer.from_pretrained("klue/bert-base")
    if args.use_ipa:
        word_to_ipa, ipa_set = load_ipa(args.ipa_path)
    else:
        word_to_ipa, ipa_set = None, []
    TOKENIZER = add_tokens(TOKENIZER)

    vocab_size = len(TOKENIZER)
    args.vocab_size = vocab_size
    args.ipa_vocab_size = len(ipa_set) + 4
    if not os.path.exists(args.output_path): os.makedirs(args.output_path)

    data_loader = loader[args.loader_type](args, TOKENIZER, word_to_ipa)
    train_iterator = data_loader(data_path=args.dataset)

    model = Producer[args.model_type](args)
    print(model)
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_num)
    print(args.input_type)
    print('use ipa: ', args.use_ipa)
    model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    criterion = loss_f[args.loss_type]()

    for e in range(args.epochs):
        epoch_loss = 0
        batch_num = 0
        s_time = time()

        for words, ipas, oririn_repre, aug_repre_ids, aug_repre_ids_ipa, mask, mask_ipa in train_iterator:
            model.train()
            optimizer.zero_grad()
            batch_num += 1

            if batch_num % 1000 == 0:
                print('sample = {b}, loss = {a}'.format(a=epoch_loss/batch_num, b=batch_num*args.batch_size))

            # get produced vectors
            oririn_repre = oririn_repre.cuda()
            aug_repre_ids = aug_repre_ids.cuda()
            mask = mask.cuda()
            if mask_ipa is not None: 
                aug_repre_ids_ipa = aug_repre_ids_ipa.cuda()
                mask_ipa = mask_ipa.cuda()
            aug_embeddings, _ = model(aug_repre_ids, mask, aug_repre_ids_ipa, mask_ipa)

            # calculate loss
            loss = criterion(oririn_repre, aug_embeddings)

            # backward
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print('[ lr rate] = {a}'.format(a=optimizer.state_dict()['param_groups'][0]['lr']))

        print('----------------------')
        print('this is the {a} epoch, loss = {b}'.format(a=e + 1, b=epoch_loss / len(train_iterator)))
        print('elapsed time = %.1f second' % (time()-s_time))

        if (e) % 1 == 0 and e >= (args.min_epoch_to_save-1):
            if args.use_ipa: 
                model_path = os.path.join(args.output_path, 'model_{a}_{b}_{c}.pt'.format(a=e+1,b=args.seed,c=args.lamda))
            else:
                model_path = os.path.join(args.output_path, 'model_{a}_{b}.pt'.format(a=e+1,b=args.seed))
            torch.save(model.state_dict(), model_path)
            scores_list = overall(args, model_path=model_path, tokenizer=TOKENIZER)

            output_all_eval_file = os.path.join(args.output_path, "eval_all_results.txt")
            with open(output_all_eval_file, "a") as all_writer:
                all_writer.write("eval results in epoch %d:\n" % (e+1))
                #all_writer.write("loss: %s\n" % (str(epoch_loss / len(train_iterator))))
                all_writer.write("prob: %s\n" % (str(args.probs)))
                all_writer.write("probs_ipa: %s\n" % (str(args.probs_ipa)))
                all_writer.write("# of params: %s\n" % (str(trainable_num)))
                all_writer.write("learning rate: %s\n" % (str(args.learning_rate)))
                all_writer.write("att_head_num: %s\n" % (str(args.att_head_num)))
                all_writer.write("encoder_layer: %s\n" % (str(args.encoder_layer)))
                all_writer.write("model_type: %s\n" % (args.model_type))
                all_writer.write("input_type %s\n" % (args.input_type))
                if args.use_ipa: 
                    all_writer.write("lamda %s\n" % (args.lamda))
                    all_writer.write("alpha %s\n" % (args.alpha))
                if args.seed is not None: all_writer.write("seed %s\n" % (args.seed))
                for s in scores_list:
                    all_writer.write("%s\n" % s)
                all_writer.write("-"*25+"\n")

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()