import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import os
import sklearn.metrics
from typing import Any
import json

parser = argparse.ArgumentParser(description='CNN Text Classification')
parser.add_argument('--word_embed_dim', type=int, default=300)
parser.add_argument('--cnn_filter_num', type=int, default=100)
parser.add_argument('--cnn_kernel_size', type=list, default=[3,4,5])
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--pretrain_embed_path', default='./embeddings/kold/love_kor_jm.emb')
parser.add_argument('--pretrain_embed_path_ipa', default='./embeddings/kold/love_kor_ipa.emb')
parser.add_argument('--output_path', default='extrinsic/kold/output')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lamda', type=float, default=0.098839020547)
parser.add_argument('--c', type=float, default=[1/3,1/3,1/3], nargs=3)
parser.add_argument('--vocab_path', default='extrinsic/kold/data/words.txt')
parser.add_argument('--train_path', default='extrinsic/kold/data/kold_v1_split_train.json')
parser.add_argument('--dev_path', default='extrinsic/kold/data/kold_v1_split_val.json')
parser.add_argument('--test_path', default='extrinsic/kold/data/kold_v1_split_test.json')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--postfix', type=str, default='')

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 dropout, num_class, pre_embedding=None, pre_embedding_ipa=None, PAD_IDX=0, lamda=0.0):
        super().__init__()

        # word embedding
        self.embedding_word = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_IDX)
        self.embedding_word.weight.data.copy_(torch.from_numpy(pre_embedding))
        self.embedding_word.weight.requires_grad = False

        # ipa embedding
        self.embedding_ipa = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_IDX)
        self.embedding_ipa.weight.data.copy_(torch.from_numpy(pre_embedding_ipa))
        self.embedding_ipa.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, num_class)

        self.dropout = nn.Dropout(dropout)
        self.lamda = lamda

    def forward(self, word):

        embedded = self.embedding_word(word)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        embedded_ipa = self.embedding_ipa(word)
        embedded_ipa = embedded_ipa.unsqueeze(1)
        conved_ipa = [F.relu(conv(embedded_ipa)).squeeze(3) for conv in self.convs]
        pooled_ipa = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_ipa]
        cat_ipa = self.dropout(torch.cat(pooled_ipa, dim=1))

        embedded_mixup = (1.0-self.lamda)*embedded + self.lamda*embedded_ipa
        conved_mixup = [F.relu(conv(embedded_mixup)).squeeze(3) for conv in self.convs]
        pooled_mixup = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_mixup]
        cat_mixup = self.dropout(torch.cat(pooled_mixup, dim=1))

        return self.fc(cat), self.fc(cat_ipa), self.fc(cat_mixup)

def kold_binary_f1(preds: np.ndarray, targets: np.ndarray) -> Any:
    return sklearn.metrics.f1_score(targets, preds, average="binary")

def train(model, iterator, optimizer, criterion, device, c):
    epoch_loss = 0

    model.train()

    all_preds, all_labels = [],[]
    for words, labels in iterator:
        optimizer.zero_grad()
        words = words.to(device)
        labels = labels.to(device)

        predictions, predictions_ipa, predictions_mixup = model(words)

        loss = criterion(predictions_ipa, labels)
        #loss = criterion(predictions_ipa, labels) + criterion(predictions, labels)
        #loss = criterion(predictions_ipa, labels) + criterion(predictions_mixup, labels)
        #loss = criterion(predictions_mixup, labels) + criterion(predictions, labels)
        #loss = c[0]*criterion(predictions_mixup, labels) + c[1]*criterion(predictions_ipa, labels) + c[2]*criterion(predictions, labels)
        #loss = criterion(predictions_mixup, labels) + criterion(predictions_ipa, labels) + criterion(predictions, labels)

        predictions_all = predictions_ipa
        #predictions_all = c[0]*predictions_mixup + c[1]*predictions_ipa + c[2]*predictions
        all_preds.append(torch.max(predictions_all, dim=-1).indices.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)

    return epoch_loss / len(iterator), kold_binary_f1(all_preds, all_labels)


def evaluate(model, iterator, criterion, device, c):
    epoch_loss = 0

    model.eval()

    all_preds, all_labels = [],[]
    with torch.no_grad():
        for words, labels in iterator:
            words = words.to(device)
            labels = labels.to(device)

            predictions, predictions_ipa, predictions_mixup = model(words)

            loss = criterion(predictions_ipa, labels)
            #loss = criterion(predictions_ipa, labels) + criterion(predictions, labels)
            #loss = criterion(predictions_ipa, labels) + criterion(predictions_mixup, labels)
            #loss = criterion(predictions_mixup, labels) + criterion(predictions, labels)
            #loss = criterion(predictions_mixup, labels) + criterion(predictions_ipa, labels) + criterion(predictions, labels)

            predictions_all = predictions_ipa
            #predictions_all = c[0]*predictions_mixup + c[1]*predictions_ipa + c[2]*predictions
            all_preds.append(torch.max(predictions_all, dim=-1).indices.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            epoch_loss += loss.item()

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)

    return epoch_loss / len(iterator), kold_binary_f1(all_preds, all_labels)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def gen_vocabulary(doc_files, vocab_file, max_size=30000):
    '''
    generate a word list given an input corpus
    :param data_file:
    :param vocab_file:
    :return:
    '''
    word_freq = dict()
    all_sentences = list()
    for file in doc_files:
        with open(file, encoding='utf8')as f:
            sentences = [s.split('\t')[0] for s in f.readlines()]
            all_sentences.extend(sentences)
    print('data_size = {a}'.format(a=len(all_sentences)))
    for sentence in all_sentences:
        tokens = sentence.strip().split(' ')
        for token in tokens:
            if token not in word_freq:
                word_freq[token] = 0
            word_freq[token] += 1

    sorted_word_freq = sorted(word_freq.items(), key=lambda e:e[1], reverse=True)[:max_size]
    sorted_word_freq = [k for k, v in sorted_word_freq]

    with open(vocab_file, 'w', encoding='utf8')as f:
        f.write('\n'.join(sorted_word_freq))

    print('done! The size of vocabulary is {a}.'.format(a=len(sorted_word_freq)))


def load_vocabulary(path, PAD_IDX=0):
    vocab_dict = dict()
    word_list = []
    for index, line in enumerate(open(path, encoding='utf8')):
        row = line.strip()
        w_id = index + 1
        text = row
        vocab_dict[text] = w_id
        word_list.append(text)
    vocab_dict['<pad>'] = 0
    vocab_dict['<sep>'] = len(vocab_dict)
    vocab_dict['<unk>'] = len(vocab_dict)
    word_list.append('<sep>')
    word_list.append('<unk>')
    word_list.insert(PAD_IDX, '<pad>')
    print('load vocabulary, size = {a}'.format(a=len(word_list)))
    return vocab_dict, word_list


def pad_sentence(sentence, max_len, PAD_IDX=0):
    if len(sentence) > max_len:
        return sentence[:max_len]
    else:
        for i in range(max_len - len(sentence)):
            sentence.append(PAD_IDX)
        return sentence

def load_sent_by_id(data_file, vocab_dict):
    all_sents = list()
    labels = list()
    with open(data_file) as f:
        examples = json.load(f)

    for ex in examples:
        sentence, label = ex['comment'], ex['OFF']
        words = sentence.strip().split(' ')
        word_ids = [vocab_dict.get(w, len(vocab_dict)-1) for w in words]
        all_sents.append(word_ids)
        labels.append(int(label))

    print('load data, size = {a}'.format(a=len(all_sents)))
    return {'sentence': all_sents, 'label':labels}

def load_all_embeddings(path, emb_size):
    word_embed_dict = dict()
    with open(path)as f:
        line = f.readline()
        while line:
            row = line.strip().split(' ')
            word_embed_dict[row[0]] = row[1:emb_size+1]
            line = f.readline()
    print('load word embedding, size = {a}'.format(a=len(word_embed_dict)))
    return word_embed_dict


def get_emb_weights(embeddings, word_list, embed_size):
    #torch_weights = torch.zeros(len(word_list), embed_size)
    torch_weights = []
    miss_set = set()
    for index, word in enumerate(word_list):
        if word not in embeddings:
            miss_set.add(word)
            weight = np.zeros(embed_size)
        else:
            weight = [float(v) for v in embeddings[word]]
        #weight[index, :] = torch.from_numpy(np.array(weight))
        torch_weights.append(weight)

    print('in total, {a} words do not have embeddings'.format(a=len(miss_set)))
    #print(miss_set)
    return np.array(torch_weights)


def random_emb_value():
    return random.uniform(0, 0)


def process_embedding(embed_path, vocab_list, emb_size, out_path):
    result = []
    w_l = ''
    print('word list size = {a}'.format(a=len(vocab_list)))
    word_embed_dict = load_all_embeddings(embed_path, emb_size)
    print('loaded pre-trained embedding size = {a} '.format(a=len(word_embed_dict)))
    missing_set = set()
    cnt = 0
    for index, word in enumerate(vocab_list):
        if index % 10 == 0:print('process {a} words ...'.format(a=index))
        if word in word_embed_dict:
            emd = word_embed_dict[word]
        else:
            emd = [str(round(random_emb_value(), 5)) for _ in range(emb_size)]
            cnt += 1
            missing_set.add(word)
        result.append(emd)
        w_l += word + ' ' + ' '.join(emd) + '\n'
    print('word size = {a}, {b} words have not embedding value'.format(a=len(vocab_list), b=cnt))
    np.save(out_path, np.array(result))
    print(missing_set)

    with open(out_path, 'w', encoding='utf8')as f:
        f.write(w_l)


class TextData(Dataset):
    def __init__(self, data):
        self.sentence = data['sentence']
        self.label = data['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.sentence[idx], self.label[idx]


def collate_fn(batch_data, PAD_IDX=0):
    sentence, label = list(zip(*batch_data))
    max_len = max([len(sent) for sent in sentence])
    sentence = [sent+[PAD_IDX]*(max_len-len(sent)) for sent in sentence]

    sentence = torch.LongTensor(sentence)
    label = torch.LongTensor(label)
    return sentence, label


def load_pretrained_emb(emb_path, dim=300, words=None):
    vectors = dict()
    cnt = 0
    for line in open(emb_path, encoding='utf8'):
        cnt += 1
        #if cnt >=1000:return vectors
        row = line.strip().split(' ')
        if len(row) != dim+1:continue
        word = row[0]
        if words is not None and word not in words:continue
        vec = np.array(row[1:], dtype=str)
        if word not in vectors:
            vectors[word] = vec
    return vectors


def fixed_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_kold(args):
    word_dict, word_list = load_vocabulary(args.vocab_path)
    word_num = len(word_list)  # len(TEXT.vocab)
    pre_embeddings = load_all_embeddings(args.pretrain_embed_path, args.word_embed_dim)
    pre_embeddings_ipa = load_all_embeddings(args.pretrain_embed_path_ipa, args.word_embed_dim)
    print('embedding word size = {a}'.format(a=len(pre_embeddings)))
    pre_embeddings = get_emb_weights(pre_embeddings, word_list, args.word_embed_dim)
    pre_embeddings_ipa = get_emb_weights(pre_embeddings_ipa, word_list, args.word_embed_dim)
    #pre_embeddings_pool = (1.0-args.lamda)*pre_embeddings + args.lamda*pre_embeddings_ipa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.output_path, 'model.pt')

    train_data = load_sent_by_id(args.train_path, word_dict)
    dev_data = load_sent_by_id(args.dev_path, word_dict)
    test_data = load_sent_by_id(args.test_path, word_dict)

    test_data_70_nat = load_sent_by_id('./extrinsic/kold/data/kold_v1_split_test_natural_70.json', word_dict)
    test_data_pool = [test_data_70_nat]

    # print('#'*10)
    # print(len(test_data['label']))
    # print(len(test_data_70_nat['label']))

    train_data = TextData(train_data)
    dev_data = TextData(dev_data)
    test_data = TextData(test_data)
    test_data_oov = [TextData(data) for data in test_data_pool]
    
    train_iterator = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_iterator = DataLoader(dataset=dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_iterator = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_iterator_oov = [DataLoader(dataset=data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) for data in test_data_oov]

    model = CNN(word_num, args.word_embed_dim, args.cnn_filter_num, args.cnn_kernel_size, args.dropout, args.num_class, 
                pre_embedding=pre_embeddings, pre_embedding_ipa=pre_embeddings_ipa, lamda=args.lamda).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_valid_f1 = 0.0
    print('epoch = {a}'.format(a=args.epochs))
    for e in range(args.epochs):
        start_time = time.time()
        train_loss, train_f1 = train(model, train_iterator, optimizer, criterion, device, args.c)
        valid_loss, valid_f1 = evaluate(model, dev_iterator, criterion, device, args.c)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save(model.state_dict(), model_path)

        print(f'Epoch: {e + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train F1: {train_f1 * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1 * 100:.2f}%')

    model.load_state_dict(torch.load(model_path))
    f1 = predict(args, model, test_iterator, test_iterator_oov, criterion, device, model_path)
    return f1


def predict(args, model, test_iterator, test_iterator_oov, criterion, device, model_path):
    model.load_state_dict(torch.load(model_path))
    test_loss, test_f1 = evaluate(model, test_iterator, criterion, device, args.c)
    print('\t-------------------------------------------------------------')
    print(f'\tTest Loss: {test_loss:.3f} | Test F1: {test_f1 * 100:.2f}%')
    print('\t-------------------------------------------------------------')

    ratios = ['70%_']
    f1_list = [test_f1]
    print('\t------------------------------------------------------------------------')
    for ratio, iterator in zip(ratios, test_iterator_oov):
        loss, f1 = evaluate(model, iterator, criterion, device, args.c)
        print(f'\tOOV Ratio: {ratio} | Test Loss: {loss:.3f} | Test F1: {f1 * 100:.2f}%')
        f1_list.append(f1)
    print('\t------------------------------------------------------------------------')

    output_all_eval_file = os.path.join(args.output_path, "eval_all_results_kold.txt")
    with open(output_all_eval_file, "a") as all_writer:
        all_writer.write("eval results:\n")
        all_writer.write("%s\n" % (args.postfix))
        for f1 in f1_list:
            all_writer.write("%s\n" % (f"{f1:.4f}"))
        all_writer.write("-"*25+"\n")

    return test_f1


if __name__ == '__main__':
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args = parser.parse_args()
    fixed_seed(args.seed)
    if not os.path.exists(args.output_path): os.makedirs(args.output_path)

    args.word_embed_dim = 300
    run_kold(args)
