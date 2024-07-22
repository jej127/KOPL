import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import os
from utils import create_examples
import sklearn.metrics
from typing import List, Any, Optional
from collections import Counter


dep_label_list = ["NP", "NP_AJT", "VP", "NP_SBJ", "VP_MOD", "NP_OBJ", "AP", "NP_CNJ", "NP_MOD", "VNP", "DP", "VP_AJT",
                  "VNP_MOD", "NP_CMP", "VP_SBJ", "VP_CMP", "VP_OBJ", "VNP_CMP", "AP_MOD", "X_AJT", "VP_CNJ", "VNP_AJT", "IP", "X", "X_SBJ",
                  "VNP_OBJ", "VNP_SBJ", "X_OBJ", "AP_AJT", "L", "X_MOD", "X_CNJ", "VNP_CNJ", "X_CMP", "AP_CMP", "AP_SBJ", "R", "NP_SVJ",]

pos_label_list = ["NNG", "NNP", "NNB", "NP", "NR", "VV", "VA", "VX", "VCP", "VCN", "MMA", "MMD", "MMN", "MAG", "MAJ", "JC", "IC", "JKS",
                  "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "EP", "EF", "EC", "ETN", "ETM", "XPN", "XSN", "XSV", "XSA", "XR", "SF",
                  "SP", "SS", "SE", "SO", "SL", "SH", "SW", "SN", "NA",]

pos_label_map = {label: i for i, label in enumerate(pos_label_list)}
dep_label_map = {label: i for i, label in enumerate(dep_label_list)}

parser = argparse.ArgumentParser(description='CNN Text Classification')
parser.add_argument('--word_embed_dim', type=int, default=300)
parser.add_argument('--arc_space', type=int, default=200)
parser.add_argument('--type_space', type=int, default=100)
parser.add_argument('--pos_dim', type=int, default=100)
parser.add_argument("--encoder_layers", type=int, default=1)
parser.add_argument("--decoder_layers", type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--pretrain_embed_path', default='/mnt/oov/klue_dp/love_kor_jm.emb')
parser.add_argument('--pretrain_embed_path_ipa', default='/mnt/oov/klue_dp/love_kor_ipa.emb')
parser.add_argument('--output_path', default='extrinsic/klue_dp/output')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lamda', type=float, default=0.098839020547)
parser.add_argument('--c', type=float, default=[1/3,1/3,1/3], nargs=3)
parser.add_argument('--vocab_path', default='extrinsic/klue_dp/data/words.txt')
parser.add_argument('--train_path', default='extrinsic/klue_dp/data/klue-dp-v1.1_split_train.tsv')
parser.add_argument('--dev_path', default='extrinsic/klue_dp/data/klue-dp-v1.1_split_val.tsv')
parser.add_argument('--test_path', default='extrinsic/klue_dp/data/klue-dp-v1.1_dev.tsv')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--no_pos', action="store_true")
parser.add_argument('--postfix', type=str, default='')

class DPResult:
    """Result object for DataParallel"""

    def __init__(self, heads: torch.Tensor, types: torch.Tensor) -> None:
        self.heads = heads
        self.types = types

class BiAttention(nn.Module):
    def __init__(  # type: ignore[no-untyped-def]
        self, input_size_encoder: int, input_size_decoder: int, num_labels: int, biaffine: bool = True, **kwargs
    ) -> None:
        super(BiAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter("U", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.W_d)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(
        self,
        input_d: torch.Tensor,
        input_e: torch.Tensor,
        mask_d: Optional[torch.Tensor] = None,
        mask_e: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input_d.size(0) == input_e.size(0)
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        if self.biaffine:
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))
            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output

class BiLinear(nn.Module):
    def __init__(self, left_features: int, right_features: int, out_features: int):
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left: torch.Tensor, input_right: torch.Tensor) -> torch.Tensor:
        left_size = input_left.size()
        right_size = input_right.size()
        assert left_size[:-1] == right_size[:-1], "batch size of left and right inputs mis-match: (%s, %s)" % (
            left_size[:-1],
            right_size[:-1],
        )
        batch = int(np.prod(left_size[:-1]))

        input_left = input_left.contiguous().view(batch, self.left_features)
        input_right = input_right.contiguous().view(batch, self.right_features)

        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)
        return output.view(left_size[:-1] + (self.out_features,))

class DP_LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, arc_space, type_space, pos_dim, no_pos, encoder_layers, decoder_layers, dropout,
                 pre_embedding=None, pre_embedding_ipa=None, PAD_IDX=0, lamda=0.0):
        super().__init__()

        # word embedding
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
        self.embedding.weight.requires_grad = False

        # embedding for root
        self.root_emb = nn.parameter.Parameter(torch.Tensor(1,hidden_size))
        nn.init.xavier_uniform_(self.root_emb)

        # ipa embedding
        self.embedding_ipa = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=PAD_IDX)
        self.embedding_ipa.weight.data.copy_(torch.from_numpy(pre_embedding_ipa))
        self.embedding_ipa.weight.requires_grad = False

        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.arc_space = arc_space
        self.type_space = type_space

        self.n_pos_labels = len(pos_label_list)
        self.n_dp_labels = len(dep_label_list)

        if no_pos:
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(self.n_pos_labels + 1, pos_dim)

        enc_dim = self.input_size
        if self.pos_embedding is not None: enc_dim += pos_dim

        self.encoder = nn.LSTM(
            enc_dim, self.hidden_size, encoder_layers, batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.decoder = nn.LSTM(
            self.hidden_size, self.hidden_size, decoder_layers, batch_first=True, dropout=dropout
        )

        self.dropout = nn.Dropout1d(p=dropout)

        self.src_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.hx_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.arc_c = nn.Linear(self.hidden_size * 2, self.arc_space)
        self.type_c = nn.Linear(self.hidden_size * 2, self.type_space)
        self.arc_h = nn.Linear(self.hidden_size, self.arc_space)
        self.type_h = nn.Linear(self.hidden_size, self.type_space)

        self.attention = BiAttention(self.arc_space, self.arc_space, 1)
        self.bilinear = BiLinear(self.type_space, self.type_space, self.n_dp_labels)
        self.lamda = lamda

    def make_prediction(self, embedding_output, pos_ids, head_ids, mask_e, mask_d, batch_index, sent_len, is_training):
        outputs = torch.cat([self.root_emb.expand(embedding_output.size(0),-1).unsqueeze(1), embedding_output], dim=1)
        # outputs = [batch_size, max_word_length+1, hidden_size]

        if self.pos_embedding is not None:
            pos_outputs = self.pos_embedding(pos_ids)
            pos_outputs = self.dropout(pos_outputs)
            outputs = torch.cat([outputs, pos_outputs], dim=2)
            # outputs = [batch_size, max_word_length+1, hidden_size+pos_dim]

        # encoder
        sent_len_extend = [i + 1 for i in sent_len]
        packed_outputs = pack_padded_sequence(outputs, sent_len_extend, batch_first=True, enforce_sorted=False)
        encoder_outputs, hn = self.encoder(packed_outputs)
        encoder_outputs, outputs_len = pad_packed_sequence(encoder_outputs, batch_first=True)
        encoder_outputs = self.dropout(encoder_outputs.transpose(1, 2)).transpose(1, 2)  # apply dropout for last layer
        hn = self._transform_decoder_init_state(hn)
        # encoder_outputs = [batch_size, max_word_length+1, 2*hidden_size] (2 for bidirectional)

        # decoder
        src_encoding = F.elu(self.src_dense(encoder_outputs[:,1:]))
        # src_encoding = [batch_size, max_word_length, hidden_size]
        #sent_len = [i - 1 for i in sent_len]
        packed_outputs = pack_padded_sequence(src_encoding, sent_len, batch_first=True, enforce_sorted=False)
        decoder_outputs, _ = self.decoder(packed_outputs, hn)
        decoder_outputs, outputs_len = pad_packed_sequence(decoder_outputs, batch_first=True)
        decoder_outputs = self.dropout(decoder_outputs.transpose(1,2)).transpose(1,2)  # apply dropout for last layer
        # decoder_outputs = [batch_size, max_word_length, hidden_size]

        # compute output for arc and type
        arc_c = F.elu(self.arc_c(encoder_outputs))
        type_c = F.elu(self.type_c(encoder_outputs))

        arc_h = F.elu(self.arc_h(decoder_outputs))
        type_h = F.elu(self.type_h(decoder_outputs))
        # arc_c = [batch_size, max_word_length+1, arc_space]
        # type_c = [batch_size, max_word_length+1, type_space]
        # arc_h = [batch_size, max_word_length, arc_space]
        # type_h = [batch_size, max_word_length, type_space]

        out_arc = self.attention(arc_h, arc_c, mask_d=mask_d, mask_e=mask_e).squeeze(dim=1)
        # out_arc = [batch_size, max_word_length, max_word_length+1]

        # use predicted head_ids when validation step
        if not is_training:
            head_ids = torch.argmax(out_arc, dim=2)

        type_c = type_c[batch_index, head_ids.data.t()].transpose(0, 1).contiguous()
        out_type = self.bilinear(type_h, type_c)
        # out_type = [batch_size, max_word_length, n_dp_labels]

        return out_arc, out_type

    def forward(self, pos_ids, head_ids, mask_e, mask_d, batch_index, sent_len, word, is_training=True):
        
        outputs = self.embedding(word)
        # outputs = [batch_size, max_word_length, hidden_size]
        outputs_ipa = self.embedding_ipa(word)
        # outputs_ipa = [batch_size, max_word_length, hidden_size]
        outputs_mixup = (1.0-self.lamda)*outputs + self.lamda*outputs_ipa
        # outputs_mixup = [batch_size, max_word_length, hidden_size]

        out_arc, out_type = self.make_prediction(outputs, pos_ids, head_ids, mask_e, mask_d, batch_index, sent_len, is_training)
        out_arc_ipa, out_type_ipa = self.make_prediction(outputs_ipa, pos_ids, head_ids, mask_e, mask_d, batch_index, sent_len, is_training)
        out_arc_mixup, out_type_mixup = self.make_prediction(outputs_mixup, pos_ids, head_ids, mask_e, mask_d, batch_index, sent_len, is_training)

        return out_arc, out_type, out_arc_ipa, out_type_ipa, out_arc_mixup, out_type_mixup


    def _transform_decoder_init_state(self, hn: torch.Tensor) -> torch.Tensor:
        hn, cn = hn
        cn = cn[-2:]  # take the last layer
        _, batch_size, hidden_size = cn.size()
        cn = cn.transpose(0, 1).contiguous()
        cn = cn.view(batch_size, 1, 2 * hidden_size).transpose(0, 1)
        cn = self.hx_dense(cn)
        if self.decoder.num_layers > 1:
            cn = torch.cat(
                [
                    cn,
                    torch.autograd.Variable(cn.data.new(self.decoder.num_layers - 1, batch_size, hidden_size).zero_()),
                ],
                dim=0,
            )
        hn = torch.tanh(cn)
        hn = (hn, cn)
        return hn


def compute_loss(out_arc, out_type, mask_d, mask_e, batch_index, head_index, head_ids, type_ids, minus_inf=-1e8):
    minus_mask_d = (1 - mask_d) * minus_inf
    minus_mask_e = (1 - mask_e) * minus_inf
    out_arc = out_arc + minus_mask_d.unsqueeze(2) + minus_mask_e.unsqueeze(1)

    loss_arc = F.log_softmax(out_arc, dim=2)
    loss_type = F.log_softmax(out_type, dim=2)

    loss_arc = loss_arc * mask_d.unsqueeze(2) * mask_e.unsqueeze(1)
    loss_type = loss_type * mask_d.unsqueeze(2)
    num = mask_d.sum()

    loss_arc = loss_arc[batch_index, head_index, head_ids.data.t()].transpose(0, 1)
    loss_type = loss_type[batch_index, head_index, type_ids.data.t()].transpose(0, 1)
    loss_arc = -loss_arc.sum() / num
    loss_type = -loss_type.sum() / num
    loss = loss_arc + loss_type
    return loss

def klue_dp_las_macro_f1(preds: List[DPResult], labels: List[DPResult]) -> Any:
    """KLUE-DP LAS macro f1. (UAS : head correct / LAS : head + type correct)"""
    # UAS : head correct / LAS : head + type correct
    head_preds = list()
    head_labels = list()
    type_preds = list()
    type_labels = list()
    for pred, label in zip(preds, labels):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
        type_preds += pred.types.cpu().flatten().tolist()
        type_labels += label.types.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    type_preds = np.array(type_preds)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_preds = np.delete(type_preds, index)
    type_labels = np.delete(type_labels, index)

    # classify others label as -3
    others_idx = 15
    for i, (pred, label) in enumerate(zip(type_preds, type_labels)):
        if pred >= others_idx:
            type_preds[i] = -3
        if label >= others_idx:
            type_labels[i] = -3

    # pad wrong UAS
    PAD = -2
    uas_correct = np.equal(head_preds, head_labels)
    uas_incorrect = np.nonzero(np.invert(uas_correct))
    for idx in uas_incorrect:
        type_preds[idx] = PAD
    return sklearn.metrics.f1_score(type_labels.tolist(), type_preds.tolist(), average="macro")

def train(model, iterator, optimizer, device, c):
    epoch_loss = 0
    model.train()
    all_preds, all_labels = [],[]
    for sentence, head_ids, type_ids, pos_ids, mask_e, mask_d, max_len, sent_len in iterator:
        optimizer.zero_grad()
        sentence = sentence.to(device)
        head_ids = head_ids.to(device)
        type_ids = type_ids.to(device)
        if pos_ids is not None: pos_ids = pos_ids.to(device)
        mask_e = mask_e.to(device)
        mask_d = mask_d.to(device)

        batch_size = head_ids.size(0)
        batch_index = torch.arange(0, int(batch_size)).long().to(device)
        head_index = (torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch_size).long().to(device))

        # forward
        out_arc, out_type, out_arc_ipa, out_type_ipa, out_arc_mixup, out_type_mixup = \
            model(pos_ids, head_ids, mask_e, mask_d, batch_index, sent_len, sentence)

        # predict arc and its type
        heads = torch.argmax(c[0]*out_arc_mixup + c[1]*out_arc_ipa + c[2]*out_arc, dim=2)
        types = torch.argmax(c[0]*out_type_mixup + c[1]*out_type_ipa + c[2]*out_type, dim=2)

        preds = DPResult(heads.cpu().detach(), types.cpu().detach())
        labels = DPResult(head_ids.cpu().detach(), type_ids.cpu().detach())

        loss_word = compute_loss(out_arc, out_type, mask_d, mask_e, batch_index, head_index, head_ids, type_ids)
        loss_ipa = compute_loss(out_arc_ipa, out_type_ipa, mask_d, mask_e, batch_index, head_index, head_ids, type_ids)
        loss_mixup = compute_loss(out_arc_mixup, out_type_mixup, mask_d, mask_e, batch_index, head_index, head_ids, type_ids)
        #c = [0.2,0.3,0.5]
        loss = c[0]*loss_mixup + c[1]*loss_ipa + c[2]*loss_word
        #loss = loss_mixup + loss_ipa + loss_word

        loss.backward()
        optimizer.step()

        all_preds.append(preds)
        all_labels.append(labels)

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), klue_dp_las_macro_f1(all_preds, all_labels)

def evaluate(model, iterator, device, c):
    epoch_loss = 0
    model.eval()
    all_preds, all_labels = [],[]
    with torch.no_grad():
        for sentence, head_ids, type_ids, pos_ids, mask_e, mask_d, max_len, sent_len in iterator:
            sentence = sentence.to(device)
            head_ids = head_ids.to(device)
            type_ids = type_ids.to(device)
            if pos_ids is not None: pos_ids = pos_ids.to(device)
            mask_e = mask_e.to(device)
            mask_d = mask_d.to(device)

            batch_size = head_ids.size(0)
            batch_index = torch.arange(0, int(batch_size)).long().to(device)
            head_index = (torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch_size).long().to(device))

            out_arc, out_type, out_arc_ipa, out_type_ipa, out_arc_mixup, out_type_mixup =\
                  model(pos_ids, head_ids, mask_e, mask_d, batch_index, sent_len, sentence, is_training=False)

            # predict arc and its type
            heads = torch.argmax(c[0]*out_arc_mixup + c[1]*out_arc_ipa + c[2]*out_arc, dim=2)
            types = torch.argmax(c[0]*out_type_mixup + c[1]*out_type_ipa + c[2]*out_type, dim=2)

            preds = DPResult(heads, types)
            labels = DPResult(head_ids, type_ids)

            loss_word = compute_loss(out_arc, out_type, mask_d, mask_e, batch_index, head_index, head_ids, type_ids)
            loss_ipa = compute_loss(out_arc_ipa, out_type_ipa, mask_d, mask_e, batch_index, head_index, head_ids, type_ids)
            loss_mixup = compute_loss(out_arc_mixup, out_type_mixup, mask_d, mask_e, batch_index, head_index, head_ids, type_ids)
            loss = c[0]*loss_mixup + c[1]*loss_ipa + c[2]*loss_word
            #loss = loss_mixup + loss_ipa + loss_word

            all_preds.append(preds)
            all_labels.append(labels)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), klue_dp_las_macro_f1(all_preds, all_labels)


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
    examples = create_examples(data_file)
    c = Counter([ex.sent_id for ex in examples])

    # mask_d, mask_e, head_ids
    start_idx = 0
    all_input_ids, all_head_ids, all_type_ids, all_pos_ids, sent_lens = list(), list(), list(), list(), list()
    for i in range(len(c)):
        input_ids, pos_ids, head_ids, dep_ids = [],[],[],[]
        word_length = c[i]
        for j in range(start_idx, start_idx+word_length):
            word = examples[j].token.strip().replace(' ','')
            word_id = vocab_dict.get(word, len(vocab_dict)-1)

            input_ids.append(word_id)
            head_ids.append(int(examples[j].head))
            dep_ids.append(dep_label_map[examples[j].dep])
            pos_ids.append(pos_label_map[examples[j].pos.split("+")[-1]]) # 맨 뒤 pos정보만 사용

        all_input_ids.append(input_ids)
        all_head_ids.append(head_ids)
        all_type_ids.append(dep_ids)
        all_pos_ids.append(pos_ids)
        sent_lens.append(word_length)
        start_idx += word_length

    print('load data, size = {a}'.format(a=len(all_input_ids)))
    return {'input_ids': all_input_ids, 
            'head_ids': all_head_ids, 
            'type_ids': all_type_ids, 
            'pos_ids': all_pos_ids, 
            'sent_lens': sent_lens}

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
        self.all_input_ids = data['input_ids']
        self.all_head_ids = data['head_ids']
        self.all_type_ids = data['type_ids']
        self.all_pos_ids = data['pos_ids']
        self.sent_lens = data['sent_lens']

    def __len__(self):
        return len(self.sent_lens)

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.all_head_ids[idx], self.all_type_ids[idx], self.all_pos_ids[idx], self.sent_lens[idx]


def collate_fn(batch_data, PAD_IDX=0):
    all_input_ids, all_head_ids, all_type_ids, all_pos_ids, sent_lens = list(zip(*batch_data))
    
    max_len = max([len(sent) for sent in all_input_ids])
    assert max_len == max(sent_lens)
    input_ids = [sent + [PAD_IDX]*(max_len-len(sent)) for sent in all_input_ids]
    head_ids = [heads + [-1]*(max_len-len(heads)) for heads in all_head_ids]
    type_ids = [types + [-1]*(max_len-len(types)) for types in all_type_ids]
    mask_e = [[1]*(word_len+1) + [0]*(max_len-word_len) for word_len in sent_lens]

    # head_ids = torch.zeros(batch_size, max_len).long()
    # type_ids = torch.zeros(batch_size, max_len).long()
    # pos_ids = torch.zeros(batch_size, max_len + 1).long()
    # mask_e = torch.zeros(batch_size, max_len + 1).long()
    # for batch_id in range(batch_size):
    #     word_length = sent_lens[batch_id]
    #     head_ids[batch_id] = torch.LongTensor(all_heads[batch_id] + [-1]*(max_len-word_length))
    #     type_ids[batch_id] = torch.LongTensor(all_types[batch_id] + [-1]*(max_len-word_length))
    #     mask_e[batch_id] = torch.LongTensor([1]*(word_length+1) + [0]*(max_len-word_length))

    #     pos_ids[batch_id][0] = torch.tensor(pos_padding_idx)
    #     pos_ids[batch_id][1:] = torch.LongTensor(all_poses[batch_id] + [pos_padding_idx]*(max_len-word_length))

    input_ids = torch.LongTensor(input_ids)
    head_ids = torch.LongTensor(head_ids)
    type_ids = torch.LongTensor(type_ids)
    mask_e = torch.LongTensor(mask_e)
    mask_d = mask_e[:, 1:]

    if args.no_pos:
        pos_ids = None
    else:
        pos_padding_idx = len(pos_label_list)
        pos_ids = [[pos_padding_idx] + poses + [pos_padding_idx]*(max_len-len(poses)) for poses in all_pos_ids]
        pos_ids = torch.LongTensor(pos_ids)

    return input_ids, head_ids, type_ids, pos_ids, mask_e, mask_d, max_len, sent_lens


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
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_klue_dp(args):
    word_dict, word_list = load_vocabulary(args.vocab_path)
    word_num = len(word_list)  # len(TEXT.vocab)
    pre_embeddings = load_all_embeddings(args.pretrain_embed_path, args.word_embed_dim)
    pre_embeddings_ipa = load_all_embeddings(args.pretrain_embed_path_ipa, args.word_embed_dim)
    print('embedding word size = {a}'.format(a=len(pre_embeddings)))
    pre_embeddings = get_emb_weights(pre_embeddings, word_list, args.word_embed_dim)
    pre_embeddings_ipa = get_emb_weights(pre_embeddings_ipa, word_list, args.word_embed_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.output_path, 'model.pt')

    train_data = load_sent_by_id(args.train_path, word_dict)
    dev_data = load_sent_by_id(args.dev_path, word_dict)
    test_data = load_sent_by_id(args.test_path, word_dict)

    test_data_30_nat = load_sent_by_id('./extrinsic/klue_dp/data/klue-dp-v1.1_dev_natural_30.tsv', word_dict)
    test_data_pool = [test_data_30_nat]

    train_data = TextData(train_data)
    dev_data = TextData(dev_data)
    test_data = TextData(test_data)
    test_data_oov = [TextData(data) for data in test_data_pool]
    
    train_iterator = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_iterator = DataLoader(dataset=dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_iterator = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_iterator_oov = [DataLoader(dataset=data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) for data in test_data_oov]

    model = DP_LSTM(word_num, args.word_embed_dim, args.arc_space, args.type_space, args.pos_dim, args.no_pos, args.encoder_layers,
                    args.decoder_layers, args.dropout, pre_embedding=pre_embeddings, pre_embedding_ipa=pre_embeddings_ipa, lamda=args.lamda).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_valid_f1 = 0.0
    print('epoch = {a}'.format(a=args.epochs))
    for e in range(args.epochs):
        start_time = time.time()
        train_loss, train_f1 = train(model, train_iterator, optimizer, device, args.c)
        valid_loss, valid_f1 = evaluate(model, dev_iterator, device, args.c)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save(model.state_dict(), model_path)

        print(f'Epoch: {e + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train F1: {train_f1 * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1 * 100:.2f}%')

    model.load_state_dict(torch.load(model_path))
    f1 = predict(args, model, test_iterator, test_iterator_oov, device, model_path)
    return f1


def predict(args, model, test_iterator, test_iterator_oov, device, model_path):
    model.load_state_dict(torch.load(model_path))
    test_loss, test_f1 = evaluate(model, test_iterator, device, args.c)
    print('\t-------------------------------------------------------------')
    print(f'\tTest Loss: {test_loss:.3f} | Test F1: {test_f1 * 100:.2f}%')
    print('\t-------------------------------------------------------------')

    ratios = ['30%_']
    f1_list = [test_f1]
    print('\t------------------------------------------------------------------------')
    for ratio, iterator in zip(ratios, test_iterator_oov):
        loss, f1 = evaluate(model, iterator, device, args.c)
        print(f'\tOOV Ratio: {ratio} | Test Loss: {loss:.3f} | Test F1: {f1 * 100:.2f}%')
        f1_list.append(f1)
    print('\t------------------------------------------------------------------------')

    output_all_eval_file = os.path.join(args.output_path, "eval_all_results_klue_dp.txt")
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
    run_klue_dp(args)
