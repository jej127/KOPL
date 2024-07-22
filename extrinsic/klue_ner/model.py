import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from crf import CRF
from charcnn import CharCNN


class NamedEntityRecog(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, word_hidden_dim, alphabet_size, char_embedding_dim, char_hidden_dim,
                 feature_extractor, tag_num, dropout, pretrain_embed=None, use_char=False, use_crf=False, use_gpu=False):
        super(NamedEntityRecog, self).__init__()
        self.use_crf = use_crf
        self.use_char = use_char
        self.drop = nn.Dropout(dropout)
        self.input_dim = word_embed_dim
        self.feature_extractor = feature_extractor

        self.embeds = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        if pretrain_embed is not None:
            self.embeds.weight.data.copy_(torch.from_numpy(pretrain_embed))
            self.embeds.weight.requires_grad = False
        else:
            self.embeds.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, word_embed_dim)))

        if self.use_char:
            self.input_dim += char_hidden_dim
            self.char_feature = CharCNN(alphabet_size, char_embedding_dim, char_hidden_dim, dropout)

        if feature_extractor == 'lstm':
            self.lstm = nn.LSTM(self.input_dim, word_hidden_dim, batch_first=True, bidirectional=True)
        elif feature_extractor == 'linear':
            self.linear = nn.Sequential(
                  nn.Linear(self.input_dim, word_hidden_dim*2),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(word_hidden_dim*2, word_hidden_dim*2),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(word_hidden_dim * 2, word_hidden_dim * 2),

                )

        else:
            self.word2cnn = nn.Linear(self.input_dim, word_hidden_dim*2)
            self.cnn_list = list()
            for _ in range(4):
                self.cnn_list.append(nn.Conv1d(word_hidden_dim*2, word_hidden_dim*2, kernel_size=3, padding=1))
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(dropout))
                self.cnn_list.append(nn.BatchNorm1d(word_hidden_dim*2))
            self.cnn = nn.Sequential(*self.cnn_list)

        if self.use_crf:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num + 2)
            self.crf = CRF(tag_num, use_gpu)
        else:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(1, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embeding = self.embeds(word_inputs)
        word_list = [word_embeding]
        if self.use_char:
            char_features = self.char_feature(char_inputs).contiguous().view(batch_size, seq_len, -1)
            word_list.append(char_features)
        word_embeding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embeding)
        if self.feature_extractor == 'lstm':
            packed_words = pack_padded_sequence(word_represents, word_seq_lengths, True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            lstm_out = lstm_out.transpose(0, 1)
            feature_out = self.drop(lstm_out)
        elif self.feature_extractor == 'linear':
            feature_out = self.linear(word_represents)
        else:
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represents)).transpose(2, 1).contiguous()
            feature_out = self.cnn(word_in).transpose(1, 2).contiguous()

        feature_out = self.hidden2tag(feature_out)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label)
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            total_loss = loss_function(feature_out, batch_label.contiguous().view(batch_size * seq_len))
        return total_loss

    def forward(self, word_inputs, word_seq_lengths, char_inputs, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embeding = self.embeds(word_inputs)
        word_list = [word_embeding]
        if self.use_char:
            char_features = self.char_feature(char_inputs).contiguous().view(batch_size, seq_len, -1)
            word_list.append(char_features)
        word_embeding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embeding)
        if self.feature_extractor == 'lstm':
            packed_words = pack_padded_sequence(word_represents, word_seq_lengths, True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            lstm_out = lstm_out.transpose(0, 1)
            feature_out = self.drop(lstm_out)
        elif self.feature_extractor == 'linear':
            feature_out = self.linear(word_represents)
        else:
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represents)).transpose(2, 1).contiguous()
            feature_out = self.cnn(word_in).transpose(1, 2).contiguous()

        feature_out = self.hidden2tag(feature_out)

        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(feature_out, mask)
        else:
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(feature_out, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
        return tag_seq
    


class NamedEntityRecog_woChar(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, word_hidden_dim, num_layers,
                 feature_extractor, tag_num, dropout, pretrain_embed=None, use_crf=False, use_gpu=False):
        super(NamedEntityRecog_woChar, self).__init__()
        self.use_crf = use_crf
        self.drop = nn.Dropout(dropout)
        self.input_dim = word_embed_dim
        self.num_layers = num_layers
        self.feature_extractor = feature_extractor

        self.embeds = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        if pretrain_embed is not None:
            self.embeds.weight.data.copy_(torch.from_numpy(pretrain_embed))
            self.embeds.weight.requires_grad = False
        else:
            self.embeds.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, word_embed_dim)))

        if feature_extractor == 'lstm':
            self.lstm = nn.LSTM(self.input_dim, word_hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        elif feature_extractor == 'linear':
            self.linear = nn.Sequential(
                  nn.Linear(self.input_dim, word_hidden_dim*2),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(word_hidden_dim*2, word_hidden_dim*2),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(word_hidden_dim * 2, word_hidden_dim * 2),

                )

        else:
            self.word2cnn = nn.Linear(self.input_dim, word_hidden_dim*2)
            self.cnn_list = list()
            for _ in range(4):
                self.cnn_list.append(nn.Conv1d(word_hidden_dim*2, word_hidden_dim*2, kernel_size=3, padding=1))
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(dropout))
                self.cnn_list.append(nn.BatchNorm1d(word_hidden_dim*2))
            self.cnn = nn.Sequential(*self.cnn_list)

        if self.use_crf:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num + 2)
            self.crf = CRF(tag_num, use_gpu)
        else:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(1, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embeding = self.embeds(word_inputs)
        word_list = [word_embeding]
        word_embeding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embeding)
        if self.feature_extractor == 'lstm':
            packed_words = pack_padded_sequence(word_represents, word_seq_lengths, True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            lstm_out = lstm_out.transpose(0, 1)
            feature_out = self.drop(lstm_out)
        elif self.feature_extractor == 'linear':
            feature_out = self.linear(word_represents)
        else:
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represents)).transpose(2, 1).contiguous()
            feature_out = self.cnn(word_in).transpose(1, 2).contiguous()

        feature_out = self.hidden2tag(feature_out)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label)
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            total_loss = loss_function(feature_out, batch_label.contiguous().view(batch_size * seq_len))
        return total_loss

    def forward(self, word_inputs, word_seq_lengths, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embeding = self.embeds(word_inputs)
        word_list = [word_embeding]
        word_embeding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embeding)
        if self.feature_extractor == 'lstm':
            packed_words = pack_padded_sequence(word_represents, word_seq_lengths, True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            lstm_out = lstm_out.transpose(0, 1)
            feature_out = self.drop(lstm_out)
        elif self.feature_extractor == 'linear':
            feature_out = self.linear(word_represents)
        else:
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represents)).transpose(2, 1).contiguous()
            feature_out = self.cnn(word_in).transpose(1, 2).contiguous()

        feature_out = self.hidden2tag(feature_out)

        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(feature_out, mask)
        else:
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(feature_out, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
        return tag_seq
    
class NamedEntityRecog_woChar_ours(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, word_hidden_dim, num_layers, feature_extractor, tag_num, dropout, lamda, c,
                 pretrain_embed=None, pretrain_embed_ipa=None, use_crf=False, use_gpu=False):
        super(NamedEntityRecog_woChar_ours, self).__init__()
        self.use_crf = use_crf
        self.drop = nn.Dropout(dropout)
        self.input_dim = word_embed_dim
        self.feature_extractor = feature_extractor
        self.num_layers = num_layers
        self.lamda = lamda
        self.c = c

        self.embeds = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        if pretrain_embed is not None:
            self.embeds.weight.data.copy_(torch.from_numpy(pretrain_embed))
            self.embeds.weight.requires_grad = False
        else:
            self.embeds.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, word_embed_dim)))

        self.embeds_ipa = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        if pretrain_embed_ipa is not None:
            self.embeds_ipa.weight.data.copy_(torch.from_numpy(pretrain_embed_ipa))
            self.embeds_ipa.weight.requires_grad = False
        else:
            self.embeds_ipa.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, word_embed_dim)))

        if feature_extractor == 'lstm':
            self.lstm = nn.LSTM(self.input_dim, word_hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        elif feature_extractor == 'linear':
            self.linear = nn.Sequential(
                  nn.Linear(self.input_dim, word_hidden_dim*2),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(word_hidden_dim*2, word_hidden_dim*2),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(word_hidden_dim * 2, word_hidden_dim * 2),

                )

        else:
            self.word2cnn = nn.Linear(self.input_dim, word_hidden_dim*2)
            self.cnn_list = list()
            for _ in range(4):
                self.cnn_list.append(nn.Conv1d(word_hidden_dim*2, word_hidden_dim*2, kernel_size=3, padding=1))
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(dropout))
                self.cnn_list.append(nn.BatchNorm1d(word_hidden_dim*2))
            self.cnn = nn.Sequential(*self.cnn_list)

        if self.use_crf:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num + 2)
            self.crf = CRF(tag_num, use_gpu)
        else:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(1, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb
    
    def compute_feature_out(self, embeding, word_inputs, word_seq_lengths):
        word_list = [embeding]
        embeding = torch.cat(word_list, 2)
        represents = self.drop(embeding)
        if self.feature_extractor == 'lstm':
            packed_words = pack_padded_sequence(represents, word_seq_lengths, True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            lstm_out = lstm_out.transpose(0, 1)
            feature_out = self.drop(lstm_out)
        elif self.feature_extractor == 'linear':
            feature_out = self.linear(represents)
        else:
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(represents)).transpose(2, 1).contiguous()
            feature_out = self.cnn(word_in).transpose(1, 2).contiguous()

        feature_out = self.hidden2tag(feature_out)
        return feature_out

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embedding = self.embeds(word_inputs)
        ipa_embedding = self.embeds_ipa(word_inputs)
        mixup_embedding = (1.0-self.lamda)*word_embedding + self.lamda*ipa_embedding

        feature_out = self.compute_feature_out(word_embedding, word_inputs, word_seq_lengths)
        feature_out_ipa = self.compute_feature_out(ipa_embedding, word_inputs, word_seq_lengths)
        feature_out_mixup = self.compute_feature_out(mixup_embedding, word_inputs, word_seq_lengths)

        if self.use_crf:
            # total_loss = self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label)
            # total_loss = self.crf.neg_log_likelihood_loss(feature_out_ipa, mask, batch_label)
            #total_loss = self.crf.neg_log_likelihood_loss(feature_out_mixup, mask, batch_label)
            # total_loss = self.crf.neg_log_likelihood_loss(feature_out_ipa, mask, batch_label) +\
            #              self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label)
            # total_loss = self.crf.neg_log_likelihood_loss(feature_out_mixup, mask, batch_label) +\
            #              self.crf.neg_log_likelihood_loss(feature_out_ipa, mask, batch_label)
            # total_loss = self.crf.neg_log_likelihood_loss(feature_out_mixup, mask, batch_label) +\
            #              self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label)
            total_loss = self.crf.neg_log_likelihood_loss(feature_out_mixup, mask, batch_label) +\
                         self.crf.neg_log_likelihood_loss(feature_out_ipa, mask, batch_label) +\
                         self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label)
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            feature_out_ipa = feature_out_ipa.contiguous().view(batch_size * seq_len, -1)
            feature_out_mixup = feature_out_mixup.contiguous().view(batch_size * seq_len, -1)

            # total_loss = loss_function(feature_out, batch_label.contiguous().view(batch_size * seq_len)) +\
            #              loss_function(feature_out_ipa, batch_label.contiguous().view(batch_size * seq_len)) +\
            #              loss_function(feature_out_mixup, batch_label.contiguous().view(batch_size * seq_len))
            total_loss = self.c[0]*loss_function(feature_out, batch_label.contiguous().view(batch_size * seq_len)) +\
                         self.c[1]*loss_function(feature_out_ipa, batch_label.contiguous().view(batch_size * seq_len)) +\
                         self.c[2]*loss_function(feature_out_mixup, batch_label.contiguous().view(batch_size * seq_len))
            #total_loss = loss_function(feature_out_ipa, batch_label.contiguous().view(batch_size * seq_len))
        return total_loss

    def forward(self, word_inputs, word_seq_lengths, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embedding = self.embeds(word_inputs)
        ipa_embedding = self.embeds_ipa(word_inputs)
        mixup_embedding = (1.0-self.lamda)*word_embedding + self.lamda*ipa_embedding

        feature_out = self.compute_feature_out(word_embedding, word_inputs, word_seq_lengths)
        feature_out_ipa = self.compute_feature_out(ipa_embedding, word_inputs, word_seq_lengths)
        feature_out_mixup = self.compute_feature_out(mixup_embedding, word_inputs, word_seq_lengths)

        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(self.c[0]*feature_out_mixup + self.c[1]*feature_out_ipa + self.c[2]*feature_out, mask)
        else:
            feature_out = (feature_out + feature_out_ipa + feature_out_mixup).contiguous().view(batch_size * seq_len, -1)
            #feature_out = feature_out_ipa.contiguous().view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(feature_out, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
        return tag_seq
    

class NamedEntityRecog_woChar_ours2(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, word_hidden_dim, feature_extractor, tag_num, dropout, lamda, c,
                 pretrain_embed=None, pretrain_embed_ipa=None, use_crf=False, use_gpu=False):
        super(NamedEntityRecog_woChar_ours2, self).__init__()
        self.use_crf = use_crf
        self.drop = nn.Dropout(dropout)
        self.input_dim = word_embed_dim
        self.feature_extractor = feature_extractor
        self.lamda = lamda
        self.c = c

        self.embeds = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        if pretrain_embed is not None:
            self.embeds.weight.data.copy_(torch.from_numpy(pretrain_embed))
            self.embeds.weight.requires_grad = False
        else:
            self.embeds.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, word_embed_dim)))

        self.embeds_ipa = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        if pretrain_embed_ipa is not None:
            self.embeds_ipa.weight.data.copy_(torch.from_numpy(pretrain_embed_ipa))
            self.embeds_ipa.weight.requires_grad = False
        else:
            self.embeds_ipa.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, word_embed_dim)))

        if feature_extractor == 'lstm':
            self.lstm = nn.LSTM(self.input_dim, word_hidden_dim, batch_first=True, bidirectional=True)
        elif feature_extractor == 'linear':
            self.linear = nn.Sequential(
                  nn.Linear(self.input_dim, word_hidden_dim*2),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(word_hidden_dim*2, word_hidden_dim*2),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(word_hidden_dim * 2, word_hidden_dim * 2),

                )

        else:
            self.word2cnn = nn.Linear(self.input_dim, word_hidden_dim*2)
            self.cnn_list = list()
            for _ in range(4):
                self.cnn_list.append(nn.Conv1d(word_hidden_dim*2, word_hidden_dim*2, kernel_size=3, padding=1))
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(dropout))
                self.cnn_list.append(nn.BatchNorm1d(word_hidden_dim*2))
            self.cnn = nn.Sequential(*self.cnn_list)

        if self.use_crf:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num + 2)
            self.crf = CRF(tag_num, use_gpu)
        else:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(1, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb
    
    def compute_feature_out(self, embeding, word_inputs, word_seq_lengths):
        word_list = [embeding]
        embeding = torch.cat(word_list, 2)
        represents = self.drop(embeding)
        if self.feature_extractor == 'lstm':
            packed_words = pack_padded_sequence(represents, word_seq_lengths, True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            lstm_out = lstm_out.transpose(0, 1)
            feature_out = self.drop(lstm_out)
        elif self.feature_extractor == 'linear':
            feature_out = self.linear(represents)
        else:
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(represents)).transpose(2, 1).contiguous()
            feature_out = self.cnn(word_in).transpose(1, 2).contiguous()

        feature_out = self.hidden2tag(feature_out)
        return feature_out

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embedding = self.embeds(word_inputs)
        ipa_embedding = self.embeds_ipa(word_inputs)
        mixup_embedding = (1.0-self.lamda)*word_embedding + self.lamda*ipa_embedding

        feature_out = self.compute_feature_out(word_embedding, word_inputs, word_seq_lengths)
        feature_out_ipa = self.compute_feature_out(ipa_embedding, word_inputs, word_seq_lengths)
        feature_out_mixup = self.compute_feature_out(mixup_embedding, word_inputs, word_seq_lengths)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label) +\
                         self.crf.neg_log_likelihood_loss(feature_out_ipa + 0.5*feature_out, mask, batch_label) +\
                         self.crf.neg_log_likelihood_loss(feature_out_mixup, mask, batch_label)
            # total_loss = self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label) +\
            #              self.crf.neg_log_likelihood_loss(0.5*feature_out_ipa + 0.5*feature_out, mask, batch_label) +\
            #              self.crf.neg_log_likelihood_loss(feature_out_mixup, mask, batch_label)
            #total_loss = self.crf.neg_log_likelihood_loss(feature_out_ipa, mask, batch_label)
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            feature_out_ipa = feature_out_ipa.contiguous().view(batch_size * seq_len, -1)
            feature_out_mixup = feature_out_mixup.contiguous().view(batch_size * seq_len, -1)

            # total_loss = loss_function(feature_out, batch_label.contiguous().view(batch_size * seq_len)) +\
            #              loss_function(feature_out_ipa, batch_label.contiguous().view(batch_size * seq_len)) +\
            #              loss_function(feature_out_mixup, batch_label.contiguous().view(batch_size * seq_len))
            total_loss = loss_function(feature_out_ipa, batch_label.contiguous().view(batch_size * seq_len))
        return total_loss

    def forward(self, word_inputs, word_seq_lengths, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embedding = self.embeds(word_inputs)
        ipa_embedding = self.embeds_ipa(word_inputs)
        mixup_embedding = (1.0-self.lamda)*word_embedding + self.lamda*ipa_embedding

        feature_out = self.compute_feature_out(word_embedding, word_inputs, word_seq_lengths)
        feature_out_ipa = self.compute_feature_out(ipa_embedding, word_inputs, word_seq_lengths)
        feature_out_mixup = self.compute_feature_out(mixup_embedding, word_inputs, word_seq_lengths)

        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(feature_out + feature_out_ipa + feature_out_mixup, mask)
            #scores, tag_seq = self.crf._viterbi_decode(feature_out_ipa, mask)
        else:
            #feature_out = (feature_out + feature_out_ipa + feature_out_mixup).contiguous().view(batch_size * seq_len, -1)
            feature_out = feature_out_ipa.contiguous().view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(feature_out, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
        return tag_seq