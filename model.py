import numpy as np
import math
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
import torch.nn.functional as F
from registry import register
from functools import partial
registry = {}
register = partial(register, registry=registry)


@register('rnn')
class RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.emb_dim
        self.encoder_layer = args.encoder_layer
        self.embedding = nn.Embedding(args.vocab_size, self.dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.rnn = nn.LSTM(input_size=self.dim, hidden_size=int(self.dim // 2), batch_first=True, num_layers=self.encoder_layer, bidirectional=True)
        self.linear1 = nn.Linear(self.dim, self.dim)
        self.linear2 = nn.Linear(self.dim, self.dim)
        self.activiation = nn.Tanh()
        self.dropout = nn.Dropout(p=args.drop_rate)

    def forward(self, x, mask=None):
        x_embed = self.embedding(x)
        shape = x_embed.size()

        mask = mask.squeeze().cpu().detach().numpy()
        mask = [np.sum(e != 0) for e in mask]

        # rnn pack
        packed = pack_padded_sequence(x_embed, mask, batch_first=True, enforce_sorted=False)
        encoder_outputs_packed, (h_last, c_last) = self.rnn(packed)
        rnn_output, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)

        output = list()
        for index in range(len(mask)):
            temp = rnn_output[index, mask[index]-1, :]
            output.append(temp)
        output = torch.reshape(torch.cat(output, dim=0), (shape[0], self.dim))

        output = self.dropout(output)
        output = self.linear1(output)
        output = self.activiation(output)
        output = self.dropout(output)
        output = self.linear2(output)
        return output


@register('rnn2')
class RNN2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.input_dim
        self.emb_dim = args.emb_dim
        self.hidden_dim = self.dim if args.hidden_dim is None else args.hidden_dim
        self.encoder_layer = args.encoder_layer
        self.lamda = args.lamda
        self.ipa = args.use_ipa

        self.embedding = nn.Embedding(args.vocab_size, self.dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.rnn = nn.LSTM(input_size=self.dim, hidden_size=int(self.hidden_dim // 2), batch_first=True, num_layers=self.encoder_layer, bidirectional=True)
        self.linear1 = nn.Linear(self.hidden_dim, 50)
        self.linear2 = nn.Linear(50, self.emb_dim)

        if self.ipa:
            d = args.drop_rate
            args.drop_rate = 0.1
            self.encoder_layer_ipa = args.encoder_layer_ipa
            self.embedding_ipa = nn.Embedding(args.ipa_vocab_size, self.emb_dim, padding_idx=0)
            self.embedding_ipa.weight.requires_grad = True

            # self.rnn_ipa = nn.LSTM(input_size=self.dim, hidden_size=int(self.hidden_dim // 2), batch_first=True, 
            #                        num_layers=self.encoder_layer_ipa, bidirectional=True)
            # self.linear1_ipa = nn.Linear(self.hidden_dim, self.hidden_dim)
            # self.linear2_ipa = nn.Linear(self.hidden_dim, self.emb_dim)

            self.position = PositionalEncoding(self.emb_dim)
            self.encoders_ipa = nn.ModuleList([SAM(args) for _ in range(self.encoder_layer_ipa)])
            self.sublayers_ipa = nn.ModuleList([SublayerConnection(args.drop_rate, self.emb_dim) for _ in range(self.encoder_layer_ipa)])
            args.drop_rate = d

        self.activiation = nn.Tanh()
        self.dropout = nn.Dropout(p=args.drop_rate)

    def forward(self, x, mask, x_ipa=None, mask_ipa=None):
        x_embed = self.embedding(x)
        shape = x_embed.size()

        mask = mask.squeeze().cpu().detach().numpy()
        mask = [np.sum(e != 0) for e in mask]

        # rnn pack
        packed = pack_padded_sequence(x_embed, mask, batch_first=True, enforce_sorted=False)
        encoder_outputs_packed, (h_last, c_last) = self.rnn(packed)
        rnn_output, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)

        output = list()
        for index in range(len(mask)):
            temp = rnn_output[index, mask[index]-1, :]
            output.append(temp)
        output = torch.reshape(torch.cat(output, dim=0), (shape[0], self.hidden_dim))

        output = self.dropout(output)
        output = self.linear1(output)
        output = self.activiation(output)
        output = self.dropout(output)
        output = self.linear2(output)

        if (self.ipa) and (mask_ipa is not None):
            # x_embed_ipa = self.embedding_ipa(x_ipa)
            # shape_ipa = x_embed_ipa.size()

            # mask_ipa = mask_ipa.squeeze().cpu().detach().numpy()
            # mask_ipa = [np.sum(e != 0) for e in mask_ipa]

            # # rnn pack
            # packed_ipa = pack_padded_sequence(x_embed_ipa, mask_ipa, batch_first=True, enforce_sorted=False)
            # encoder_outputs_packed_ipa, _ = self.rnn_ipa(packed_ipa)
            # rnn_output_ipa, _ = pad_packed_sequence(encoder_outputs_packed_ipa, batch_first=True)

            # output_ipa = list()
            # for index in range(len(mask_ipa)):
            #     temp_ipa = rnn_output_ipa[index, mask_ipa[index]-1, :]
            #     output_ipa.append(temp_ipa)
            # output_ipa = torch.reshape(torch.cat(output_ipa, dim=0), (shape_ipa[0], self.hidden_dim))

            # output_ipa = self.dropout(output_ipa)
            # output_ipa = self.linear1_ipa(output_ipa)
            # output_ipa = self.activiation(output_ipa)
            # output_ipa = self.dropout(output_ipa)
            # output_ipa = self.linear2_ipa(output_ipa)

            # final_output = (1.0-self.lamda)*output + self.lamda*output_ipa

            x_ipa = self.embedding_ipa(x_ipa) + self.position(x_ipa)
            for encoder, sublayer in zip(self.encoders_ipa, self.sublayers_ipa):
                x_ipa = sublayer(x_ipa, lambda z: encoder(z, mask_ipa))
            ipa_emb = x_ipa.masked_fill_(~mask_ipa, -float('inf')).max(dim=1)[0]
            final_output = (1.0-self.lamda)*output + self.lamda*ipa_emb

            return final_output, (output, ipa_emb)

        return output, None

@register('cnn')
class CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.emb_dim
        self.embedding = nn.Embedding(args.vocab_size, self.dim, padding_idx=0)
        self.embedding.weight.data.uniform_(-1, 1)
        self.windows = [3, 4, 5]
        self.convs = list()
        for window in self.windows:
            self.convs.append(nn.Conv2d(1, self.dim, (window, self.dim)).cuda())
        self.dropout = torch.nn.Dropout(p=args.drop_rate)
        self.proj = nn.Linear(len(self.windows)*self.dim, self.dim)

    def forward(self, x, x_lens):
        embeddings = self.embedding(x)
        embeddings = embeddings.unsqueeze(1)

        poolings = list()
        for conv in self.convs:
            conv_f = conv(embeddings)
            conv_f = F.relu(conv_f.squeeze(3))
            pooling = conv_f.max(dim=-1)[0]
            poolings.append(pooling)
        poolings = self.dropout(torch.cat(poolings, dim=-1))
        new_embed = self.proj(poolings)
        return new_embed


@register('attention')
class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.dim = args.emb_dim
        self.encoder_layer = args.encoder_layer
        self.embedding = nn.Embedding(args.vocab_size, self.dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.encoders = nn.ModuleList([OriAttention(args) for _ in range(self.encoder_layer)])
        self.sublayer = SublayerConnection(args.drop_rate, self.dim)
        self.ipa = (args.input_type == 'ipa')
        if self.ipa:
            self.std = args.std
            self.noise_ratio = args.noise_ratio

    def forward(self, x, mask, mask_ipa=None, changed=None):
        x = self.embedding(x)
        if (self.ipa) and (mask_ipa is not None):   
            mask_ipa = (torch.rand(*mask_ipa.size()) <= self.noise_ratio).to(x.device) * mask_ipa * changed
            num_normal = torch.sum(mask_ipa).item()
            noise = self.std * torch.normal(0,1,size=(num_normal, x.size(-1))).to(x.device)
            x[mask_ipa] = x[mask_ipa] + noise
        for i, encoder in enumerate(self.encoders):
            x = self.sublayer(x, lambda x: encoder(x, mask))
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]


class OriAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.head = args.att_head_num
        self.dropout = nn.Dropout(p=args.drop_rate)

    def forward(self, x, mask):
        q = x
        k = x
        v = x
        q, k, v = (split_last(a, (self.head, -1)).transpose(1, 2)
                   for a in [q, k, v])

        scores = torch.matmul(q, k.transpose(2, 3)) / (k.size(-1) ** 0.25)
        mask = torch.matmul(mask.float(), mask.transpose(1, 2).float()).bool()
        mask = mask.unsqueeze(1)
        mask = mask.repeat([1, self.head, 1, 1])
        scores.masked_fill_(~mask, -1e7)
        scores = F.softmax(scores, dim=2)
        scores = scores.transpose(2, 3)
        v_ = torch.matmul(scores, v)
        v_ = v_.transpose(1, 2).contiguous()
        v_ = merge_last(v_, 2)
        return v_


def l2norm(x):
    return x / x.norm(p=2, dim=1, keepdim=True)


@register('pam')
class Pamela(nn.Module):
    def __init__(self, args):
        super(Pamela, self).__init__()
        self.dim = args.emb_dim
        self.encoder_layer = args.encoder_layer
        self.embedding = nn.Embedding(args.vocab_size, self.dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.head = args.att_head_num
        self.encoders = nn.ModuleList([Pamelaformer(args) for _ in range(self.encoder_layer)])
        self.sublayer = SublayerConnection(args.drop_rate, self.dim)

    def forward(self, x, mask):
        x = self.embedding(x)
        shape = list(x.size())
        position = PositionalEncoding(shape[-1], shape[-2])
        pos_att = position(x)

        for i, encoder in enumerate(self.encoders):
            x = self.sublayer(x, lambda x: encoder(x, mask, pos_att))

        x = x.masked_fill_(~mask, 0).sum(dim=1)
        return l2norm(x)


class Pamelaformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attention = SAM(args)
        self.pos_attention = PAM(args)
        dim = args.emb_dim
        proj_dim = args.emb_dim
        self.merge = args.merge
        if self.merge:
            proj_dim = 2 * args.emb_dim
        self.projection = nn.Sequential(
            nn.Linear(proj_dim, dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=args.drop_rate)

    def forward(self, x, mask, position, merge=True):
        att = self.self_attention(x, mask)
        pos = self.pos_attention(x, mask, position)
        if self.merge:
            c = self.projection(torch.cat([att, pos], dim=-1))
        else:
            c = self.projection(pos)
        return c


class PAM(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.emb_dim
        self.head = args.att_head_num
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=args.drop_rate)

    def forward(self, x, mask, pos):
        q = pos
        k = pos
        v = x
        q, k, v = (split_last(a, (self.head, -1)).transpose(1, 2) for a in [q, k, v])
        scores = torch.matmul(q, k.transpose(2, 3)) / (k.size(-1) ** 0.25)
        mask = torch.matmul(mask.float(), mask.transpose(1, 2).float()).bool()
        mask = mask.unsqueeze(1)
        mask = mask.repeat([1, self.head, 1, 1])
        scores.masked_fill_(~mask, -1e7)

        scores = F.softmax(scores, dim=2)
        scores = scores.transpose(2, 3)
        v_ = torch.matmul(scores, v)
        v_ = v_.transpose(1, 2).contiguous()
        v_ = merge_last(v_, 2)
        v_ = self.projection(v_)
        return v_


class SAM(nn.Module):
    def __init__(self, args):
        super().__init__()
        attention_dim = 32
        self.head = args.att_head_num
        hidden_size = args.emb_dim
        self.projectionq = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.ReLU()
        )
        self.projectionk = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=args.drop_rate)

    def forward(self, x, mask):
        q = self.projectionq(x)
        k = self.projectionk(x)
        v = x
        q, k, v = (split_last(a, (self.head, -1)).transpose(1, 2)
                   for a in [q, k, v])

        scores = torch.matmul(q, k.transpose(2, 3)) / (k.size(-1) ** 0.25)
        mask = torch.matmul(mask.float(), mask.transpose(1, 2).float()).bool()
        mask = mask.unsqueeze(1)
        mask = mask.repeat([1, self.head, 1, 1])
        scores.masked_fill_(~mask, -1e7)
        scores = F.softmax(scores, dim=2)
        scores = scores.transpose(2, 3)
        v_ = torch.matmul(scores, v)
        v_ = v_.transpose(1, 2).contiguous()
        v_ = merge_last(v_, 2)

        return v_

@register('self_attention')
class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.emb_dim
        self.encoder_layer = args.encoder_layer
        self.embedding = nn.Embedding(args.vocab_size, self.dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.encoders = nn.ModuleList([SAM(args) for _ in range(self.encoder_layer)])
        self.sublayer = SublayerConnection(args.drop_rate, self.dim)
        self.ipa = (args.input_type == 'ipa')
        if self.ipa:
            self.std = args.std
            self.noise_ratio = args.noise_ratio

    def forward(self, x, mask, x_ipa=None, mask_ipa=None):
        x = self.embedding(x)
        for i, encoder in enumerate(self.encoders):
            x = self.sublayer(x, lambda x: encoder(x, mask))
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0], None

@register('self_attention_2')
class SelfAttention2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.emb_dim
        self.encoder_layer = args.encoder_layer
        self.lamda = args.lamda
        self.position = PositionalEncoding(self.dim)
        self.embedding = nn.Embedding(args.vocab_size, self.dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.encoders = nn.ModuleList([SAM(args) for _ in range(self.encoder_layer)])
        self.sublayers = nn.ModuleList([SublayerConnection(args.drop_rate, self.dim) for _ in range(self.encoder_layer)])
        self.ipa = args.use_ipa
        if self.ipa:
            self.embedding_ipa = nn.Embedding(args.ipa_vocab_size, self.dim, padding_idx=0)
            self.embedding_ipa.weight.requires_grad = True
            self.encoders_ipa = nn.ModuleList([SAM(args) for _ in range(self.encoder_layer)])
            self.sublayers_ipa = nn.ModuleList([SublayerConnection(args.drop_rate, self.dim) for _ in range(self.encoder_layer)])

    def forward(self, x, mask, x_ipa=None, mask_ipa=None):
        x = self.embedding(x) + self.position(x)
        for encoder, sublayer in zip(self.encoders, self.sublayers):
            x = sublayer(x, lambda z: encoder(z, mask))

        if (self.ipa) and (mask_ipa is not None):  
            # x_ipa (batch_size, max_length)
            x_ipa = self.embedding_ipa(x_ipa) + self.position(x_ipa)
            # x_ipa (batch_size, max_length, dim)
            # mask_ipa (batch_size, max_length, 1)
            for encoder, sublayer in zip(self.encoders_ipa, self.sublayers_ipa):
                x_ipa = sublayer(x_ipa, lambda z: encoder(z, mask_ipa))
            # x_ipa (batch_size, max_length, dim)
            emb = x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]
            ipa_emb = x_ipa.masked_fill_(~mask_ipa, -float('inf')).max(dim=1)[0]
            final_emb = (1.0-self.lamda)*emb + self.lamda*ipa_emb
            # final_emb (batch_size, dim)
            return final_emb, (emb, ipa_emb)
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0], None


@register('self_attention_3')
class SelfAttention3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.emb_dim
        self.encoder_layer = args.encoder_layer
        self.position = PositionalEncoding(self.dim)
        self.embedding = nn.Embedding(args.vocab_size, self.dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.encoders = nn.ModuleList([SAM(args) for _ in range(self.encoder_layer)])
        self.sublayers = nn.ModuleList([SublayerConnection(args.drop_rate, self.dim) for _ in range(self.encoder_layer)])
        self.ipa = (args.input_type == 'ipa')
        if self.ipa:
            self.embedding_ipa = nn.Embedding(args.ipa_vocab_size, self.dim, padding_idx=0)
            self.embedding_ipa.weight.requires_grad = True
            self.encoders_ipa = nn.ModuleList([SAM(args) for _ in range(self.encoder_layer)])
            self.sublayers_ipa = nn.ModuleList([SublayerConnection(args.drop_rate, self.dim) for _ in range(self.encoder_layer)])
            self.mlp = MLPMOE()

    def forward(self, step, x, mask, x_ipa=None, mask_ipa=None):
        x = self.embedding(x) + self.position(x)
        for encoder, sublayer in zip(self.encoders, self.sublayers):
            x = sublayer(x, lambda z: encoder(z, mask))

        if (self.ipa) and (mask_ipa is not None):   
            x_ipa = self.embedding_ipa(x_ipa) + self.position(x_ipa)
            for encoder, sublayer in zip(self.encoders_ipa, self.sublayers_ipa):
                x_ipa = sublayer(x_ipa, lambda z: encoder(z, mask_ipa))

            emb = x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]
            ipa_emb = x_ipa.masked_fill_(~mask_ipa, -float('inf')).max(dim=1)[0]
            if step >= 3000:
                weight = self.mlp(emb.clone().detach(), ipa_emb.clone().detach()).exp().unsqueeze(-1)
            else:
                weight = 0.5 * torch.ones(emb.size(0),2).unsqueeze(-1).to(emb.device)
            emb_fin = torch.sum(weight * torch.stack((emb,ipa_emb), dim=1), dim=1)
            return emb_fin, weight[:,0,0].mean()
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0], torch.tensor(1.0)


@register('self_attention_4')
class SelfAttention4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.emb_dim
        self.encoder_layer = args.encoder_layer
        self.lamda = args.lamda
        self.position = PositionalEncoding(self.dim)
        self.embedding = nn.Embedding(args.vocab_size, self.dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.encoders = nn.ModuleList([SAM(args) for _ in range(self.encoder_layer)])
        self.sublayers = nn.ModuleList([SublayerConnection(args.drop_rate, self.dim) for _ in range(self.encoder_layer)])
        self.ipa = (args.input_type == 'ipa')
        if self.ipa:
            self.embedding_ipa = nn.Embedding(args.ipa_vocab_size, self.dim, padding_idx=0)
            self.embedding_ipa.weight.requires_grad = True
            self.encoders_ipa = nn.ModuleList([SAM(args) for _ in range(self.encoder_layer)])
            self.sublayers_ipa = nn.ModuleList([SublayerConnection(args.drop_rate, self.dim) for _ in range(self.encoder_layer)])

    def forward(self, x, mask, x_ipa=None, mask_ipa=None):
        x = self.embedding(x) + self.position(x)
        for encoder, sublayer in zip(self.encoders, self.sublayers):
            x = sublayer(x, lambda z: encoder(z, mask))

        if (self.ipa) and (mask_ipa is not None):   
            x_ipa = self.embedding_ipa(x_ipa) + self.position(x_ipa)
            for encoder, sublayer in zip(self.encoders_ipa, self.sublayers_ipa):
                x_ipa = sublayer(x_ipa, lambda z: encoder(z, mask_ipa))

            emb = x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]
            ipa_emb = x_ipa.masked_fill_(~mask_ipa, -float('inf')).max(dim=1)[0]
            final_emb = (1.0-self.lamda)*emb + self.lamda*ipa_emb
            return final_emb, emb, ipa_emb
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, dropout, dim):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(self.norm(sublayer(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def cal_fixed_pos_att(max_len, window_size):
    win = (window_size - 1) // 2
    weight = float(1 / window_size)
    attn_dict = dict()
    for sen_len in range(1, max_len+1):
        attn = np.eye(sen_len)
        if sen_len < window_size:
            attn_dict[sen_len] = attn
            continue
        for i in range(sen_len):
            attn[i, i-win:i+win+1] = weight
        attn[0, 0:win+1] = weight
        attn_dict[sen_len] = torch.FloatTensor(attn)

    return attn_dict


class PositionalAttCached(nn.Module):
    def __init__(self, d_model, pos_attns, max_len=5000):
        super(PositionalAttCached, self).__init__()
        # Compute the positional encodings once in log space.
        self.d_model = d_model
        self.pos_attns = pos_attns
        self.max_len = max_len

    def forward(self, x):
        shape = list(x.size())
        pos_attn = self.pos_attns[shape[1]]
        p_e = Variable(pos_attn, requires_grad=False).cuda()
        p_e = p_e.repeat([shape[0], 1, 1])
        return p_e

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        position = position * 1
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        shape = list(x.size())
        p_e = Variable(self.pe[:, :x.size(1)], requires_grad=False).cuda()
        p_e = p_e.repeat([shape[0], 1, 1])
        return p_e

class MLPMOE(nn.Module):
    def __init__(self,
                 hidden_units=128,
                 nlayers=3,
                 dropout=0.2,
                 dim=300,
                 ipa_dim=300,
                 activation='relu'):
        super().__init__()

        models = [nn.Linear(ipa_dim, hidden_units), nn.Dropout(p=dropout)]
        if activation == 'relu':
            models.append(nn.ReLU())
        elif activation == 'linear':
            pass
        else:
            raise ValueError(f'activation {activation} not supported')

        for _ in range(nlayers-1):
            models.extend([nn.Linear(hidden_units, hidden_units), nn.Dropout(p=dropout)])
            if activation == 'relu':
                models.append(nn.ReLU())
            elif activation == 'linear':
                pass
            else:
                raise ValueError(f'activation {activation} not supported')

        models.append(nn.Linear(hidden_units, 2))
        models.append(nn.LogSoftmax(dim=-1))

        self.model = nn.Sequential(*models)

    def forward(self, emb, ipa_emb):
        return self.model(torch.cat([ipa_emb], -1))
