import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])           # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])           # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).to(device)        # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                  # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.to(self.device))


def get_attn_pad_mask(seq_q, seq_k, trg_pad_idx):                                # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(trg_pad_idx).unsqueeze(1)                   # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)           # 扩展成多维度


def get_attn_subsequence_mask(seq):                                 # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)            # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()    # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):                              # Q: [batch_size, n_heads, len_q, d_k]
                                                                        # K: [batch_size, n_heads, len_k, d_k]
                                                                        # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)    # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)                            # 如果时停用词P就等于负无穷，softmax之后为0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                 # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.device = device
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):    # input_Q: [batch_size, len_q, d_model]
                                                                # input_K: [batch_size, len_k, d_model]
                                                                # input_V: [batch_size, len_v(=len_k), d_model]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        d_k = int(input_Q.size(-1) / self.n_heads)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)    # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)    # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, d_k).transpose(1,
                                                                           2)       # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)                                # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)             # context: [batch_size, n_heads, len_q, d_v]
                                                                                    # attn: [batch_size, n_heads, len_q, l。en_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * d_k)                    # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                   # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, ffn_hidden, device):
        super(PoswiseFeedForwardNet, self).__init__()
        self.device = device
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, ffn_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model, bias=False))

    def forward(self, inputs):                                  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_heads, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model,
                                                n_heads=n_heads,
                                                device=device
                                                )                   # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model,
                                            ffn_hidden=ffn_hidden,
                                            device=device
                                            )                      # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):              # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V            # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                                    # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)                     # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_heads, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model,
                                                n_heads=n_heads,
                                                device=device
                                                )       # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model,
                                             ffn_hidden=ffn_hidden,
                                             device=device
                                             )          # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):  # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V             # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                                        # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)      # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)                         # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_layers, n_heads, max_len, enc_voc_size, trg_pad_idx, dropout, device):
        super(Encoder, self).__init__()
        self.trg_pad_idx = trg_pad_idx
        self.src_emb = nn.Embedding(enc_voc_size, d_model)                     # 把字转换字向量
        self.pos_emb = PositionalEncoding(d_model=d_model,
                                          dropout=dropout,
                                          max_len=max_len,
                                          device=device)                               # 加入位置信息
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_heads, device) for _ in range(n_layers)])

    def forward(self, enc_inputs):                                               # enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs)                                   # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs)                                  # enc_outputs: [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs, self.trg_pad_idx)           # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, device):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model=d_model,
                                                n_heads=n_heads,
                                                device=device)
        self.dec_enc_attn = MultiHeadAttention(d_model=d_model,
                                               n_heads=n_heads,
                                               device=device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model,
                                             ffn_hidden=ffn_hidden,
                                             device=device)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):                                             # dec_inputs: [batch_size, tgt_len, d_model]
                                                                                # enc_outputs: [batch_size, src_len, d_model]
                                                                                # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
                                                                                # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs,
                                                        dec_self_attn_mask)     # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs,
                                                      dec_enc_attn_mask)        # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)                                 # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, ffn_hidden, dropout, max_len, tgt_voc_size, trg_pad_idx, device):
        super(Decoder, self).__init__()
        self.device = device
        self.trg_pad_idx = trg_pad_idx
        self.tgt_emb = nn.Embedding(tgt_voc_size, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model,
                                          dropout=dropout,
                                          max_len=max_len,
                                          device=device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, ffn_hidden, device) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):                         # dec_inputs: [batch_size, tgt_len]
                                                                                    # enc_intpus: [batch_size, src_len]
                                                                                    # enc_outputs: [batsh_size, src_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)                                      # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs).to(self.device)                             # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.trg_pad_idx).to(self.device)   # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(self.device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).to(self.device)   # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.trg_pad_idx)               # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:                                                   # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                    # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
                                                                                    # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, d_model, max_len, n_layers, n_heads, ffn_hidden, drop_out, enc_voc_size, tgt_voc_size, trg_pad_idx, device):
        super(Transformer, self).__init__()
        self.Encoder = Encoder(d_model=d_model,
                               ffn_hidden=ffn_hidden,
                               n_layers=n_layers,
                               n_heads=n_heads,
                               max_len=max_len,
                               enc_voc_size=enc_voc_size,
                               trg_pad_idx=trg_pad_idx,
                               dropout=drop_out,
                               device=device).to(device)
        self.Decoder = Decoder(d_model=d_model,
                               n_layers=n_layers,
                               n_heads=n_heads,
                               ffn_hidden=ffn_hidden,
                               dropout=drop_out,
                               max_len=max_len,
                               tgt_voc_size=tgt_voc_size,
                               trg_pad_idx=trg_pad_idx,
                               device=device).to(device)
        self.projection = nn.Linear(d_model, tgt_voc_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):                          # enc_inputs: [batch_size, src_len]
                                                                        # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)          # enc_outputs: [batch_size, src_len, d_model],
                                                                        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)                        # dec_outpus    : [batch_size, tgt_len, d_model],
                                                                        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
                                                                        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)                       # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns #变为二维[batch_size*tgt_len, tgt_vocab_size]
