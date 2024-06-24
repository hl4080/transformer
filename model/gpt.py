import torch.nn as nn
import numpy as np
import torch
from model.transformer import PositionalEncoding, MultiHeadAttention, PoswiseFeedForwardNet

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

    def forward(self, dec_inputs, dec_self_attn_mask):                          # dec_inputs: [batch_size, tgt_len, d_model]
                                                                                # enc_outputs: [batch_size, src_len, d_model]
                                                                                # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
                                                                                # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs,
                                                        dec_self_attn_mask)     # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
                                                                                # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)                                 # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn
    

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

    def forward(self, dec_inputs):                         
                                                                                    # enc_intpus: [batch_size, src_len]
                                                                                    # enc_outputs: [batsh_size, src_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)                                      # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs).to(self.device)                             # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.trg_pad_idx).to(self.device)   # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(self.device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).to(self.device)   # [batch_size, tgt_len, tgt_len]
        dec_self_attns= []
        for layer in self.layers:                                                   # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                    # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
                                                                                    # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns
    
class GPT(nn.Module):
    def __init__(self, d_model, max_len, n_layers, n_heads, ffn_hidden, drop_out, tgt_voc_size, trg_pad_idx, device):
        super(GPT, self).__init__()
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

    def forward(self, dec_inputs):                       
                                                                        # dec_inputs: [batch_size, tgt_len]
        dec_outputs, dec_self_attns = self.Decoder(dec_inputs)        # dec_outpus    : [batch_size, tgt_len, d_model],
                                                                        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
                                                                        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)                       # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns #变为二维[batch_size*tgt_len, tgt_vocab_size]
