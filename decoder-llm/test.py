import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import re

import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class SimplePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(SimpleSelfAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = self.key(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = self.value(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(context)

class SimpleDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(SimpleDecoderLayer, self).__init__()
        self.self_attn = SimpleSelfAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        tgt2 = self.self_attn(tgt, mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class SimpleDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, nhead, dim_feedforward=2048, dropout=0.1):
        super(SimpleDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SimplePositionalEncoding(d_model)
        self.layers = nn.ModuleList([SimpleDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt, tgt_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        for layer in self.layers:
            tgt = layer(tgt, tgt_mask)
        output = self.fc_out(tgt)
        return output
    
    # 加载数据
train_iter = PennTreebank(split='train')

# 构建词汇表
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
vocab.set_default_index(vocab["<unk>"])

# 数据预处理
def data_process(data_iter, vocab, tokenizer):
    data = []
    for text in data_iter:
        tokens = [vocab['<bos>']] + vocab(tokenizer(text)) + [vocab['<eos>']]
        data.append(torch.tensor(tokens, dtype=torch.long))
    return data

train_data = data_process(PennTreebank(split='train'), vocab, tokenizer)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def collate_fn(batch):
    src_batch = pad_sequence(batch, padding_value=vocab['<pad>'])
    return src_batch

def train(model, data, vocab_size, batch_size, num_epochs, lr):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            src_batch = collate_fn(batch)

            tgt_input = src_batch[:-1, :]
            tgt_output = src_batch[1:, :]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(tgt_input.device)

            optimizer.zero_grad()
            output = model(tgt_input, tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, vocab_size), tgt_output.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch: {epoch+1}, Loss: {total_loss / len(data)}')

vocab_size = len(vocab)
d_model = 512
num_layers = 6
nhead = 8
batch_size = 32
num_epochs = 10
lr = 0.001

model = SimpleDecoder(vocab_size, d_model, num_layers, nhead)
train(model, train_data, vocab_size, batch_size, num_epochs, lr)

def greedy_decode(model, src, max_len, start_symbol):
    src = torch.tensor([start_symbol]).unsqueeze(1)
    memory = src

    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(src.size(0)).to(src.device)
        out = model(src, tgt_mask=tgt_mask)
        prob = out[-1, :].squeeze().argmax(dim=-1)
        src = torch.cat([src, prob.unsqueeze(0)], dim=0)

    return src

start_symbol = vocab['<bos>']
max_len = 20
model.eval()
with torch.no_grad():
    input_sentence = "The quick brown fox"
    input_tensor = torch.tensor([vocab['<bos>']] + vocab(tokenizer(input_sentence)) + [vocab['<eos>']], dtype=torch.long).unsqueeze(1)
    output = greedy_decode(model, input_tensor, max_len, start_symbol)
    print("Generated response:", ' '.join([vocab.itos[idx] for idx in output.squeeze().tolist()]))