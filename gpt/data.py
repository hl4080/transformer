import torch
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

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
tgt_voc_size = len(vocab)
trg_pad_idx = vocab['<pad>']