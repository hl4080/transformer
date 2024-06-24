from util.datasets import DataSet
from util.tokenizer import Tokenizer
from conf import *

tokenizer = Tokenizer()
loader = DataSet(
    ('de', 'en'), 
    ['train', 'valid', 'test'],
    tokenizer.tokenize_en, 
    tokenizer.tokenize_de,
    False
    )

train_iter, valid_iter, test_iter = loader.load_iter()
src_vocab, tgt_vocab = loader.build_vocab(train_iter)
enc_inputs_batchs, dec_inputs_batchs, dec_outputs_batchs = loader.load_data(train_iter, src_vocab, tgt_vocab, batch_size)
enc_voc_size = len(src_vocab)
tgt_voc_size = len(tgt_vocab)

src_pad_idx = src_vocab.get_stoi()['<pad>']
trg_pad_idx = tgt_vocab.get_stoi()['<pad>']
trg_sos_idx = tgt_vocab.get_stoi()['<bos>']


