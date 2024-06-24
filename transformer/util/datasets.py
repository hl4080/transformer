# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
import numpy as np

sentences = [('Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.', 'Two young, White males are outside near many bushes.'), 
             ('Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.', 'Several men in hard hats are operating a giant pulley system.'), 
             ('Ein kleines Mädchen klettert in ein Spielhaus aus Holz.', 'A little girl climbing into a wooden playhouse.'),
             ('Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.', 'A man in a blue shirt is standing on a ladder cleaning a window.'),
             ('Zwei Männer stehen am Herd und bereiten Essen zu.', 'Two men are at the stove preparing food.')] 

# 数据集函数
class DataSet(Data.Dataset):
    def __init__(self, ext, split, tokenize_de, tokenize_en, Multi30k=False):
        super(DataSet, self).__init__()
        self.ext = ext
        self.split = split
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.Multi30k = Multi30k
        print('dataset initializing start')

    '''
    load iters(train, valid, test) from Multi30k dataset
    eg. [('Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.', 'Two young, White males are outside near many bushes.'),]
    '''
    def load_iter(self):
        iters = []
        if self.Multi30k:
            for split in self.split:
                iter = [(src, tgt) for src, tgt in Multi30k(split=split, language_pair=self.ext)]
                iters.append(iter)
        else:
            iters.append([(src, tgt) for src, tgt in sentences])
            iters.extend([[] for _ in range(2)])  # 添加两个空列表
        print('dataset initializing done')
        return iters[0], iters[1], iters[2]

    def yield_tokens(self, data_iter, tokenize):
        for src, tgt in data_iter:
            yield tokenize(src)
            yield tokenize(tgt)

    '''
    split the iter to build vocalulary according to tokenizer
    eg. ['<unk>', '<pad>', '<bos>', '<eos>', '.', ',', 'Büsche', 'Freien', 'Männer', 'Nähe', 'Two', 'White', 'Zwei', 'are', 'bushes']
    '''
    def build_vocab(self, iter):
        src_vocab = build_vocab_from_iterator(
            self.yield_tokens(iter, self.tokenize_de), 
            specials=["<unk>", "<pad>", "<bos>", "<eos>"]
        )
        tgt_vocab = build_vocab_from_iterator(
            self.yield_tokens(iter, self.tokenize_en), 
            specials=["<unk>", "<pad>", "<bos>", "<eos>"]
        )
        src_vocab.set_default_index(src_vocab["<unk>"])
        tgt_vocab.set_default_index(src_vocab["<unk>"])
        return src_vocab, tgt_vocab

    def pad_sequences(self, sequences, padding_value):
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = np.full((len(sequences), max_len), padding_value)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences
    '''
    load data in terms of list
    split the sentence, then tensorize the words according to vocabulary
    eg. [[13 56 82  7 74 53 21  6 45 29 81 18  4  1  1], [27  7 64 30 38  9 17  4  1  1  1  1  1  1  1]]
    '''
    def load_data(self, raw_text_iter, src_vocab, tgt_vocab, batch_size=1):
        #enc_inputs, dec_inputs, dec_outputs = [], [], []
        src_pad_idx = src_vocab.get_stoi()['<pad>']
        trg_pad_idx = tgt_vocab.get_stoi()['<pad>']
        src_inputs, tar_inputs = [], []
        for (raw_src, raw_tgt) in raw_text_iter:
            src_indexs = [src_vocab[token] for token in self.tokenize_de(raw_src)]
            tar_indexs = [tgt_vocab[token] for token in self.tokenize_en(raw_tgt)]
            '''
            enc_input = [src_vocab[token] for token in self.tokenize_de(raw_src)]
            dec_input = [tgt_vocab["<bos>"]] + [tgt_vocab[token] for token in self.tokenize_en(raw_tgt)]
            dec_output = [tgt_vocab[token] for token in self.tokenize_en(raw_tgt)] + [tgt_vocab["<eos>"]]
            enc_inputs.append(enc_input)
            dec_inputs.append(dec_input)
            dec_outputs.append(dec_output)
            '''
            src_inputs.append(src_indexs)
            tar_inputs.append(tar_indexs)
        '''
        enc_inputs = self.pad_sequences(enc_inputs, src_pad_idx) #add padding
        dec_inputs = self.pad_sequences(dec_inputs, src_pad_idx)
        dec_outputs = self.pad_sequences(dec_outputs, trg_pad_idx)
        '''
        src_tensors = self.pad_sequences(src_inputs, src_pad_idx) #add padding
        tar_tensors = self.pad_sequences(tar_inputs, trg_pad_idx)
        src_tensors = torch.LongTensor(src_tensors)
        tar_tensors = torch.LongTensor(tar_tensors)
        dataset = TensorDataset(src_tensors, tar_tensors) # batch tensors
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        enc_input_batchs, dec_input_batchs, dec_output_batchs = [], [], []
        for batch in dataloader:
            src_batch, tar_batch = batch
            enc_input_batchs.append(src_batch)
            bos = torch.full((tar_batch.size(0), 1), tgt_vocab["<bos>"], dtype=tar_batch.dtype)
            eos = torch.full((tar_batch.size(0), 1), tgt_vocab["<eos>"], dtype=tar_batch.dtype)
            dec_input_batchs.append(torch.cat((bos, tar_batch), dim=1))
            dec_output_batchs.append(torch.cat((tar_batch, eos), dim=1))
        return enc_input_batchs, dec_input_batchs, dec_output_batchs
    

'''
    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
'''
