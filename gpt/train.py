# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from data import *
from conf import *
from model.gpt import GPT

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

if __name__ == "__main__":
    model = GPT(d_model=d_model,
                max_len=max_len,
                n_layers=n_layers,
                n_heads=n_heads,
                ffn_hidden=ffn_hidden,
                drop_out=drop_prob,
                tgt_voc_size=tgt_voc_size,
                trg_pad_idx=trg_pad_idx,
                device=device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)         # 忽略 占位符 索引.
    optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    print(f'dataset size {len(train_data)}')
    model.apply(initialize_weights)
    for epoch in range(num_epoch):
        epoch_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i: i+batch_size]
            src_batch = pad_sequence(batch, padding_value=vocab['<pad>'])
            tgt_input = src_batch[:-1, :].to(device)       # tgt_input : [batch_size, tgt_len]
            tgt_output = src_batch[1:, :].to(device)       # tgt_output : [batch_size, tgt_len]
            outputs, _, = model(tgt_input)
            loss = criterion(outputs, tgt_output.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(epoch_loss/(len(train_data)/batch_size)))
    torch.save(model, 'model.pth')
    print("模型已保存")


