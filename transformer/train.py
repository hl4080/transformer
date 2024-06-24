# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch.nn as nn
import torch
from torch.optim import Adam
from data import *
from model.transformer import Transformer

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

if __name__ == "__main__":
    model = Transformer(d_model=d_model,
                        max_len=max_len,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        ffn_hidden=ffn_hidden,
                        drop_out=drop_prob,
                        enc_voc_size=enc_voc_size,
                        tgt_voc_size=tgt_voc_size,
                        trg_pad_idx=trg_pad_idx,
                        device=device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)         # 忽略 占位符 索引.
    optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)
    assert len(enc_inputs_batchs) == len(dec_inputs_batchs) == len(dec_outputs_batchs), "Size mismatch between tensors"

    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    model.apply(initialize_weights)
    for epoch in range(num_epoch):
        epoch_loss = 0
        for i, _ in enumerate(enc_inputs_batchs):
            enc_inputs = enc_inputs_batchs[i].to(device)    # enc_inputs : [batch_size, src_len]
            dec_inputs = dec_inputs_batchs[i].to(device)    # dec_inputs : [batch_size, tgt_len]
            dec_outputs = dec_outputs_batchs[i].to(device)  # dec_outputs: [batch_size, tgt_len]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(epoch_loss/len(enc_inputs_batchs)))
    torch.save(model, 'model.pth')
    print("模型已保存")
