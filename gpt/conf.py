import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 32
max_len = 256
d_model = 512 # 字 Embedding 的维度
n_layers = 6 # 有多少个encoder和decoder
n_heads = 8 # Multi-Head Attention设置为8
ffn_hidden = 2048 # 前向传播隐藏层维度
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-5
weight_decay = 5e-4
adam_eps = 5e-9

num_epoch = 50