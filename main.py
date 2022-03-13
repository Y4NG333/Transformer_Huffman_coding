import huffman
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from matplotlib import pyplot as plt
from model import Transformer
from utils import data_generator, DataInfo

# Used to configure the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

alphabet = ["a", "b", "c"]
weight = np.array([2, 1, 1])
prob = weight / np.sum(weight)

num_epoch = 5
batch_num = 50
seq_len = 5
max_len = 12
pad_symbol = "2"

weighted_tuple = [(alphabet[i], weight[i]) for i in range(len(alphabet))]
codebook = huffman.codebook(weighted_tuple)

datainfo = DataInfo(
    alphabet=alphabet,
    prob=prob,
    codebook=codebook,
    seq_len=seq_len,
    max_len=max_len,
    pad_symbol=pad_symbol,
)

# Define model
n_heads, d_model, n_layers, src_vocab_size, tgt_vocab_size = 8, 512, 6, len(alphabet), 3
model = Transformer(n_heads, d_model, n_layers, src_vocab_size, tgt_vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

# Train
for epoch in range(num_epoch):
    seq, seq_int, code, code_int = data_generator(datainfo, batch_num)
    seq_int = torch.LongTensor(seq_int)
    code_int = torch.LongTensor(code_int)

    optimizer.zero_grad()
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(seq_int, code_int)
    loss = criterion(outputs, code_int.view(-1))
    loss.backward()
    optimizer.step()

    print("Epoch:", "%04d" % (epoch + 1), "loss =", f"{loss}")
