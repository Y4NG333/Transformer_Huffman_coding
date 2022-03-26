import huffman
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from matplotlib import pyplot as plt
from model import Transformer
from utils import data_generator, DataInfo, draw_plot

# Used to configure the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

alphabet = ["a", "b", "c"]
weight = np.array([2, 1, 1])
prob = weight / np.sum(weight)

seq_len = 5
max_len = 12
pad_symbol = "2"
test_nums = 10
dimension = 3
fname = "./model/20net.pth"

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
criterion = nn.MSELoss()

# Define model
n_heads, d_model, n_layers, src_vocab_size, tgt_vocab_size = 8, 512, 6, len(alphabet), 3

# Load model parameters
path_model = "./model/"
model_test = Transformer(n_heads, d_model, n_layers, src_vocab_size, tgt_vocab_size)
model_test.load_state_dict(torch.load(fname))

# Generate test set
seq, seq_int, code, code_int, code_onehot = data_generator(datainfo, test_nums)
seq_int = torch.LongTensor(seq_int)
code_int = torch.LongTensor(code_int)
outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model_test(seq_int, code_int)
real_out = [torch.argmax(x).item() for x in outputs]

print("real out", np.array(real_out).reshape(-1, max_len))
print("code int", code_int)
loss = criterion(outputs.float(), torch.LongTensor(code_onehot).view(-1, dimension).float())
print("loss =", f"{loss}")
