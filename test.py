import argparse
import huffman
import numpy as np
import os
import torch
import torch.nn as nn

from model import Transformer
from utils import data_generator, DataInfo, draw_plot

parser = argparse.ArgumentParser()
parser.add_argument("--test_nums", type=int, default=10, help="size of the test")
parser.add_argument("--n_heads", type=int, default=8, help="the nums of attention")
parser.add_argument("--d_model", type=int, default=512, help="the dimmension of vocab")
parser.add_argument("--n_layers", type=int, default=6, help="the nums of layer")
parser.add_argument("--fname", type=str, default="./model/100net.pth", help="the name of the model ")
opt = parser.parse_args()

alphabet = ["a", "b", "c"]
weight = np.array([2, 1, 1])
prob = weight / np.sum(weight)

seq_len = 5
max_len = 12
pad_symbol = "2"
dimension = 3
src_vocab_size = len(alphabet)
tgt_vocab_size = 3

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

# Load model
path_model = "./model/"
model_test = Transformer(opt.n_heads, opt.d_model, opt.n_layers, src_vocab_size, tgt_vocab_size)
model_test.load_state_dict(torch.load(opt.fname))

# Generate test set
seq, seq_int, code, code_int, code_onehot = data_generator(datainfo, opt.test_nums)
seq_int = torch.LongTensor(seq_int)
code_int = torch.LongTensor(code_int)
outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model_test(seq_int, code_int)
real_out = [torch.argmax(x).item() for x in outputs]

loss = criterion(outputs.float(), torch.LongTensor(code_onehot).view(-1, dimension).float())
print("loss =", f"{loss}")
