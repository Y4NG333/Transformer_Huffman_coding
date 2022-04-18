import argparse
import huffman
import numpy as np
import os
import torch
import torch.nn as nn

from model import Transformer
from utils import data_generator, DataInfo, draw_plot

parser = argparse.ArgumentParser()
parser.add_argument("--path_model", type=str, default="./model/300net.pth", help="the path of model")
parser.add_argument("--test_nums", type=int, default=1, help="size of the test")
parser.add_argument("--n_heads", type=int, default=8, help="the nums of attention")
parser.add_argument("--d_model", type=int, default=256, help="the dimmension of vocab")
parser.add_argument("--n_layers", type=int, default=6, help="the nums of layer")
parser.add_argument("--sentence", type=str, default="abcab", help="length 5 sequences ")
parser.add_argument("--fname", type=str, default="./model/300net.pth", help="the name of the model ")
opt = parser.parse_args()

# generate the test dataset
alphabet = ["a", "b", "c"]
weight = np.array([2, 1, 1])
prob = weight / np.sum(weight)

seq_len = 5
max_len = 12
pad_symbol = "0"
src_vocab_size = len(alphabet) + 1
tgt_vocab_size = 4

weighted_tuple = [(alphabet[i], weight[i]) for i in range(len(alphabet))]
codebook = huffman.codebook(weighted_tuple)

print(codebook)
for item in codebook:
    codebook[item] = codebook[item].replace("0", "2")
print(codebook)

datainfo = DataInfo(
    alphabet=alphabet,
    prob=prob,
    codebook=codebook,
    seq_len=seq_len,
    max_len=max_len,
    pad_symbol=pad_symbol,
)

# Load model
path_model = "./model/"
model_test = Transformer(opt.n_heads, opt.d_model, opt.n_layers, src_vocab_size, tgt_vocab_size)
model_test.load_state_dict(torch.load(opt.fname))

# Generate test set
criterion = nn.CrossEntropyLoss()
seq, seq_int, code, code_int, code_onehot, code_int_c = data_generator(datainfo, opt.test_nums)
seq_int = torch.LongTensor(seq_int)
code_int = torch.LongTensor(code_int)
outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model_test(seq_int, code_int)
real_out = [torch.argmax(x).item() for x in outputs]

loss = criterion(outputs, torch.LongTensor(code_int_c).view(-1))  # code_int.view(-1)
print("loss =", f"{loss}")
draw_plot(outputs, torch.LongTensor(code_int_c).view(-1), dec_enc_attns, seq)


def inference(sentence):
    sentence_n = [item for item in sentence]
    code_symbolwise = np.vectorize(datainfo.codebook.__getitem__)(sentence_n)
    seq_chr_map = {c: i for i, c in enumerate(datainfo.alphabet)}
    seq_int = np.vectorize(seq_chr_map.get)(sentence_n)
    seq_int += 1
    print("input", seq_int)

    # encoder
    seq_int = torch.LongTensor(seq_int).view(1, 5)
    with torch.no_grad():
        enc_outputs, enc_self_attns = model_test.encoder(seq_int)

    # get dec_input
    dec_input = torch.zeros(1, 0).type_as(seq_int.data)
    next_symbol = 3
    for i in range(max_len):
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]])], -1)
        dec_outputs, dec_self_attns, dec_enc_attns = model_test.decoder(dec_input, seq_int, enc_outputs)
        projected = model_test.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word

    # output
    predict, _, _, _ = model_test(seq_int, dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print("output", [n.item() for n in predict.squeeze()])


print("translate_sentence")
inference(opt.sentence)
