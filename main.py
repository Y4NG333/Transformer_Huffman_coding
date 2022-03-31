import argparse
import huffman
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from matplotlib import pyplot as plt
from model import Transformer
from torch.optim import lr_scheduler
from utils import data_generator, DataInfo, draw_plot

parser = argparse.ArgumentParser()
parser.add_argument("--num_epoch", type=int, default=101, help="number of epochs of training")
parser.add_argument("--batch_num", type=int, default=50, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-5, help="sgd: learning rate")
parser.add_argument("--momentum", type=float, default=0.99, help="sgd: momentum")
parser.add_argument("--epoch_plot", type=int, default=100, help="the epoch of plotting")
parser.add_argument("--lr_scheduler", type=int, default=50, help="the epoch of lr_scheduler")
parser.add_argument("--lr_gamma", type=float, default=0.05, help="the gamma of lr_scheduler")
parser.add_argument("--n_heads", type=int, default=8, help="the nums of attention")
parser.add_argument("--d_model", type=int, default=512, help="the dimmension of vocab")
parser.add_argument("--n_layers", type=int, default=6, help="the nums of layer")
parser.add_argument("--fname", type=str, default="./model/100net.pth", help="the name of the model ")
opt = parser.parse_args()

# Used to configure the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

alphabet = ["a", "b", "c"]
weight = np.array([2, 1, 1])
prob = weight / np.sum(weight)

seq_len = 5
max_len = 12
pad_symbol = "2"
dimension = 3
tgt_vocab_size = 3
src_vocab_size = len(alphabet)
fname = "./model/100net.pth"

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
model = Transformer(opt.n_heads, opt.d_model, opt.n_layers, src_vocab_size, tgt_vocab_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opt.lr_scheduler], gamma=opt.lr_gamma)

# Train
for epoch in range(opt.num_epoch):
    seq, seq_int, code, code_int, code_onehot = data_generator(datainfo, opt.batch_num)
    seq_int = torch.LongTensor(seq_int)
    code_int = torch.LongTensor(code_int)

    optimizer.zero_grad()
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(seq_int, code_int)
    loss = criterion(outputs.float(), torch.LongTensor(code_onehot).view(-1, dimension).float())
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch == opt.epoch_plot:
        draw_plot(outputs, code_int.view(-1), dec_enc_attns, seq)
    print("Epoch:", "%04d" % (epoch + 1), "loss =", f"{loss}")

path_model = "./model/"
if not os.path.exists(path_model):
    os.makedirs(path_model)
torch.save(model.state_dict(), opt.fname)
