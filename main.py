import argparse
import huffman
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import Transformer
from torch.optim import lr_scheduler
from utils import data_generator, DataInfo, draw_plot, dataset_gen

parser = argparse.ArgumentParser()
parser.add_argument("--num_epoch", type=int, default=1701, help="number of epochs of training")
parser.add_argument("--batch_num", type=int, default=50, help="size of the batches")
parser.add_argument("--image_nums", type=int, default=1, help="number of generated images")
parser.add_argument("--lr", type=float, default=0.0001, help="sgd: learning rate")
parser.add_argument("--momentum", type=float, default=0.99, help="sgd: momentum")
parser.add_argument("--epoch_plot", type=int, default=1700, help="the epoch of plotting")
parser.add_argument("--lr_scheduler_b", type=int, default=1400, help="the epoch of lr_scheduler")
parser.add_argument("--lr_gamma", type=float, default=0.1, help="the gamma of lr_scheduler")
parser.add_argument("--n_heads", type=int, default=8, help="the nums of attention")
parser.add_argument("--d_model", type=int, default=256, help="the dimmension of vocab")
parser.add_argument("--n_layers", type=int, default=6, help="the nums of layer")
parser.add_argument("--fname", type=str, default="./model/300net.pth", help="the name of the model ")
opt = parser.parse_args()

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# generate the dataset
seq_len = 20
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h"]
weight = np.array([2, 1, 1, 1, 1, 1, 1, 1])
datainfo, src_vocab_size, tgt_vocab_size, max_len = dataset_gen(seq_len, alphabet, weight)

# Define model
model = Transformer(opt.n_heads, opt.d_model, opt.n_layers, src_vocab_size, tgt_vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opt.lr_scheduler_b], gamma=opt.lr_gamma)


# Train
for epoch in range(opt.num_epoch):
    seq, seq_int, code, code_int, code_onehot, code_int_c = data_generator(datainfo, opt.batch_num)
    seq_int = torch.LongTensor(seq_int).to(device)
    code_int = torch.LongTensor(code_int).to(device)
    code_int_c = torch.LongTensor(code_int_c).to(device)
    optimizer.zero_grad()
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(seq_int, code_int)
    loss = criterion(outputs, code_int_c.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()

    # plot
    if epoch == opt.epoch_plot:
        outputs = [torch.argmax(x).item() for x in outputs]
        draw_plot(
            torch.LongTensor(outputs).view(-1),
            code_int_c.cpu().view(-1),
            dec_enc_attns,
            seq,
            seq_len,
            max_len,
            datainfo.codebook,
            opt.image_nums,
            "attention",
        )
    print("Epoch:", "%04d" % (epoch + 1), "loss =", f"{loss}")

# save the model
path_model = "./model/"
if not os.path.exists(path_model):
    os.makedirs(path_model)
torch.save(model.state_dict(), opt.fname)
