# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:02:58 2022

@author: 73182
"""
import argparse
import huffman
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from model import Transformer
from torch.optim import lr_scheduler
from utils import data_generator, DataInfo, draw_plot, dataset_gen,get_alphabet_weight
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--num_epoch", type=int, default=30001, help="number of epochs of training")
parser.add_argument("--batch_num", type=int, default=50, help="size of the batches")
parser.add_argument("--image_nums", type=int, default=1, help="number of generated images")
parser.add_argument("--lr", type=float, default=0.0001, help="sgd: learning rate")
parser.add_argument("--momentum", type=float, default=0.99, help="sgd: momentum")
parser.add_argument("--epoch_plot", type=int, default=5000, help="the epoch of plotting")
parser.add_argument("--lr_scheduler_b", type=int, default=15000, help="the epoch of lr_scheduler")
parser.add_argument("--lr_gamma", type=float, default=0.1, help="the gamma of lr_scheduler")
#parser.add_argument("--n_heads", type=int, default=1, help="the nums of attention")
parser.add_argument("--d_model", type=int, default=256, help="the dimmension of vocab")
parser.add_argument("--n_layers", type=int, default=6, help="the nums of layer")
parser.add_argument("--fname", type=str, default="model.pth", help="the name of the model ")
opt = parser.parse_args()

# Device configuration
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# generate the dataset
n_heads = 8#1 2 4 8
seq_len = 20#10 20 50 100
alphabet_len = 10# 5 10 20
name = "head"+str(n_heads)+"seq_len"+str(seq_len)+"___alphabet_len"+str(alphabet_len)
path_cur = "./"+name+"/"
if not os.path.exists(path_cur):
    os.makedirs(path_cur)
path_images = "./images/"+"attention"+name+"/"
if not os.path.exists(path_images):
    os.makedirs(path_images)

alphabet,weight = get_alphabet_weight(alphabet_len)
print(alphabet)
print(weight)
np.savetxt(path_cur+'weight.txt',weight,fmt='%d',newline=' ')


datainfo, src_vocab_size, tgt_vocab_size, max_len = dataset_gen(seq_len, alphabet, weight)

# Define model
model = Transformer(n_heads, opt.d_model, opt.n_layers, src_vocab_size, tgt_vocab_size).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opt.lr_scheduler_b], gamma=opt.lr_gamma)

LOSS = []
min_loss = 0
# Train
for epoch in range(opt.num_epoch):
    seq, seq_int, code, code_int, code_onehot, code_int_c = data_generator(datainfo, opt.batch_num)
    seq_int = torch.LongTensor(seq_int).cuda()
    code_int = torch.LongTensor(code_int).cuda()
    code_int_c = torch.LongTensor(code_int_c).cuda()
    optimizer.zero_grad()
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(seq_int, code_int)
    loss = criterion(outputs, code_int_c.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    cur_loss = loss.cpu().item()
    LOSS.append(cur_loss)

        
    # plot
    if (epoch+1) % opt.epoch_plot==0:
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
            "attention"+name+"/"+"/"+str(epoch),
        )
    print("Epoch:", "%04d" % (epoch + 1), "loss =", f"{loss}")

# save the model
    if epoch==29990:
        min_loss = cur_loss
        path_model = path_cur+"/model/"
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        torch.save(model.state_dict(), path_model+opt.fname)
    if epoch>29990 and cur_loss<min_loss:
        print(epoch)
        min_loss = cur_loss
        torch.save(model.state_dict(), path_model+opt.fname)

plt.title("loss")
plt.plot(LOSS)
plt.savefig(path_cur + "loss.png")
plt.close()



