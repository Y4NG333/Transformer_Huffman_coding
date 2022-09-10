# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 22:41:14 2022

@author: 73182
"""
import argparse
import huffman
import numpy as np
import os
import torch
import torch.nn as nn

from model import Transformer
from utils import data_generator, DataInfo, draw_plot, dataset_gen, get_alphabet_weight
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



def inference(sequence):
    sequence_correct = 0
    sequence_all = 0

    for j in range(int(sequence / opt.batch_inference)):
        # seq_int, label = get_seq_int(sequence)
        seq, seq_int, code, code_int, code_onehot, label = data_generator(datainfo, opt.batch_inference)

        # encoder
        seq_int = torch.LongTensor(seq_int).cuda()
        with torch.no_grad():
            enc_outputs, enc_self_attns = model_test.encoder(seq_int)
        # get dec_input
        dec_input = torch.zeros(opt.batch_inference, 0).type_as(seq_int.data).cuda()
        next_symbol = torch.tensor([[3] for i in range(opt.batch_inference)])
        for i in range(datainfo.max_len):
            dec_input = torch.cat([dec_input.detach(), next_symbol.cuda()], -1)
            dec_outputs, dec_self_attns, dec_enc_attns = model_test.decoder(dec_input, seq_int, enc_outputs)
            projected = model_test.projection(dec_outputs)
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            prob = prob[:, -1].resize(opt.batch_inference, 1)
            next_word = prob.data
            next_symbol = next_word
            

        # output
        predict, _, _, dec_enc_attns = model_test(seq_int, dec_input)
        predict = predict.data.max(1, keepdim=True)[1]
        output = [n.item() for n in predict.squeeze()]
        output = np.array(output).reshape(-1, max_len)
        #print(label[0])
        #print(output[0])
        
        correct = 0 
        cur_all = 0
        for m in range(opt.batch_inference):
            flag = 0;
            for n in range(datainfo.max_len):
                if output[m][n]!=label[m][n]:
                    flag = 1
                if output[m][n]==4:
                    break
            if flag==0:
                correct += 1
            cur_all += 1
            

        print(j, correct / cur_all)
        sequence_correct += correct
        sequence_all += cur_all
        
        if j == 1:
            draw_plot(
                torch.LongTensor(output).view(-1),
                torch.LongTensor(label).view(-1),
                dec_enc_attns,
                seq,
                seq_len,
                max_len,
                datainfo.codebook,
                opt.batch_inference,
                name_image=str(name)+"/"
            )
        
    print("accuracy", sequence_correct / sequence_all)

    np.savetxt(path_cur+'accuracy.txt', [sequence_correct / sequence_all],fmt='%s',newline=' ')

parser = argparse.ArgumentParser()
parser.add_argument("--test_nums", type=int, default=1, help="size of the test")
parser.add_argument("--image_nums", type=int, default=1, help="number of generated images")
#parser.add_argument("--n_heads", type=int, default=1, help="the nums of attention")
parser.add_argument("--d_model", type=int, default=256, help="the dimmension of vocab")
parser.add_argument("--n_layers", type=int, default=6, help="the nums of layer")
parser.add_argument("--sequence", type=str, default=10000, help="nums of sequences ")
parser.add_argument("--batch_inference", type=int, default=50, help="the nums of the inference ")
#parser.add_argument("--fname", type=str, default=path_cur+"/model/model.pth", help="the name of the model ")
opt = parser.parse_args()


n_heads = 8#1 2 4 8
seq_len = 20#10 20 50 100
alphabet_len = 10# 5 10 20
name = "head"+str(n_heads)+"seq_len"+str(seq_len)+"___alphabet_len"+str(alphabet_len)
path_cur = "./"+name+"/"
if not os.path.exists(path_cur):
    os.makedirs(path_cur)

path_images = "./images/"+name+"/"
if not os.path.exists(path_images):
    os.makedirs(path_images)

fname = path_cur+"/model/model.pth"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# generate the test dataset
alphabet,weight = get_alphabet_weight(alphabet_len)
weight = np.loadtxt(path_cur+'weight.txt')
print(weight)
datainfo, src_vocab_size, tgt_vocab_size, max_len = dataset_gen(seq_len, alphabet, weight)
# Load model
model_test = Transformer(n_heads, opt.d_model, opt.n_layers, src_vocab_size, tgt_vocab_size).cuda()
model_test.load_state_dict(torch.load(fname))

criterion = nn.CrossEntropyLoss()

seq, seq_int, code, code_int, code_onehot, code_int_c = data_generator(datainfo, opt.test_nums)
seq_int = torch.LongTensor(seq_int).cuda()
code_int = torch.LongTensor(code_int).cuda()

outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model_test(seq_int, code_int)
real_out = [torch.argmax(x).item() for x in outputs]

loss = criterion(outputs, torch.LongTensor(code_int_c).cuda().view(-1))  # code_int.view(-1)
print("loss =", f"{loss}")
outputs = [torch.argmax(x).item() for x in outputs]
draw_plot(
    torch.LongTensor(outputs).view(-1),
    torch.LongTensor(code_int_c).view(-1),
    dec_enc_attns,
    seq,
    seq_len,
    max_len,
    datainfo.codebook,
    opt.image_nums,
    "attention2",
)
    
    

    
print("translate_sentence")
inference(opt.sequence)
