import os
import random
import huffman
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from utils import rotate
from utils import MyDataSet
from model import Transformer
from utils import data_generator
from matplotlib import pyplot as plt

# Used to configure the environment (if there is no this code, the picture cannot be generated, the specific reason is unknown)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

alpha = [(chr(97+i), 1) for i in range(26)]
codebook = huffman.codebook(alpha)

# we need to use padding 0 ，so the vocab is 27 and 3
# 1 and 2 used for as encoding (old a-> 10101 now a -> 12121)
for i in codebook:
    codebook[i] = codebook[i].replace('0', '2')
    while(len(codebook[i]) != 5):
        codebook[i] += '0'
# codebook =  {'a': '12121', 'b': '22220', 'c': '12122', 'd': '11121', 'e': '11112',
#              'f': '21112', 'g': '21121', 'h': '21210', 'i': '12212', 'j': '12111',
#              'k': '11211', 'l': '21111', 'm': '12222', 'n': '22120', 'o': '21122',
#              'p': '11222', 'q': '12211', 'r': '11111', 's': '11221', 't': '11122',
#              'u': '21220', 'v': '12221', 'w': '11212', 'x': '22110', 'y': '12112', 'z': '22210'}


seq_len = 10
res, original_in, original_out = data_generator(
    codebook, seq_len=seq_len, data_num=300)


enc_inputs = torch.LongTensor(original_in)
dec_inputs = torch.LongTensor(original_out)
dec_outputs = torch.LongTensor(original_out)

# attention
loader = Data.DataLoader(
    MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)


src_vocab_size = 27
tgt_vocab_size = 3  # 0 is used for padding. 1 and 2 is used for encoding
src_len = 10  # enc_input max sequence length
tgt_len = 50  # dec_input(=dec_output) max sequence length
model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
flag = 0
for epoch in range(2):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(
            enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', f'{loss}')
        if epoch == 1 and flag == 0:
            real_out = [torch.argmax(outputs[x]).item() for x in range(0, 50)]
            print("output vec", real_out)
            print("label vec", dec_outputs[0])
            str_out = ""
            print("output", str_out.join(str(i) for i in real_out))

            cur_ = [i.item() for i in dec_outputs[0]]
            for item in res:
                if cur_ == item[3]:
                    print(" label", item[2])
                    print("str", item[0])
                    y_label = item[0]
            flag = 1

            plt.figure(figsize=(10, 10.5))
            plt.xticks([i for i in range(50)], [i for i in real_out])
            plt.yticks([i for i in range(10)], [y_label[i]
                       for i in range(len(y_label))])
            plt.imshow(rotate(dec_enc_attns[5][1][7].detach().numpy()))
            plt.show()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
