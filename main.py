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

# codebook =   {'a': '10101', 'b': '0000', 'c': '10100', 'd': '11101', 'e': '11110', 'f': '01110', 'g': '01101', 'h': '0101',
# 'i': '10010', 'j': '10111', 'k': '11011', 'l': '01111', 'm': '10000', 'n': '0010', 'o': '01100', 'p': '11000', 'q': '10011',
# 'r': '11111', 's': '11001', 't': '11100', 'u': '0100', 'v': '10001', 'w': '11010', 'x': '0011', 'y': '10110', 'z': '0001'}

seq_len = 10
res, original_in, original_out01, original_out, = data_generator(
    codebook, seq_len=seq_len, data_num=200)


enc_inputs = torch.LongTensor(original_in)
dec_inputs = torch.LongTensor(original_out)
dec_outputs = torch.LongTensor(original_out)

# attention
loader = Data.DataLoader(
    MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

src_vocab_size = 27
tgt_vocab_size = 4  # 3 is used for padding. 0 and 1 is used for encoding
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

            real_out = [0 if x == 2 else x for x in real_out]
            real_out = [2 if x == 3 else x for x in real_out]

            label_out = [1 if x == 1 else x for x in dec_outputs[0]]
            label_out = [0 if x == 2 else x for x in label_out]
            label_out = [2 if x == 3 else x for x in label_out]

            str_out = ""
            print("output", str_out.join(str(i) for i in real_out))
            for item in res:
                if label_out == item[3]:
                    print(" label", item[2])
                    print("str", item[0])
                    y_label = item[0]
            flag = 1

            plt.figure(figsize=(10, 10.5))
            plt.xticks([i for i in range(50)], [real_out[i]
                       for i in range(len(real_out))])
            plt.yticks([i for i in range(10)], [y_label[i]
                       for i in range(len(y_label))])
            plt.imshow(rotate(dec_enc_attns[5][0][7].detach().numpy()))
            plt.show()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
