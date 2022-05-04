import argparse
import huffman
import numpy as np
import os
import torch
import torch.nn as nn

from model import Transformer
from utils import data_generator, DataInfo, draw_plot, dataset_gen, draw_plot_test


parser = argparse.ArgumentParser()
parser.add_argument("--path_model", type=str, default="./model/300net.pth", help="the path of model")
parser.add_argument("--test_nums", type=int, default=1, help="size of the test")
parser.add_argument("--n_heads", type=int, default=8, help="the nums of attention")
parser.add_argument("--d_model", type=int, default=256, help="the dimmension of vocab")
parser.add_argument("--n_layers", type=int, default=6, help="the nums of layer")
parser.add_argument("--sequence", type=str, default=10000, help="nums of sequences ")
parser.add_argument("--batch_inference", type=int, default=100, help="the nums of the inference ")
parser.add_argument("--fname", type=str, default="./model/300net.pth", help="the name of the model ")

opt = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# generate the test dataset
seq_len = 20
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h"]
weight = np.array([2, 1, 1, 1, 1, 1, 1, 1])
datainfo, src_vocab_size, tgt_vocab_size, max_len = dataset_gen(seq_len, alphabet, weight)
# Load model
path_model = "./model/"
model_test = Transformer(opt.n_heads, opt.d_model, opt.n_layers, src_vocab_size, tgt_vocab_size).to(device)
model_test.load_state_dict(torch.load(opt.fname))

criterion = nn.CrossEntropyLoss()

seq, seq_int, code, code_int, code_onehot, code_int_c = data_generator(datainfo, opt.test_nums)
seq_int = torch.LongTensor(seq_int).to(device)
code_int = torch.LongTensor(code_int).to(device)

outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model_test(seq_int, code_int)
real_out = [torch.argmax(x).item() for x in outputs]

loss = criterion(outputs, torch.LongTensor(code_int_c).to(device).view(-1))  # code_int.view(-1)
print("loss =", f"{loss}")
draw_plot(
    outputs,
    torch.LongTensor(code_int_c).view(-1),
    dec_enc_attns,
    seq,
    seq_len,
    max_len,
    datainfo.codebook,
    "attention2",
)


def inference(sequence):
    sequence_correct = 0
    sequence_all = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for j in range(int(sequence / opt.batch_inference)):
        # seq_int, label = get_seq_int(sequence)
        seq, seq_int, code, code_int, code_onehot, label = data_generator(datainfo, opt.batch_inference)

        # encoder
        seq_int = torch.LongTensor(seq_int).to(device)
        with torch.no_grad():
            enc_outputs, enc_self_attns = model_test.encoder(seq_int)
        # get dec_input
        dec_input = torch.zeros(opt.batch_inference, 0).type_as(seq_int.data).to(device)
        next_symbol = torch.tensor([[3] for i in range(opt.batch_inference)])
        for i in range(datainfo.max_len):
            dec_input = torch.cat([dec_input.detach(), next_symbol.to(device)], -1)
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
        result = [(label[i] == output[i]).all() for i in range(opt.batch_inference)]
        sequence_correct += sum(result)
        sequence_all += len(result)
        print(j, sum(result) / len(result))
        if j == 0:
            draw_plot_test(
                torch.LongTensor(output).view(-1),
                torch.LongTensor(label).view(-1),
                dec_enc_attns,
                seq,
                seq_len,
                max_len,
                datainfo.codebook,
                opt.batch_inference,
            )

    print("accuracy", sequence_correct / sequence_all)


print("translate_sentence")
inference(opt.sequence)
