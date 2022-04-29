import torch
import huffman
import numpy as np
import torch.utils.data as Data

from dataclasses import dataclass
from matplotlib import pyplot as plt


@dataclass
class DataInfo:
    alphabet: list
    prob: np.ndarray
    codebook: dict
    seq_len: int
    max_len: int
    pad_symbol: chr


def data_generator(datainfo, batch_num):
    # 2d array of alphabets
    # ['a' 'a' 'b' 'c' 'a']
    seq = np.random.choice(datainfo.alphabet, size=(batch_num, datainfo.seq_len), p=datainfo.prob)

    # apply codebook to each symbol
    # ['0' '0' '11' '10' '0']
    code_symbolwise = np.vectorize(datainfo.codebook.__getitem__)(seq)

    # merge binary codes (join) and add padding (ljust)
    # ['0', '0', '1', '1', '1', '0', '0', '2', '2', '2', '2', '2']
    code_merged = [list("".join(s_arr).ljust(datainfo.max_len, datainfo.pad_symbol)) for s_arr in code_symbolwise]
    # ['0' '0' '1' '1' '1' '0' '0' '2' '2' '2' '2' '2']
    code = np.array(code_merged)

    # generate sequence int
    # [0 0 1 2 0]
    seq_chr_map = {c: i for i, c in enumerate(datainfo.alphabet)}
    seq_int = np.vectorize(seq_chr_map.get)(seq)

    # generate onehot (currently not needed)
    seq_onehot = np.zeros((seq.shape[0], seq.shape[1], len(seq_chr_map)))
    idx_seq, idx_base = np.meshgrid(np.arange(seq.shape[0]), np.arange(seq.shape[1]))
    seq_onehot[idx_seq, idx_base, seq_int.T] = 1

    # generate code int
    # [0 0 1 1 1 0 0 2 2 2 2 2]
    code_int = code.astype(int)

    code_int_b = []
    for item in code_int:
        cur = [3]
        for i in range(len(item) - 1):
            cur.append(item[i])
        code_int_b.append(cur)
    code_int_b = np.array(code_int_b)

    # generate onehot (currently not needed)
    code_onehot = np.zeros((code.shape[0], code.shape[1], 4))
    idx_code, idx_bin = np.meshgrid(np.arange(code.shape[0]), np.arange(code.shape[1]))
    code_onehot[idx_code, idx_bin, code_int.T] = 1

    seq_int += 1

    return seq, seq_int, code, code_int_b, code_onehot, code_int


# generate the dataset
def dataset_gen(seq_len, alphabet, weight):
    prob = weight / np.sum(weight)
    pad_symbol = "0"
    tgt_vocab_size = 4
    src_vocab_size = len(alphabet) + 1
    weighted_tuple = [(alphabet[i], weight[i]) for i in range(len(alphabet))]
    codebook = huffman.codebook(weighted_tuple)
    max_len = 0
    for item in codebook:
        codebook[item] = codebook[item].replace("0", "2")
        max_len = max(max_len, len(codebook[item]))
    max_len = max_len * seq_len + 2
    datainfo = DataInfo(
        alphabet=alphabet,
        prob=prob,
        codebook=codebook,
        seq_len=seq_len,
        max_len=max_len,
        pad_symbol=pad_symbol,
    )
    return datainfo, src_vocab_size, tgt_vocab_size, max_len


def replace(inputs):
    replace_dict = {2: 0, 0: 2, 1: 1}
    outputs = np.vectorize(replace_dict.get)(inputs)
    return outputs


# Plotting tool
def draw_plot(outputs, labels, dec_enc_attns, seq, seq_len, max_len):
    real_out = [torch.argmax(x).item() for x in outputs]
    real_out = replace(real_out)
    label = replace([labels[x].item() for x in range(0, max_len)])
    real_out = real_out[0:max_len]
    attn = dec_enc_attns[5][0][7].cpu().detach().numpy()
    attn = list(map(list, zip(*attn)))

    plt.figure(figsize=(15, 15))
    plt.xticks([i for i in range(max_len)], [real_out[i] for i in range(len(real_out))])
    plt.yticks([i for i in range(seq_len)], [seq[0][i] for i in range(len(seq[0]))])
    plt.imshow(attn)
    plt.show()
