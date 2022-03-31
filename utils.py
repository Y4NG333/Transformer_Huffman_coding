from dataclasses import dataclass
import numpy as np
import torch.utils.data as Data
import torch
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

    # generate onehot (currently not needed)
    code_onehot = np.zeros((code.shape[0], code.shape[1], 3))  # assuming binary code (with padding)
    idx_code, idx_bin = np.meshgrid(np.arange(code.shape[0]), np.arange(code.shape[1]))
    code_onehot[idx_code, idx_bin, code_int.T] = 1
    # code_onehot = code_onehot.reshape(600,3)

    return seq, seq_int, code, code_int, code_onehot


# Plotting tool
def draw_plot(outputs, labels, dec_enc_attns, seq):
    real_out = [torch.argmax(x).item() for x in outputs]
    attn = dec_enc_attns[5][0][7].detach().numpy()
    attn = list(map(list, zip(*attn)))

    plt.figure(figsize=(10, 10.5))
    plt.xticks([i for i in range(12)], [real_out[i] for i in range(len(real_out))])
    plt.yticks([i for i in range(5)], [seq[0][i] for i in range(len(seq[0]))])
    plt.imshow(attn)
    plt.show()

    plt.figure(figsize=(10, 10.5))
    plt.yticks([i for i in range(3)], [0, 1, 2])
    plt.xticks([i for i in range(12)], [real_out[i] for i in range(len(real_out))])
    tu = outputs[0:12].detach().numpy()
    tu += 1
    tu = list(map(list, zip(*tu)))
    plt.imshow(tu)
    plt.show()
