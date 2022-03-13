from dataclasses import dataclass
import numpy as np
import torch.utils.data as Data


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
    seq = np.random.choice(datainfo.alphabet, size=(batch_num, datainfo.seq_len), p=datainfo.prob)

    # apply codebook to each symbol
    code_symbolwise = np.vectorize(datainfo.codebook.__getitem__)(seq)

    # merge binary codes (join) and add padding (ljust)
    code_merged = [list("".join(s_arr).ljust(datainfo.max_len, datainfo.pad_symbol)) for s_arr in code_symbolwise]
    code = np.array(code_merged)

    # generate sequence int
    seq_chr_map = {c: i for i, c in enumerate(datainfo.alphabet)}
    seq_int = np.vectorize(seq_chr_map.get)(seq)

    # generate onehot (currently not needed)
    seq_onehot = np.zeros((seq.shape[0], seq.shape[1], len(seq_chr_map)))
    idx_seq, idx_base = np.meshgrid(np.arange(seq.shape[0]), np.arange(seq.shape[1]))
    seq_onehot[idx_seq, idx_base, seq_int.T] = 1

    # generate code int
    code_int = code.astype(int)

    # generate onehot (currently not needed)
    code_onehot = np.zeros((code.shape[0], code.shape[1], 3))  # assuming binary code (with padding)
    idx_code, idx_bin = np.meshgrid(np.arange(code.shape[0]), np.arange(code.shape[1]))
    code_onehot[idx_code, idx_bin, code_int.T] = 1

    return seq, seq_int, code, code_int


# Used for drawing to rotate the x-axis and y-axis
def rotate(nums):
    a = len(nums)
    b = len(nums[0])
    res = []
    for j in range(b):
        cur = []
        for i in range(a):
            cur.append(nums[i][j])
        res.append(cur)
    return res


# Plotting tool
def draw_plot():
    if epoch == 1:
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
        plt.xticks([i for i in range(50)], [real_out[i] for i in range(len(real_out))])
        plt.yticks([i for i in range(10)], [y_label[i] for i in range(len(y_label))])
        plt.imshow(rotate(dec_enc_attns[5][0][7].detach().numpy()))
        plt.show()
    pass
