import torch.utils.data as Data
from numpy import random


def data_generator(codebook, seq_len, data_num):
    alphabet = [0, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    res = []
    data = random.randint(1, 27, (data_num, seq_len))
    data_t = []
    for i in range(data_num):
        in_s = ""
        in_d = data[i]
        in_s = in_s.join(alphabet[order] for order in in_d)
        out_s = ""
        out_s = out_s.join(codebook[alphabet[order]] for order in in_d)
        out_d = [int(k) for k in out_s]
        res.append([in_s, in_d, out_s, out_d])
        data_t.append(out_d)
    return res, data, data_t


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


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
