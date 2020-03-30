import time

start_time = time.clock()

import numpy as np

# 读取数据
def read_data():
    with open("dataset_campus_competition.txt") as train_file:
        lines = train_file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("NA", "nan")
        ids = [line.split(" ")[0] for line in lines]
        seqs = [[float(x) for x in line.split(" ")[1][1:-2].split(',')] for line in lines]
        return np.array(seqs), ids
seqs, ids =  read_data()

# 线性插值
def linear_interpolation(seq):
    i = 0
    while i < len(seq):
        if i == 0 and np.isnan(seq[i]):
            # Find first i that seq[i] is not nan
            while np.isnan(seq[i]):
                i += 1
            for j in range(i):
                seq[j] = seq[i]
        elif np.isnan(seq[i]):
            start = i - 1
            while np.isnan(seq[i]):
                i += 1
                if i == len(seq):
                    break
            if i < len(seq):
                for j in range(start, i):
                    seq[j] = (i - j)/(i - start) * seq[start] + (j - start)/(i - start) * seq[i]
            else:
                for j in range(start, i):
                    seq[j] = seq[start]
        i += 1
    return seq
for row in range(seqs.shape[0]):
    seqs[row] = linear_interpolation(seqs[row])
    


# 数据展示
# import matplotlib.pyplot as plt
# plt.plot(seqs[0])
# plt.show()
# plt.plot(seqs[2])
# plt.show()
# plt.plot(seqs[21])
# plt.show()
np.max(seqs), np.min(seqs)
seqs_normalized = np.copy(seqs)
for row in range(seqs_normalized.shape[0]):
    row_min = seqs_normalized[row].min()
    row_max = seqs_normalized[row].max()
    seqs_normalized[row] = (seqs_normalized[row] - row_min) / (row_max - row_min)
   
# 使用Pytorch训练模型
import torch
import torch.nn as nn

cnn_model = nn.Sequential(
        nn.Conv1d(1, 8, 32), 
        nn.Sigmoid(),
        nn.Conv1d(8, 8, 17),
        nn.Sigmoid(),
        nn.Conv1d(8, 1, 16),
        nn.Sigmoid())
# Input [batch_size, 1, 64]
# Output [batch_size, 1, 1]
input_seq_len = 63
train_set = seqs_normalized[:100]
test_set = seqs_normalized[100:]

def get_train_batch(batch_size=32):
    row_indices = np.random.choice(100, batch_size)
    rows = train_set[row_indices].tolist()
    for r in range(len(rows)):
        start_ind = np.random.randint(len(rows[0]) - input_seq_len)
        rows[r] = rows[r][start_ind: start_ind + input_seq_len + 1]
    rows = np.array(rows)
    rows = rows.reshape([batch_size, 1, input_seq_len + 1])
    return rows[:, :, :-1], rows[:, :, -1:]

def pred_next_n(seqs, n=72):
    assert seqs.shape[1] == input_seq_len, "Input sequence length error"
    seqs_pred = np.copy(seqs).reshape([-1, 1, input_seq_len])
    for i in range(n):
        pred_next = cnn_model(torch.from_numpy(seqs_pred[:, :, -input_seq_len:]).float()).cpu().detach().numpy()
        seqs_pred = np.concatenate([seqs_pred, pred_next], axis=2)
    return seqs_pred[:, :, -n:].reshape([-1, n])
    	
def train(n_rounds=500, batch_size=16):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn_model.parameters())
    for i in range(n_rounds):
        if i % 100 == 0:
            test_pred = pred_next_n(test_set[:, -(input_seq_len + 72):-72])
            mse = np.mean(np.square(test_pred - test_set[:, -72:]))
            print("Test set MSE:", mse)
        optimizer.zero_grad()
        xs, ys = get_train_batch(batch_size)
        xs = torch.from_numpy(xs).float()
        ys = torch.from_numpy(ys).float()
        pred_ys = cnn_model(xs)
        loss = criterion(pred_ys, ys)
        loss.backward()
        optimizer.step()
        

train(10000)

#保存模型

torch.save(cnn_model.state_dict(),'cnn_model.pkl')

#进行预测

# def plot_pred(index):
#     pred_seq_normed = pred_next_n(np.array([seqs_normalized[index, -input_seq_len:]]), 72)[0]
#     seq_min = seqs[index].min()
#     seq_max = seqs[index].max()
#     pred_seq = pred_seq_normed * (seq_max - seq_min) + seq_min
#     plt.plot(np.arange(168), seqs[index])
#     plt.plot(np.arange(168, 168+72), pred_seq)
#     plt.show()
# for i in range(4):
#     plot_pred(i)

def pred_all():
    pred_seqs_normed = pred_next_n(seqs_normalized[:, -input_seq_len:], 72)
    seq_min = seqs.min(axis=1, keepdims=True)
    seq_max = seqs.max(axis=1, keepdims=True)
    pred_seqs = (pred_seqs_normed * (seq_max - seq_min) + seq_min)
    return pred_seqs

all_preds = pred_all()
with open("output.txt", "w") as out:
    for i in range(all_preds.shape[0]):
         line = ids[i] + " \"" + ",".join(["{:.2f}".format(x) for x in all_preds[i]]) + "\""
         out.write(line + "\n")

end_time = time.clock()
print(end_time - start_time)