from flask import Flask,jsonify,render_template
from flask_restful import Api, Resource, reqparse
import numpy as np
app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('history_data', type=str)

#读取数据
def get_seq(line):
    line = line.replace("NA", "nan")
    id = line.split(" ")[0]
    seq = [float(x) for x in line.split(" ")[1][1:-2].split(',')]
    seqs = np.array([seq])
    return id, seqs

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

# 使用cnn_model预测
import torch
import torch.nn as nn

cnn_model = nn.Sequential(
        nn.Conv1d(1, 8, 32), 
        nn.Sigmoid(),
        nn.Conv1d(8, 8, 17),
        nn.Sigmoid(),
        nn.Conv1d(8, 1, 16),
        nn.Sigmoid())
cnn_model.load_state_dict(torch.load('cnn_model.pkl'))

# Input [batch_size, 1, 64]
# Output [batch_size, 1, 1]
input_seq_len = 63

def pred_next_n(seq, n=72):
    assert seq.shape[1] == input_seq_len, "Input sequence length error"
    seq_pred = np.copy(seq).reshape([-1, 1, input_seq_len])
    for i in range(n):
        pred_next = cnn_model(torch.from_numpy(seq_pred[:, :, -input_seq_len:]).float()).cpu().detach().numpy()
        seq_pred = np.concatenate([seq_pred, pred_next], axis=2)
    return seq_pred[:, :, -n:].reshape([-1, n])
    	
def pred_all():
    pred_seq_normed = pred_next_n(seq_normalized[:, -input_seq_len:], 72)
    seq_min = seq.min(axis=1, keepdims=True)
    seq_max = seq.max(axis=1, keepdims=True)
    pred_seq = (pred_seq_normed * (seq_max - seq_min) + seq_min)
    return pred_seq


class Predict(Resource):
    def get(self):
        pass
    def post(self):
        args = parser.parse_args()
        seq_id, seq = get_seq(args["history_data"])
        seq = seq[:, -63:]
        seq[0] = linear_interpolation(seq[0])
        seq_min = seq.min()
        seq_max = seq.max()
        pred_seq = pred_next_n(seq/(seq_max-seq_min) - seq_min)[0] * (seq_max -seq_min) + seq_min
        #输出结果
        resp_line = seq_id + " \"" + ",".join(["{:.2f}".format(x) for x in pred_seq]) + "\""
        return jsonify({'msg':'success!', 'data':resp_line})
    def put(self):
        pass

    def delete(self):
        pass
@app.route('/predict')
def index():
    return render_template('index.html')
api.add_resource(Predict, "/do_predict")
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80,debug=True)
