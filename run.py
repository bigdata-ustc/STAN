import torch
import numpy as np
import pickle
from dataloader import load_word2vec
from module import BiLSTM, Stimulus, TaskMCP, QuesDiff
from util import PrefetchIter, device
import random

emb_size = 128
feat_size = 128
read_size = 12
rnn = "LSTM"
batch_size = 32
trade_off = 1

seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

state_dict = torch.load('./pretrained/model.pt')
ques_encoder = BiLSTM(emb_size=emb_size, embs=load_word2vec('./pretrained/words.emb.npy'),
                      feat_size=feat_size, layers=2, rnn=rnn).to(device)
stimulus_model = Stimulus(deep_extractor=ques_encoder, feat_size=feat_size * 2, read_size=read_size).to(device)
task_model = TaskMCP(deep_extractor=ques_encoder, in_features=feat_size).to(device)
model = QuesDiff(feat_size=feat_size, read_size=read_size).to(device)
ques_encoder.load_state_dict(state_dict['ques_encoder'])
stimulus_model.load_state_dict(state_dict['stimulus'])
task_model.load_state_dict(state_dict['task'])
model.load_state_dict(state_dict['model'])


def make_label(_qs):
    labels = [[q.labels['diff'] - 0.5] for q in _qs]
    return torch.tensor(labels).to(device)


def make_result(y_true, y_pred):
    y_pred = torch.cat(y_pred, 0).view(-1).cpu().numpy() + .5
    y_true = torch.cat(y_true, 0).view(-1).cpu().numpy() + .5
    y_pred, y_true = zip(*sorted(zip(y_pred, y_true)))
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return {
        'diff/mae-': float(np.abs(y_pred - y_true).mean()),
        'diff/rmse-': float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    }


# sys_path = '\\'
# test_ques_tar = QuestionLoader('./data/sample_data/phy_test.txt'.replace('/', sys_path),
#                                './data/words.txt'.replace('/', sys_path),
#                                './data/sample_data/phy_test_val.obj'.replace('/', sys_path),
#                                './data/sample_data/phy_test_val_diff.txt'.replace('/', sys_path))
# Use Pickle for Convenience
test_ques_tar = pickle.load(open('./data/target_test.obj', 'rb'))
stimulus_model.eval()
task_model.eval()
model.eval()
y_src_true = []
y_src_pred = []
y_tar_true = []
y_tar_pred = []
test_tar_iter = PrefetchIter(test_ques_tar, shuffle=False, batch_size=batch_size)
with torch.no_grad():
    for qs in test_tar_iter:
        labels_t = make_label(qs)
        x_t, rf_t = ques_encoder.make_batch(qs)
        _, f_st = stimulus_model(x_t, rf_t)
        ta, t0, t1, t2 = ques_encoder.make_batch_mcp(qs)
        _, f_tt = task_model(ta, t0, t1, t2)
        y_t = model(f_st, f_tt)
        y_tar_true.append(labels_t)
        y_tar_pred.append(y_t)
result = make_result(y_tar_true, y_tar_pred)

print('Result:%s' % str(result))
