from typing import Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import string
from util import SeqBatch, device


class FeatureExtractor(nn.Module):
    def __init__(self, feat_size=512):
        super(FeatureExtractor, self).__init__()
        self.feat_size = feat_size

    def make_batch(self, data, pretrain=False):
        """Make batch from input data (python data / np arrays -> tensors)"""
        return torch.tensor(data)

    def load_emb(self, emb):
        pass

    def pretrain_loss(self, batch):
        """Returns pretraining loss on a batch of data"""
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def forward(self, input_x, input_y):
        attention_matrix = torch.matmul(input_y, input_x.transpose(1, 2)) / 16
        attention_weight = torch.softmax(attention_matrix, dim=2)
        attention_out = torch.matmul(attention_weight, input_x)
        attention_out = torch.mean(attention_out, dim=1)
        return attention_out


class Stimulus(nn.Module):
    def __init__(self, deep_extractor, feat_size, read_size):
        super(Stimulus, self).__init__()
        self.device = device
        self.deep_extractor = deep_extractor
        self.ia = InterAgg(in_features=(feat_size, read_size))

    def forward(self, x, rf):
        df = self.deep_extractor(x)[1]
        sr = self.ia(df, rf)
        return 0, torch.cat([df, rf, sr], dim=1)


class TaskMCP(nn.Module):
    def __init__(self, deep_extractor, in_features):
        super(TaskMCP, self).__init__()
        self.device = device
        self.deep_extractor = deep_extractor
        self.attention = AttentionLayer()
        self.linear = nn.Linear(2 * in_features, in_features)

    def forward(self, xa, x0, x1, x2):
        sa = self.deep_extractor(xa)[0]
        s0 = self.deep_extractor(x0)[0]
        s1 = self.deep_extractor(x1)[0]
        s2 = self.deep_extractor(x2)[0]
        f0 = self.attention(s0, sa)
        f1 = self.attention(s1, sa)
        f2 = self.attention(s2, sa)
        s0 = F.relu(self.linear(torch.cat([torch.mean(sa, dim=1) - f0, torch.mean(sa, dim=1) * f0], dim=1)))
        s1 = F.relu(self.linear(torch.cat([torch.mean(sa, dim=1) - f1, torch.mean(sa, dim=1) * f1], dim=1)))
        s2 = F.relu(self.linear(torch.cat([torch.mean(sa, dim=1) - f2, torch.mean(sa, dim=1) * f2], dim=1)))
        return 0, torch.cat([s0, s1, s2], dim=1)


class TaskFBP(nn.Module):
    def __init__(self, deep_extractor, in_features):
        super(TaskFBP, self).__init__()
        self.device = device
        self.deep_extractor = deep_extractor
        self.attention = AttentionLayer()
        self.linear = nn.Linear(2 * in_features, in_features)
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, xa, x0):
        sa = self.deep_extractor(xa)[0]
        s0 = self.deep_extractor(x0)[0]
        f0 = self.attention(s0, sa)
        s0 = F.relu(self.linear(torch.cat([torch.mean(sa, dim=1) - f0, torch.mean(sa, dim=1) * f0], dim=1)))
        return 0, s0


class QuesDiff(nn.Module):
    def __init__(self, feat_size, read_size):
        super(QuesDiff, self).__init__()
        self.device = device
        self.predictor = FCLayer(feat_size * 7 + read_size, 1).to(device)

    def forward(self, sti, task):
        return self.predictor(torch.cat([sti, task], dim=1))


class BiLSTM(FeatureExtractor):
    """Sequence-to-sequence feature extractor based on RNN. Supports different
    input forms and different RNN types (LSTM/GRU), """
    def __init__(self, emb_size=(256, 'size of embedding vectors'),
                 feat_size=(256, 'size of hidden vectors'),
                 rnn=('LSTM', 'size of rnn hidden states', ['LSTM', 'GRU']),
                 layers=(1, 'number of layers'),
                 embs='data/word.emb'):
        super(BiLSTM, self).__init__()

        self.feat_size = feat_size
        rnn_size = self.rnn_size = feat_size // 2
        self.we = nn.Embedding(embs.shape[0], emb_size)
        self.we.weight.data.copy_(torch.from_numpy(embs))
        self.we.weight.requires_grad = False

        self.rnn_type = rnn
        self.config = dict()
        self.config['rnn'] = rnn
        if rnn == 'GRU':
            self.rnn = nn.GRU(emb_size, rnn_size, layers,
                              bidirectional=True, batch_first=True)
            self.h0 = nn.Parameter(torch.rand(layers * 2, 1, rnn_size))
        else:
            self.rnn = nn.LSTM(emb_size, rnn_size, layers,
                               bidirectional=True, batch_first=True)
            self.h0 = nn.Parameter(torch.rand(layers * 2, 1, rnn_size))
            self.c0 = nn.Parameter(torch.rand(layers * 2, 1, rnn_size))

        self.drop = nn.Dropout(0.2)

    def make_batch(self, data):
        """Returns embeddings"""
        embs = []
        read_feat = []

        for q in data:
            _embs = [self.we(torch.tensor([0], device=device))]
            for w in q.content:
                if isinstance(w, int):
                    word = torch.tensor([w], device=device)
                    item = self.we(word)
                    _embs.append(item)
            _embs.append(self.we(torch.tensor([1], device=device)))
            embs.append(torch.cat(_embs, dim=0))
            read_feat.append(q.read_feat)
        embs = SeqBatch(embs, device=device)
        return embs, torch.tensor(read_feat, device=device).float()

    def make_batch_mcp(self, data):
        """Returns embeddings"""
        embs_a = []
        embs_0 = []
        embs_1 = []
        embs_2 = []
        for q in data:
            _embs_a = [self.we(torch.tensor([0], device=device))]
            _embs_0 = [self.we(torch.tensor([0], device=device))]
            _embs_1 = [self.we(torch.tensor([0], device=device))]
            _embs_2 = [self.we(torch.tensor([0], device=device))]
            for w in q.content:
                if isinstance(w, int):
                    word = torch.tensor([w], device=device)
                    item = self.we(word)
                    _embs_a.append(item)
                    _embs_0.append(item)
                    _embs_1.append(item)
                    _embs_2.append(item)
            for w in q.answer:
                word = torch.tensor([w], device=device)
                item = self.we(word)
                _embs_a.append(item)
            for w in q.false_options[0]:
                word = torch.tensor([w], device=device)
                item = self.we(word)
                _embs_0.append(item)
            for w in q.false_options[1]:
                word = torch.tensor([w], device=device)
                item = self.we(word)
                _embs_1.append(item)
            for a in q.false_options[2]:
                word = torch.tensor([a], device=device)
                item = self.we(word)
                _embs_2.append(item)
            _embs_a.append(self.we(torch.tensor([1], device=device)))
            _embs_0.append(self.we(torch.tensor([1], device=device)))
            _embs_1.append(self.we(torch.tensor([1], device=device)))
            _embs_2.append(self.we(torch.tensor([1], device=device)))

            embs_a.append(torch.cat(_embs_a, dim=0))
            embs_0.append(torch.cat(_embs_0, dim=0))
            embs_1.append(torch.cat(_embs_1, dim=0))
            embs_2.append(torch.cat(_embs_2, dim=0))

        embs_a = SeqBatch(embs_a, device=device)
        embs_0 = SeqBatch(embs_0, device=device)
        embs_1 = SeqBatch(embs_1, device=device)
        embs_2 = SeqBatch(embs_2, device=device)
        return embs_a, embs_0, embs_1, embs_2

    def make_batch_fbp(self, data):
        """Returns embeddings"""
        embs_a = []
        for q in data:
            _embs_a = [self.we(torch.tensor([0], device=device))]
            for w in q.content:
                if isinstance(w, int):
                    word = torch.tensor([w], device=device)
                    item = self.we(word)
                    _embs_a.append(item)
            for w in q.answer:
                if isinstance(w, int):
                    word = torch.tensor([w], device=device)
                    item = self.we(word)
                    _embs_a.append(item)
            _embs_a.append(self.we(torch.tensor([1], device=device)))
            embs_a.append(torch.cat(_embs_a, dim=0))
        embs_a = SeqBatch(embs_a, device=device)
        return embs_a

    def forward(self, batch: SeqBatch):
        packed = batch.packed()
        h0 = self.init_h(packed.batch_sizes[0])
        ws, hl = self.rnn(packed, h0)
        if self.rnn_type == 'LSTM':
            hl = hl[0]
        hl = batch.invert(hl.permute(1, 0, 2), 0).view(hl.size(1), -1)
        ws, _ = pad_packed_sequence(ws)
        return ws.permute(1, 0, 2), hl

    def init_h(self, batch_size):
        size = list(self.h0.size())
        size[1] = batch_size
        if self.config['rnn'] == 'GRU':
            return self.h0.expand(size)
        else:
            return self.h0.expand(size).contiguous(), self.c0.expand(size).contiguous()


class FCLayer(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(FCLayer, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.layer3(x)
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1.}]


class FCLayerWithSigmoid(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(FCLayerWithSigmoid, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1.}]


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, input, coeff) -> torch.Tensor:
        ctx.coeff = coeff
        output = input[0] * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()
        self.alpha = 1
        self.lo = 0
        self.hi = 1
        self.iter_num = 0
        self.max_iters = 1000

    def forward(self, *input):
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        return GradientReverseFunction.apply(input, coeff)


class DomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean'):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = GradientReverseLayer()
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class InterAgg(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = tuple(in_features)
        self.out_features = in_features[0] * in_features[1]
        self.weight = nn.Parameter(torch.Tensor(self.out_features, *in_features))
        chars = string.ascii_lowercase
        n = len(self.in_features)
        self.einsum_str = '{}{},z{}->z{}'.format(
            chars[n], chars[:n], ',z'.join(chars[:n]), chars[n]
        )
        bound = 1 / np.sqrt(max(self.in_features))
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, *inputs):
        out = torch.einsum(self.einsum_str, self.weight, *inputs)
        out = out.reshape(out.shape[0], *self.in_features)
        query = inputs[0]
        att_mat = torch.matmul(query.unsqueeze(1), out) / np.sqrt(out.shape[0])
        return torch.matmul(out, torch.softmax(att_mat, dim=2).transpose(1, 2)).squeeze()