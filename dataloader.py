"""Pre-process input text, tokenizing, building vocabs, and pre-train word
level vectors."""

import os
from collections import namedtuple
from copy import copy
import pickle
from util import lines
from itertools import chain
import numpy as np

Question = namedtuple('Question',
                      ['id', 'content', 'read_feat', 'answer', 'false_options', 'labels'])


class QuestionLoader:
    def __init__(self, ques_file, index2str, read_dir, *label_file):
        """Read question file as data list. Same behavior on same file."""
        self.range = None
        self.ques = lines(ques_file, skip=1)
        self.range = slice(0, len(self), 1)
        self.read_dir = read_dir
        self.labels = []
        self.itos = dict()
        self.stoi = dict()
        self.itos['word'] = lines(index2str)
        self.stoi['word'] = {s: i for i, s in enumerate(self.itos['word'])}
        self.read_feat = pickle.load(open(read_dir, 'rb'))

        for filename in label_file:
            f = lines(filename)
            label_name = f[0].split('\t')[1]
            _map = {}
            for l in f[1:]:
                qid, v = l.split('\t')
                _map[qid] = v if label_name.startswith('[') else float(v)
            self.labels.append((label_name, _map))

        qs = []
        for line in self.ques:
            fields = line.split('\t')
            qid, content, answer = fields[0], fields[1], fields[2]
            false_options = fields[4]
            content = content.split()
            for i in range(len(content)):
                if content[i].startswith('{img:'):
                    content[i] = self.stoi['word']['{img}']
                else:
                    content[i] = self.stoi['word'].get(content[i]) or 0
            answer = [self.stoi['word'].get(a) or 0 for a in answer.split()]

            if len(false_options):
                false_options = [[self.stoi['word'].get(x) or 0
                                  for x in o.split()]
                                 for o in false_options.split('::')]

            else:
                false_options = None

            labels = {}
            for name, _map in self.labels:
                if qid in _map:
                    v = _map[qid]
                    if isinstance(v, float):
                        labels[name] = v
                    else:
                        labels[name] = [self.stoi[name].get(k) or 0
                                        for k in v.split(',')]

            read_feat = self.read_feat[qid]
            # word
            type_len = len(set(content))
            word_len = len(content)
            ttr = type_len / word_len

            pos_noun = 0
            pos_verb = 0
            pos_ad = 0
            pos_pu = 0
            for _, word_pos in list(chain(*read_feat['pos_tag'])):
                if word_pos.startswith('N'):
                    pos_noun += 1
                elif word_pos == 'VV':
                    pos_verb += 1
                elif word_pos.startswith('AD'):
                    pos_ad += 1
                elif word_pos == 'PU':
                    pos_pu += 1

            # sentence
            ques_len = sum([sum([len(w[0]) for w in s]) for s in read_feat['pos_tag']])

            syn_tree_height = [max([len(w) - len(w.strip()) for w in s.split('\n')]) // 2 for s in read_feat['parse']]
            # syn_tree_height_max = max(syn_tree_height)
            syn_tree_height_mean = np.mean(syn_tree_height)

            syn_dep_dis = [np.mean([abs(p[1] - p[2]) for p in s]) for s in read_feat['dependency_parse']]
            syn_dep_dis_mean = np.mean(syn_dep_dis)

            total_words = sum(map(lambda s: len(s), read_feat['pos_tag']))
            per_word = total_words / len(read_feat['pos_tag'])
            per_col = sum([len([w for w in s if len(w[0]) > 2]) for s in
                           read_feat['pos_tag']]) / total_words
            per_syl = ques_len / total_words
            gunning_fog = 0.4 * (per_word + 100 * per_col)
            fk_level = 0.39 * per_word + 11.8 * per_syl - 15.59
            _read_feat = [type_len, word_len, ttr,
                          pos_noun / word_len, pos_verb / word_len, pos_ad / word_len, pos_pu / word_len,
                          ques_len, syn_tree_height_mean, syn_dep_dis_mean,
                          gunning_fog, fk_level]
            qs.append(Question(qid, content, _read_feat, answer, false_options, labels))

        self.ques = qs

    def split_(self, split_ratio):
        first_size = int(len(self) * (1 - split_ratio))
        other = copy(self)
        self.range = slice(0, first_size, 1)
        other.range = slice(first_size, len(other), 1)
        return other

    def split(self, split_ratio):
        first_size = int(len(self) * (1 - split_ratio))
        a = copy(self)
        b = copy(self)
        a.range = slice(0, first_size, 1)
        b.range = slice(first_size, len(self), 1)
        return a, b

    def __len__(self):
        return len(self.ques) if self.range is None \
            else self.range.stop - self.range.start

    def __getitem__(self, x):
        if isinstance(x, int):
            x += self.range.start
            item = slice(x, x + 1, 1)
        else:
            item = slice(x.start + self.range.start,
                         x.stop + self.range.start, 1)
        qs = self.ques[item]

        if isinstance(x, int):
            return qs[0]
        else:
            return qs


def load_word2vec(emb_file='words.emb.npy'):
    if not os.path.exists(emb_file):
        return None
    return np.load(emb_file)
