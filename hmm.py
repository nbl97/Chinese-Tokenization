import os
import numpy as np

class HMM2(object):
    def __init__(self, p_trans, p_emit):
        # initiate two kinds of probabilities
        self.p_trans = p_trans
        self.p_emit = p_emit

        self.t_list = list(set([k.split(' ')[1] for k in self.p_emit.keys()]))
        self.t_len = len(self.t_list)
        print(self.t_len, self.t_list)

        self.start_sgn = 'bos'
        self.end_sgn = 'eos'

    def start_prob(self, term):
        key = self.start_sgn + ' ' + term
        return self.p_trans[key] if key in self.p_trans.keys() else 1E-8

    def trans_prob(self, token1, token2):
        key = token1 + ' ' + token2
        return self.p_trans[key] if key in self.p_trans.keys() else 1E-8
    
    def emit_prob(self, term, token):
        """
        """
        key = term + ' ' + token
        return self.p_emit[key] if key in self.p_emit.keys() else 1E-8

    def calc_prob(self, tokens):
        a = np.zeros((len(tokens)+1, self.t_len))
        for i, t in enumerate(self.t_list):
            a[0, i] = self.start_prob(t) * self.emit_prob(t, tokens[0])
        for i in range(1, len(tokens)):
            toke = tokens[i]
            for j, t2 in enumerate(self.t_list):
                for k, t1 in enumerate(self.t_list):
                    a[i, j] += a[i-1, k] * self.trans_prob(t1, t2) * self.emit_prob(t2, toke)
        for j in range(self.t_len):
            a[-1, j] = a[-2, j] * self.trans_prob(self.t_list[j], self.end_sgn)
        return sum(a[-1, :])

'''
class HMM3(object):
    def __init__(self, p_trans, p_emit):
        # initiate two kinds of probabilities
        self.p_trans = p_trans
        #self.p_trans2 = p_trans2
        self.p_emit = p_emit

        self.t_list = list(set([k.split(' ')[1] for k in self.p_emit.keys()]))
        self.t_len = len(self.t_list)
        print(self.t_len, self.t_list)

        self.start_sgn = 'bos'
        self.end_sgn = 'eos'

    def start_prob(self, term):
        key = self.start_sgn + ' ' + term
        return self.p_trans[key] if key in self.p_trans.keys() else 1E-8

    def trans_prob(self, token1, token2):
        key = token1 + ' ' + token2
        return self.p_trans[key] if key in self.p_trans.keys() else 1E-8
    
    def emit_prob(self, term, token):
        """
        """
        key = term + ' ' + token
        return self.p_emit[key] if key in self.p_emit.keys() else 1E-8

    def calc_prob(self, tokens):
        a = np.zeros((len(tokens)+2, self.t_len))
        for i, t in enumerate(self.t_list):
            a[0, i, 0] = self.start_prob(t) * self.emit_prob(t, tokens[0])
        for i in range(1, len(tokens)):
            toke = tokens[i]
            for j, t2 in enumerate(self.t_list):
                for k, t1 in enumerate(self.t_list):
                    a[i, j] += a[i-1, k] * self.trans_prob(t1, t2) * self.emit_prob(t2, toke)
        for j in range(self.t_len):
            a[-1, j] = a[-2, j] * self.trans_prob(self.t_list[j], self.end_sgn)
        return sum(a[-1, :])
'''

class HMM_word(object):
    def __init__(self, p_trans, bos = '<s>', eos = '</s>'):
        self.p_trans = p_trans
        self.bos = bos
        self.eos = eos

    def trans(self, w1, w2):
        key = '%s %s'%(w1, w2)
        if key in self.p_trans.keys():
            return self.p_trans[key]
        elif w2 != self.eos:
            return 1E-4**len(w2)
        else:
            return 1E-4**len(w1)

    def calc_prob(self, tokens):
        tokens = [self.bos] + tokens + [self.eos]
        prob = 1
        for w1, w2 in zip(tokens[:-1], tokens[1:]):
            prob *= self.trans(w1, w2)
        return prob


    def find(self, tokens):
        lens = len(tokens)
        a = np.zeros((lens, lens))
        b = np.zeros((lens, lens),dtype = np.long)
        for i in range(4):
            a[0, i] = self.trans(self.bos, tokens[:i+1])
            #print(0, i, a[0,i])
        for i in range(1, lens):
            for j in range(i, min(lens, i+4)):
                a[i, j] = 0.0
                for k in range(max(0, i-4), i):
                    new = a[k, i-1] * self.trans(tokens[k:i], tokens[i:j+1])
                    if new > a[i, j]:
                        a[i,j] = new
                        b[i,j] = k
                        
        k_old = lens - 1
        for i in range(lens):
            a[i, lens-1] *= self.trans(tokens[i:], self.eos)
        k = np.argmax(a[:, lens-1])
        splits = []
        while k > 0:
            splits.append(tokens[k:k_old+1])
            k, k_old = b[k, k_old], k-1
        splits.append(tokens[:k_old+1])
        return splits[::-1]

if __name__ == "__main__":
    filename = 'viterbi-tokenizer-master\\0603.mod-2gram'
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [l.strip().split('\t') for l in lines]
        probs = {}
        for l in lines:
            if len(l)<2:
                continue
            probs[l[0]] = float(l[1])
    
    model = HMM_word(probs)
    sentence = '今天是个好日子，你们想不想出去到游乐场玩一玩？'
    #sentence = '直播 这 个 行业 ， 最近 真是 越来越 热 ， 不但 有 小米 直播 横空出世 ， 连 腾讯 都 连续 推出 了 腾讯 直播 和 企鹅 直播 两 款 app ， 并 同时 投资 了 斗鱼 直播 和 龙珠 直播 ， 加上 之前 大火 的 映客 花椒 17 等等 ， 不得不 令 人 感慨 ： 风 ， 又 来 了 。'
    sentence = sentence.replace(' ', '')
    model.find(sentence)