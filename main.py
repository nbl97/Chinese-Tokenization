import numpy as np

from ngram import get_ngram_prob, gene_proposal
from hmm import HMM

def get_test_sets():
    '''Get test sentence and dict from rmrb dataset

    Return:
        test_set: a list of tokenized sentences. Each of it is a list of words.
            For example, [['今天', '是', '星期二', '。'], ...]
        dicts:    a list of all words that appear in the dataset
    '''
    with open('rmrb_modified.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        test_set = []
        dicts = []
        for line in lines:
            idata = line.strip().split()[1:]
            if len(idata) == 0:
                continue
            idata = [x.split('/')[0] for x in idata]
            test_set.append(idata)
            dicts.extend(idata)

    dicts = list(set(dicts))        
    return test_set, dicts

if __name__ == "__main__":
    params = get_ngram_prob()
    model = HMM(params['p2'], params['p1'])

    test_targets, dicts = get_test_sets()
    for sen in test_targets:
        ori_sen = ''.join(sen)
        print('input : ', ori_sen)
        cands = gene_proposal(ori_sen, dicts)
        print('cand gets')
        scores = list()
        for cand in cands[:6]:
            print(cand.split(' '))
            scores.append(model.calc_prob(cand.split(' ')))
        idx = np.argmax(scores)
        print(idx)
        print('result: ', cands[idx])
        
