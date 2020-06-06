import numpy as np
import json

from ngram import get_ngram_prob, get_proposals, Config, pre_process
from hmm_new import HMM2 as HMM, HMM_word
from evaluation import evaluateSet

from tqdm import tqdm
import time
import re
import os

def get_test_sets():
    '''Get test sentence and dict from rmrb dataset

    Return:
        test_set: a list of tokenized sentences. Each of it is a list of words.
            For example, [['今天', '是', '星期二', '。'], ...]
        dicts:    a list of all words that appear in the dataset
    '''
    with open('rmrb.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        test_set = []
        dicts = []
        for line in lines:
            line = pre_process(line, use_re=0)
            idata = line.strip().split()
            if len(idata) == 0:
                continue
            idata = [x.split('/')[0] for x in idata]
            test_set.append(idata)
            dicts.extend(idata)

    dicts = list(set(dicts))        
    return test_set, dicts


def changenum(sent):
	sent = re.sub(r"\d+\.?\d*", "0", sent)		
	sent = re.sub(r"[a-zA-Z]+", "1", sent)
	return sent

if __name__ == "__main__":

    cfg = Config()
    # Generate n-grame parameters
    param_file = 'rmrb_ngram_changed.json'
    if os.path.exists(param_file):
        f = open(param_file, 'r', encoding='utf-8')
        params = json.load(f)
        f.close()
    else:
        params = get_ngram_prob()
        f = open(param_file, 'w', encoding='utf-8')
        json.dump(params, f)
        f.close()
    test_targets, dicts = get_test_sets()

    # Build an HMM model
    model = HMM(params['p2'], params['p1'])
    results = []
    for sen in tqdm(test_targets[:10]):
        # Get candidates
        ori_sen = ''.join(sen)
        print('input : ', ori_sen)
        pro_st_time = time.time()
        nums, words, cands = get_proposals(ori_sen, dicts, cfg)
        print('pro_time: ', time.time() - pro_st_time)
        print(len(cands), 'cand gets')

        # Calculate Score for each candidates
        for_st_time = time.time()
        scores = list()
        for cand in cands[:10]:
            scores.append(model.calc_prob(cand.split(' ')))
        print('pro_time: ', time.time() - for_st_time)
        idx = np.argmax(scores)
        results.append(cands[idx].split(' '))
    
    test_targets = [[changenum(word) for word in sen] for sen in test_targets]
    evaluateSet(results, test_targets)


    '''
    model_rmrb = HMM_word(params['p3'], '<BOS>', '<EOS>')
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = model_rmrb.find(ori_line)
        results.append(res)
    evaluateSet(results, lines)
    '''