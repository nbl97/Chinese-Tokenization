import numpy as np
import json

from ngram import get_ngram_prob, get_proposals, Config, pre_process
from hmm_new import HMM2 as HMM, HMM_word
from evaluation import evaluateSet
from Dict import Dict 

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
    cfg.use_re = 1
    cfg.use_hmm = 1
    # Generate n-grame parameters
    param_file = 'rmrb_ngram_changed.json' if cfg.use_re else 'rmrb_ngram_nochanged.json'
    if os.path.exists(param_file):
        f = open(param_file, 'r', encoding='utf-8')
        params = json.load(f)
        f.close()
    else:
        params = get_ngram_prob(cfg)
        f = open(param_file, 'w', encoding='utf-8')
        json.dump(params, f)
        f.close()
    test_targets, dicts = get_test_sets()

    dicts = Dict(dicts, data_structure="set") # or "set"

    # Simple 2-gram model from rmrb-train
    model_rmrb = HMM_word(params['p3'], '<BOS>', '<EOS>')

    # Build an HMM model
    model = HMM(params['p2'], params['p1'])
    results = []
    
    for sen in tqdm(test_targets[:10000:100]):
        # Get candidates
        ori_sen = ''.join(sen)
        nums, words, cands = get_proposals(ori_sen, dicts, cfg)

        # Calculate Score for each candidates
        if cfg.use_hmm:
            scores = list()
            for cand in cands[:10]:
                scores.append(model_rmrb.calc_prob(cand.split(' ')))
            idx = np.argmax(scores)
        else:
            idx = 0
        results.append(cands[idx].split(' '))
    if cfg.use_re:
        test_targets = [[changenum(word) for word in sen] for sen in test_targets]
    evaluateSet(results, test_targets[:10000:100])

    #exit()
    
    # load test set
    nlpcc_f = open('train_data/nlpcc2016-wordseg-dev.dat', 'r', encoding='utf-8')
    ori_lines = nlpcc_f.readlines()
    lines_wochange = [line.strip().split() for line in ori_lines]
    nlpcc_f.close()

    results = []
    for sen in tqdm(lines_wochange[:]):
        # Get candidates
        ori_sen = ''.join(sen)
        nums, words, cands = get_proposals(ori_sen, dicts, cfg)

        # Calculate Score for each candidates
        for_st_time = time.time()
        if cfg.use_hmm:
            scores = list()
            for cand in cands[:10]:
                scores.append(model_rmrb.calc_prob(cand.split(' ')))
            idx = np.argmax(scores)
        else:
            idx = 0
        results.append(cands[idx].split(' '))
    if cfg.use_re:
        lines_wochange = [[changenum(word) for word in sen] for sen in lines_wochange]
    evaluateSet(results, lines_wochange[:])
