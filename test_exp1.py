import argparse
import numpy as np
import json

from ngram import get_ngram_prob, pre_process
from generate_proposals import get_proposals
from config import Config
from hmm_new import HMM2 as HMM, HMM_word
from evaluation import evaluateSet
from dict import Dict 

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
    with open('data/rmrb.txt', 'r', encoding='utf-8') as f:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-re', action='store_true', default=False, help='use re-replacement or not')
    parser.add_argument('--score', type=str, default="None", help='decide if use score function')
    args = parser.parse_args()
    print("Use Re: {}, Score Type:{}".format(args.use_re, args.score))
    assert args.score in ["None", "Markov", "HMM"], "score type must be chosen from None, Markov or HMM"


    cfg = Config()
    cfg.use_re = 1 if args.use_re else 0
    cfg.use_hmm = 1 if args.score != "None" else 0
    cfg.test_set = 'data/nlpcc2016-wordseg-dev.dat'
    cfg.param_file = 'data/rmrb_ngram_changed.json' if cfg.use_re else 'data/rmrb_ngram_nochanged.json'
    # Generate n-grame parameters
    # param_file = 'data/rmrb_ngram_changed.json' if cfg.use_re else 'data/rmrb_ngram_nochanged.json'
    print("Loading model parameters calculated from rmrb ... ")
    if os.path.exists(cfg.param_file):
        f = open(cfg.param_file, 'r', encoding='utf-8')
        params = json.load(f)
        f.close()
    else:
        params = get_ngram_prob(cfg)
        f = open(cfg.param_file, 'w', encoding='utf-8')
        json.dump(params, f)
        f.close()
    test_targets, dicts = get_test_sets()

    dicts = Dict(dicts, data_structure="set")

    # Simple 2-gram model from rmrb-train
    model_simple = HMM_word(params['p3'], '<BOS>', '<EOS>')

    # Build an HMM model
    model_hmm = HMM(params['p2'], params['p1'])
    results = []

    model_rmrb = model_simple if args.score == 'Markov' else model_hmm
    
    print("Test on rmrb train subset")
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
    print("Loading weibo dev set")
    nlpcc_f = open(cfg.test_set, 'r', encoding='utf-8')
    ori_lines = nlpcc_f.readlines()
    lines_wochange = [line.strip().split() for line in ori_lines]
    nlpcc_f.close()

    print("Test on nlpcc-dev")
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
