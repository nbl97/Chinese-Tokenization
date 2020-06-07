import numpy as np
import json

from ngram import get_ngram_prob, get_proposals, Config, pre_process
from hmm import HMM2 as HMM, HMM_word
from evaluation import evaluateSet

from tqdm import tqdm
import time
import re

def changenum(sent):
	sent = re.sub(r"\d+\.?\d*", "0", sent)		
	sent = re.sub(r"[a-zA-Z]+", "1", sent)
	return sent

from test_rmrb import get_test_sets, changenum
from Dict import Dict

def test_dev():
    # load test set
    nlpcc_f = open('NLPCC-WordSeg-Weibo/datasets/nlpcc2016-wordseg-dev.dat', 'r', encoding='utf-8')
    lines = nlpcc_f.readlines()
    lines = [changenum(line) for line in lines]
    lines = [line.strip().split() for line in lines]
    nlpcc_f.close()

    
    # get dict from rmrb
    _, dicts = get_test_sets()
    dicts = Dict(dicts, data_structure="ac") # or "set"

    # model from rmrb
    cfg = Config()
    params = get_ngram_prob(cfg)
    
    '''
    model = HMM(params['p2'], params['p1'])

    # test with proposal and HMM model
    results = []
    for sen in tqdm(lines[:]):
        ori_sen = ''.join(sen)
        print(ori_sen)
        #print('input : ', ori_sen)
        #pro_st_time = time.time()
        nums, words, cands = get_proposals(ori_sen, dicts, cfg)
        #print('pro_time: ', time.time() - pro_st_time)
        #print(len(cands), 'cand gets')

        #scores = list()
        #for_st_time = time.time()
        #for cand in cands[:10]:
        #    scores.append(model.calc_prob(cand.split(' ')))
        #    #print(len(cand.split(' ')), scores[-1])
        #print('pro_time: ', time.time() - for_st_time)
        #idx = np.argmax(scores)

        results.append(cands[0].split(' '))
    evaluateSet(results, lines)
    '''
    
    print("Simple 2-gram model from rmrb, with re-match")
    # Simple 2-gram model from rmrb-train
    model_rmrb = HMM_word(params['p3'], '<BOS>', '<EOS>')
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = model_rmrb.find(ori_line)
        results.append(res)
    evaluateSet(results, lines)


    # Simple n-gram model from weibo-train
    print("Simple 2-gram model from nlpcc, with re-match")
    filename = 'weibo_model\\nlpcc_train.replace-2gram'
    with open(filename, 'r', encoding='utf-8') as f:
        dict_lines = f.readlines()
        dict_lines = [l.strip().split('\t') for l in dict_lines]
        probs = {}
        for l in dict_lines:
            if len(l)<2:
                continue
            probs[l[0]] = float(l[1])
    model_weibo_train = HMM_word(probs)
    # Test with Simple 2-gram model
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = model_weibo_train.find(ori_line)
        results.append(res)
    evaluateSet(results, lines)



    # load test set without number and english replace
    nlpcc_f = open('NLPCC-WordSeg-Weibo/datasets/nlpcc2016-wordseg-dev.dat', 'r', encoding='utf-8')
    lines = nlpcc_f.readlines()
    lines = [line.strip().split() for line in lines]
    nlpcc_f.close()
    # Simple n-gram model from weibo-train
    print("Simple 2-gram model from nlpcc, without re-match")
    filename = 'weibo_model\\nlpcc_train.mod-2gram'
    with open(filename, 'r', encoding='utf-8') as f:
        dict_lines = f.readlines()
        dict_lines = [l.strip().split('\t') for l in dict_lines]
        probs = {}
        for l in dict_lines:
            if len(l)<2:
                continue
            probs[l[0]] = float(l[1])
    model_weibo_train = HMM_word(probs)

    # Test with Simple 2-gram model
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = model_weibo_train.find(ori_line)
        results.append(res)
    evaluateSet(results, lines)

    
    # model from rmrb
    cfg = Config()
    cfg.use_re = 0
    params = get_ngram_prob(cfg)
    
    # Simple 2-gram model from rmrb-train
    print("Simple 2-gram model from rmrb, without re-match")
    model_rmrb = HMM_word(params['p3'], '<BOS>', '<EOS>')
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = model_rmrb.find(ori_line)
        results.append(res)
    evaluateSet(results, lines)

def test_train():

    # load test set
    nlpcc_f = open('NLPCC-WordSeg-Weibo/datasets/nlpcc2016-word-seg-train.dat', 'r', encoding='utf-8')
    ori_lines = nlpcc_f.readlines()
    lines = [changenum(line) for line in ori_lines]
    lines_wchange = [line.strip().split() for line in lines]
    lines_wochange = [line.strip().split() for line in ori_lines]
    nlpcc_f.close()

    # Simple n-gram model from weibo-train
    print("Simple 2-gram model from nlpcc, with re-match")
    filename = 'weibo_model\\nlpcc_train.replace-2gram'
    with open(filename, 'r', encoding='utf-8') as f:
        dict_lines = f.readlines()
        dict_lines = [l.strip().split('\t') for l in dict_lines]
        probs = {}
        for l in dict_lines:
            if len(l)<2:
                continue
            probs[l[0]] = float(l[1])
    model_weibo_train = HMM_word(probs)
    # Test with Simple 2-gram model
    results = []
    for line in tqdm(lines_wchange):
        ori_line = ''.join(line)
        res = model_weibo_train.find(ori_line)
        results.append(res)
    evaluateSet(results, lines_wchange)

    
    # Simple n-gram model from weibo-train
    print("Simple 2-gram model from nlpcc, without re-match")
    filename = 'weibo_model\\nlpcc_train.mod-2gram'
    with open(filename, 'r', encoding='utf-8') as f:
        dict_lines = f.readlines()
        dict_lines = [l.strip().split('\t') for l in dict_lines]
        probs = {}
        for l in dict_lines:
            if len(l)<2:
                continue
            probs[l[0]] = float(l[1])
    model_weibo_train = HMM_word(probs)
    # Test with Simple 2-gram model
    results = []
    for line in tqdm(lines_wochange):
        ori_line = ''.join(line)
        res = model_weibo_train.find(ori_line)
        results.append(res)
    evaluateSet(results, lines_wochange)


    # start rmrb test

    # model from rmrb without re
    lines, _ = get_test_sets()

    cfg = Config()
    cfg.use_re = 0
    params = get_ngram_prob(cfg)
    
    # Simple 2-gram model from rmrb-train
    print("Simple 2-gram model from rmrb, without re-match")
    model_rmrb = HMM_word(params['p3'], '<BOS>', '<EOS>')
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = model_rmrb.find(ori_line)
        results.append(res)
    evaluateSet(results, lines)

    
    cfg = Config()
    cfg.use_re = 1
    params = get_ngram_prob(cfg)
    
    lines = [[changenum(word) for word in line] for line in lines]
    
    # Simple 2-gram model from rmrb-train
    print("Simple 2-gram model from rmrb, with re-match")
    model_rmrb = HMM_word(params['p3'], '<BOS>', '<EOS>')
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = model_rmrb.find(ori_line)
        results.append(res)
    evaluateSet(results, lines)

if __name__ == "__main__":
    test_train()