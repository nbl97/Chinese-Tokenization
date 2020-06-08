import numpy as np
import json

from ngram import get_ngram_prob, pre_process
from generate_proposals import get_proposals
from config import Config
from hmm_new import HMM_word
from evaluation import evaluateSet

from tqdm import tqdm
import time
import re

def changenum(sent):
	sent = re.sub(r"\d+\.?\d*", "0", sent)		
	sent = re.sub(r"[a-zA-Z]+", "1", sent)
	return sent

from test_exp1 import get_test_sets, changenum
from dict import Dict

def test_dev():
    # load test set
    nlpcc_f = open('data/nlpcc2016-wordseg-dev.dat', 'r', encoding='utf-8')
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
    nlpcc_f = open('data/nlpcc2016-wordseg-dev.dat', 'r', encoding='utf-8')
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
    cfg = Config()
    cfg.use_re = 0
    params = get_ngram_prob(cfg)

    # load test set
    cfg.test_set = 'data/nlpcc2016-word-seg-train.dat'
    nlpcc_f = open(cfg.test_set, 'r', encoding='utf-8')
    ori_lines = nlpcc_f.readlines()
    lines = [changenum(line) for line in ori_lines]
    lines_wchange = [line.strip().split() for line in lines]
    lines_wochange = [line.strip().split() for line in ori_lines]
    nlpcc_f.close()

    # Simple n-gram model from weibo-train
    print("Simple 2-gram model from nlpcc, with re-match")
    # filename = 'weibo_model\\nlpcc_train.replace-2gram'
    cfg.model_file = 'weibo_model\\nlpcc_train.replace-2gram'
    with open(cfg.model_file, 'r', encoding='utf-8') as f:
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
    cfg.model_file = 'weibo_model\\nlpcc_train.mod-2gram'
    # filename = 'weibo_model\\nlpcc_train.mod-2gram'
    with open(cfg.model_file, 'r', encoding='utf-8') as f:
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

    
    # Simple 2-gram model from rmrb-train
    print("Simple 2-gram model from rmrb, without re-match")
    model_rmrb = HMM_word(params['p3'], '<BOS>', '<EOS>')
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = model_rmrb.find(ori_line)
        results.append(res)
    evaluateSet(results, lines)

    
    # cfg = Config()
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