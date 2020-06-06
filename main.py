import numpy as np
import json

from ngram import get_ngram_prob, get_proposals, Config
from hmm import HMM2 as HMM, HMM_word
from evaluation import evaluateSet

from tqdm import tqdm
import time
import re

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
            idata = line.strip().split()
            if len(idata) == 0:
                continue
            idata = [x.split('/')[0] for x in idata]
            test_set.append(idata)
            dicts.extend(idata)

    dicts = list(set(dicts))        
    return test_set, dicts

def restore(sentence, nums, words):
    while len(words) > 0:
        ik = sentence.find('1')
        sentence = sentence

def changenum(sent):
	digit = re.findall(r"\d+\.?\d*",sent)
	english = r = re.findall(r"[a-zA-Z]+",sent)
	for d in digit:
		sent = sent.replace(d, '0')
	for e in english:
		sent = sent.replace(e, '1')
	return sent

if __name__ == "__main__":
    # load test set
    nlpcc_f = open('NLPCC-WordSeg-Weibo/datasets/nlpcc2016-wordseg-dev.dat', 'r', encoding='utf-8')
    lines = nlpcc_f.readlines()
    lines = [changenum(line) for line in lines]
    lines = [line.strip().split() for line in lines]
    nlpcc_f.close()

    cfg = Config()
    params = get_ngram_prob()
    #str_param = json.dumps(params)
    #with open('dict_v03.json', 'w', encoding='utf-8') as dict_file:
    #    dict_file.write(str_param)
    test_targets, dicts = get_test_sets()

    #
    model = HMM(params['p2'], params['p1'])

    results = []
    for sen in tqdm(lines[:30]):
        ori_sen = ''.join(sen)
        print('input : ', ori_sen)
        pro_st_time = time.time()
        nums, words, cands = get_proposals(ori_sen, dicts, cfg)
        print('pro_time: ', time.time() - pro_st_time)
        print(len(cands), 'cand gets')
        scores = list()
        for_st_time = time.time()
        for cand in cands[:10]:
            scores.append(model.calc_prob(cand.split(' ')))
            #print(len(cand.split(' ')), scores[-1])
        print('pro_time: ', time.time() - for_st_time)
        idx = np.argmax(scores)
        results.append(cands[idx].split(' '))
        print('result: ', cands[idx].split(' '))
    evaluateSet(results, lines)


    # load test set
    nlpcc_f = open('NLPCC-WordSeg-Weibo/datasets/nlpcc2016-wordseg-dev.dat', 'r', encoding='utf-8')
    lines = nlpcc_f.readlines()
    lines = [changenum(line) for line in lines]
    lines = [line.strip().split() for line in lines]
    nlpcc_f.close()
    ### weibo 1
    filename = 'viterbi-tokenizer-master\\nlpcc_train.mod-2gram'
    with open(filename, 'r', encoding='utf-8') as f:
        dict_lines = f.readlines()
        dict_lines = [l.strip().split('\t') for l in dict_lines]
        probs = {}
        for l in dict_lines:
            if len(l)<2:
                continue
            probs[l[0]] = float(l[1])
    model_weibo_train = HMM_word(probs)

    results = []
    for line in tqdm(lines[:30]):
        ori_line = ''.join(line)
        res = model_weibo_train.find(ori_line)
        results.append(res)
    evaluateSet(results, lines)
    '''
    model_rmrb = HMM_word(params['p3'], '<BOS>', '<EOS>')
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = model_rmrb.find(ori_line)
        results.append(res)
    evaluateSet(results, lines)
    '''