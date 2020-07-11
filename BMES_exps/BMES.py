from collections import Counter
from math import log
from tqdm import tqdm
import re

from evaluation import evaluateSet


def build_model(train_set):
    hmm_model = {i:Counter() for i in 'SBME'}
    trans = {'SS':0,
        'SB':0,
        'BM':0,
        'BE':0, 
        'MM':0,
        'ME':0,
        'ES':0,
        'EB':0
    }
    with open(train_set,'r',encoding='utf-8') as f:
        cha = []
        tag = []
        for l in f:
            l = l.split()
            if (len(l) == 0) :
                cha += " "
                tag += " "
            else:
                cha += l[0]
                tag += l[1]
        for i in range(len(tag)):
            if tag[i] != ' ':
                hmm_model[tag[i]][cha[i]] += int(1)
                if i+1<len(tag) and tag[i+1] != ' ':
                    trans[tag[i]+tag[i+1]] +=1
        s_ = trans['SS'] + trans['SB'] 
        trans['SS'] /= s_
        trans['SB'] /= s_

        b_ = trans['BM'] + trans['BE'] 
        trans['BM'] /= b_
        trans['BE'] /= b_

        m_ = trans['MM'] + trans['ME'] 
        trans['MM'] /= m_
        trans['ME'] /= m_

        e_ = trans['ES'] + trans['EB'] 
        trans['ES'] /= e_
        trans['EB'] /= e_

        log_total = {i:log(sum(hmm_model[i].values())) for i in 'SBME'}
        trans = {i:log(j) for i,j in trans.items()}
    return hmm_model, trans, log_total


def viterbi(nodes):
    paths = nodes[0]
    for l in range(1, len(nodes)):
        paths_ = paths
        paths = {}
        for i in nodes[l]:
            nows = {}
            for j in paths_:
                if j[-1]+i in trans:
                    nows[j+i]=paths_[j]+nodes[l][i]+trans[j[-1]+i]
            k = list(nows.values()).index(max(nows.values()))
            paths[list(nows.keys())[k]] = list(nows.values())[k]
    return list(paths.keys())[list(paths.values()).index(max(list(paths.values())))]

def hmm_cut(s):
    nodes = [{i:log(j[t]+1)-log_total[i] for i,j in hmm_model.items()} for t in s]
    tags = viterbi(nodes)
    words = [s[0]]
    for i in range(1, len(s)):
        if tags[i] in ['B', 'S']:
            words.append(s[i])
        else:
            words[-1] += s[i]
    return words


def changenum(ustring):
    rstr = ""
    for uchar in ustring:
        unic=ord(uchar)
        if unic == 12288:
            unic = 32
        elif (65296 <= unic <= 65305) or (65345 <= unic <= 65370) or (65313 <= unic <= 65338):
            unic -= 65248
        rstr += chr(unic)
    # 所有数字改为 0
    rstr = re.sub(r"\d+\.?\d*", "0", rstr)		
    # 所有英文单词改为 1
    rstr = re.sub(r"[a-zA-Z]+\/", "1/", rstr)
    return rstr


if __name__ == '__main__':
    
    print("Train Set: PKU; Test Set: Weibo, w/o re-replacement")
    hmm_model, trans, log_total = build_model("BMES_corpus/rmrb_BMES.txt")
    # load test set without number and english replace
    nlpcc_f = open('NLPCC-WordSeg-Weibo/datasets/nlpcc2016-wordseg-dev.dat', 'r', encoding='utf-8')
    lines = nlpcc_f.readlines()
    lines = [line.strip().split() for line in lines]
    nlpcc_f.close()

    # Test with Simple 2-gram model
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = hmm_cut(ori_line)
        results.append(res)
    evaluateSet(results, lines)


    
    print("Train Set: PKU; Test Set: Weibo, w/  re-replacement")
    hmm_model, trans, log_total = build_model("BMES_corpus/rmrb_BMES_nonum.txt")
    # load test set without number and english replace
    nlpcc_f = open('NLPCC-WordSeg-Weibo/datasets/nlpcc2016-wordseg-dev.dat', 'r', encoding='utf-8')
    lines = nlpcc_f.readlines()
    lines = [changenum(line) for line in lines]
    lines = [line.strip().split() for line in lines]
    nlpcc_f.close()

    # Test with Simple 2-gram model
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = hmm_cut(ori_line)
        results.append(res)
    evaluateSet(results, lines)


    
    print("Train Set: MSR; Test Set: PKU, w/  re-replacement")
    hmm_model, trans, log_total = build_model("BMES_corpus/msr_BMES_nonum.txt")

    # load test set without number and english replace
    nlpcc_f = open('NLPCC-WordSeg-Weibo/datasets/nlpcc2016-wordseg-dev.dat', 'r', encoding='utf-8')
    lines = nlpcc_f.readlines()
    lines = [changenum(line) for line in lines]
    lines = [line.strip().split() for line in lines]
    nlpcc_f.close()

    # Test with Simple 2-gram model
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = hmm_cut(ori_line)
        results.append(res)
    evaluateSet(results, lines)


    
    print("Train Set: PKU; Test Set: PKU, w/  re-replacement")
    hmm_model, trans, log_total = build_model("BMES_corpus/rmrb_BMES_nonum.txt")

    # load test set without number and english replace
    nlpcc_f = open('BMES_corpus/pku_training.utf8', 'r', encoding='utf-8')
    lines = nlpcc_f.readlines()
    lines = [changenum(line) for line in lines]
    lines = [line.strip().split() for line in lines]
    lines = [line for line in lines if len(line)]
    nlpcc_f.close()

    # Test with Simple 2-gram model
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = hmm_cut(ori_line)
        results.append(res)
    evaluateSet(results, lines)

    

    print("Train Set: MSR; Test Set: MSR, w/  re-replacement")
    hmm_model, trans, log_total = build_model("BMES_corpus/msr_BMES_nonum.txt")

    # load test set without number and english replace
    nlpcc_f = open('BMES_corpus/msr_training.utf8', 'r', encoding='utf-8')
    lines = nlpcc_f.readlines()
    lines = [changenum(line) for line in lines]
    lines = [line.strip().split() for line in lines]
    lines = [line for line in lines if len(line)]
    nlpcc_f.close()

    # Test with Simple 2-gram model
    results = []
    for line in tqdm(lines):
        ori_line = ''.join(line)
        res = hmm_cut(ori_line)
        results.append(res)
    evaluateSet(results, lines)