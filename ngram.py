import numpy as np 
import os
import re
from dict import Dict
from config import Config



def pre_process(ustring, use_re = 1):
	rstr = ""
	for uchar in ustring:
		unic=ord(uchar)
		if unic == 12288:
			unic = 32
		elif (65296 <= unic <= 65305) or (65345 <= unic <= 65370) or (65313 <= unic <= 65338):
			unic -= 65248
		rstr += chr(unic)
	
	# 若有，去掉开头的日期
	p = rstr.find(' ')
	if p != -1 and rstr[:p-2].replace('-','').isdigit() == True:
		rstr = rstr[p+1:].lstrip()

	if use_re == 1:
		# 所有数字改为 0
		rstr = re.sub(r"\d+\.?\d*", "0", rstr)		
		# 所有英文单词改为 1
		rstr = re.sub(r"[a-zA-Z]+\/", "1/", rstr)


	# 实体名词去掉注释
	rstr = rstr.replace('[','')    
	rstr = rstr.replace(']nt', '')
	rstr = rstr.replace(']ns','')
	rstr = rstr.replace(']nz','')
	rstr = rstr.replace(']l','')
	rstr = rstr.replace(']i','')
	
	return rstr

def readfile(cfg):
	'''
	return:
		sp_w = ['敲', '键盘',]
		sp_t = ['x', 'x']
	'''
	if os.path.exists(cfg.modified_train_set):
		f = open(cfg.modified_train_set,'r', encoding='utf-8')		
		lines = f.readlines()
	else:
		f = open(cfg.train_set,'r', encoding='utf-8')
		lines = f.readlines()		
		lines = [pre_process(l, cfg.use_re) for l in lines]
		fw = open(cfg.modified_train_set, 'w',encoding="utf-8")
		for l in lines:
			fw.write(l)
	
	super_line_w, super_line_t = [], []
	for l in lines:
		l = l.split()
		super_line_w.append('<BOS>')
		super_line_t.append('bos')
		for word in l:
			word = word.split('/')
			super_line_w.append(word[0])
			super_line_t.append(word[-1])
		super_line_w.append('<EOS>')
		super_line_t.append('eos')
	
	return {
		"w": super_line_w,
		"t": super_line_t, 
	}

def add_value_on_dict(dct, ky, x):
	if ky in dct.keys():
		dct[ky] += x
	else :
		dct[ky] = x

def calc_prob_fun(superline, cfg):
	N = cfg.ngram
	p1 = {} # p(w|t) 
	p2 = {} # p(t|t,..,t)
	p3 = {} # p(w|w,...,w)

	t_num1 = {} # number of each ti 
	w_num1 = {} # number of each wi
	
	t_numN = {}
	w_numN = {}
	
	t_numN_ = {}
	w_numN_ = {}
	
	super_line_w = superline['w']
	super_line_t = superline['t']

	#calc t/w_numN
	for i in range(0, len(super_line_t)-N+1):
		#[i,...,j]
		j = i + N - 1 
		tmp_w = " ".join(super_line_w[i:j+1])
		tmp_t = " ".join(super_line_t[i:j+1])
		if tmp_w.find("<BOS> <EOS>") != -1 :
			continue
		add_value_on_dict(t_numN, tmp_t, 1)
		add_value_on_dict(w_numN, tmp_w, 1)
	
	#calc t/w_numN_
	for i in range(0, len(super_line_t)-(N-1)+1):
		#[i,...,j]
		j = i + N-1 - 1 
		tmp_w = " ".join(super_line_w[i:j+1])
		tmp_t = " ".join(super_line_t[i:j+1])
		if tmp_w.find("<BOS> <EOS>") != -1 :
			continue
		add_value_on_dict(t_numN_, tmp_t, 1)
		add_value_on_dict(w_numN_, tmp_w, 1)

	# calc t/w_num1
	for i in range(0, len(super_line_w)):
		add_value_on_dict(t_num1, super_line_t[i], 1)
		add_value_on_dict(w_num1, super_line_w[i], 1)

	# calc p1: p(w|t) 	
	for t in t_num1.keys():
		for i in range(len(super_line_w)):
			if super_line_t[i] == t:
				key = super_line_w[i] + ' ' + t
				add_value_on_dict(p1, key, 1 / t_num1[t])

	# calc p2: p(t|t,..,t)
	for keysN in t_numN.keys():
		tmp = keysN.split()
		keysN_ = " ".join(tmp[:-1])
		p2[keysN] = t_numN[keysN] / t_numN_[keysN_]

	# calc p3: p(w|w,..,w)
	for keysN in w_numN.keys():
		tmp = keysN.split()
		keysN_ = " ".join(tmp[:-1])
		p3[keysN] = w_numN[keysN] / w_numN_[keysN_]

	return {
		"p1": p1,
		"p2": p2,
		"p3": p3,
	}

def get_ngram_prob(cfg):
	lines = readfile(cfg)
	return calc_prob_fun(lines, cfg)



if __name__ == '__main__':
	cfg = Config()
	superline = readfile(cfg)
	# fs = get_prob_fun(superline, cfg)
	dict = Dict(["中共","总书记"],"set")
	# s = "迈向，，，充满123希望的word新世纪，一九九八新年讲话。"
	# s = "刚刚看到的一段话：“你特别烦的时候先保持冷静或者看一部开心的电影者喝一大杯水不要试图跟朋友聊天朋友是跟你分享快乐的人而不是分享你痛苦的人不要做一个唠唠叨叨的抱怨者从现在起要学会自己去化解去承受”送给和我一样最近有点烦闷的人"
	s = "中共中央总书记、国家主席江泽民"
	# s  = "你好吗"
	digit, english, pro = get_proposals(s, dict, cfg)
	print(pro)