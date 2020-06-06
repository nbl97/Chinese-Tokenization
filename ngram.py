import numpy as np 
import os
import itertools
import re


class Config:
	ngram = 2
	ori_file = "rmrb.txt"
	modi_file = "rmrb_modified.txt"
	word_max_len = 10
	proposals_keep_ratio = 1.0
	use_re = 1

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
	if os.path.exists(cfg.modi_file):
		f = open(cfg.modi_file,'r', encoding='utf-8')		
		lines = f.readlines()
	else:
		f = open(cfg.ori_file,'r', encoding='utf-8')
		lines = f.readlines()		
		lines = [pre_process(l, cfg.use_re) for l in lines]
		fw = open(cfg.modi_file, 'w',encoding="utf-8")
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


def get_prob_fun(superline, cfg):
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


def get_ngram_prob():
	cfg = Config()
	lines = readfile(cfg)
	return get_prob_fun(lines, cfg)


def dfs(e, cnt, ans, pro):
	if cnt == len(e)-1:
		pro.append(ans)
		return
	for i in range(cnt+1, len(e)):
		if e[cnt][i] == 1:
			dfs(e, i, ans+str(i)+" ", pro)


def gene_proposal(sent, dict_set, cfg):
	'''
	sent: str
	dict_set: a set containing all words in the dictionary
	return:
		[
			"a b c d",
			"ab cd",
			"a bc d",
		]
	'''
	n = len(sent)
	if n == 0: return []
	e = [[0]*(n+1) for i in range(n+1)]
	for i in range(n):
		e[i][i+1] = 1
	for i in range(n-1):
		for j in range(i+1, n):
			if j-i+1 > cfg.word_max_len: break
			word = sent[i:j+1]
			if word in dict_set:
				e[i][j+1] = 1
	proposals = []
	dfs(e,0,"0 ", proposals)
	
	ret = []
	for p in proposals:
		p = p.split()
		tmp = ""
		for i in range(len(p)-1):
			tmp += sent[int(p[i]):int(p[i+1])] + ' '			
		tmp = tmp[:-1]
		ret.append(tmp)
	return ret


def union(old, new):
	if len(old) == 0: return new 
	old =  list(itertools.product(old, new))
	for i in range(len(old)):
		old[i] = old[i][0] + ' ' + old[i][1]
	old.sort(key=lambda x: len(x.split()))
	return old[:10]


def get_proposals(sent, dict_set, cfg):
	sent = sent.replace(' ', '')
	digit = re.findall(r"\d+\.?\d*",sent)
	english = r = re.findall(r"[a-zA-Z]+",sent)

	sent = re.sub(r"\d+\.?\d*", "0", sent)		
	sent = re.sub(r"[a-zA-Z]+", "1", sent)


	chinese_punc = set(["，", "。", "？", "！", "；", "《", "》", "“", "”"])
	props = []
	
	last = 0
	for i in range(len(sent)):
		if sent[i] in chinese_punc or i == len(sent)-1:
			tmp = gene_proposal(sent[last:i+1], dict_set, cfg)
			last = i+1
			props = union(props, tmp)

	props.sort(key=lambda x: len(x.split()))

	return digit, english, props


if __name__ == '__main__':
	cfg = Config()
	superline = readfile(cfg)
	# fs = get_prob_fun(superline, cfg)
	# dict_set = set()
	# dict_set.add("充满希望")
	# dict_set.add("希望的新世纪")
	# dict_set.add("新世纪")

	# s = "迈向，，，充满123希望的word新世纪，一九九八新年讲话。"
	# digit, english, pro = get_proposals(s, dict_set, cfg)
	# print(pro)