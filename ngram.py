import numpy as np 
import os

class Config:
	ngram = 2
	file = "rmrb.txt"
	modi_file = "rmrb_modified.txt"

def pre_process(ustring):
	rstring = ""
	for uchar in ustring:
		inside_code=ord(uchar)
		if inside_code == 12288:     
			inside_code = 32 
		elif (inside_code >= 65281 and inside_code <= 65374):
			inside_code -= 65248

		rstring += chr(inside_code)
	rstring = rstring.replace('[','')    
	rstring = rstring.replace(']nt', '')
	rstring = rstring.replace(']ns','')
	rstring = rstring.replace(']nz','')
	rstring = rstring.replace(']l','')
	rstring = rstring.replace(']i','')
	
	return rstring


def readfile(cfg):
	'''
	return:
		[
			[['敲', '/x'], ['键盘', '/x']],
			[['你', '/x'], ['好', '/x'], ['吗', '/x']],
		]
		这个结构就是万恶之首，设计得非常不好，superline比较好写
		不知道后续是否有用，先保留着这个结构吧
	'''
	if os.path.exists(cfg.modi_file):
		f = open(cfg.modi_file,'r', encoding='utf-8')		
		lines = f.readlines()
	else:
		f = open(cfg.file,'r', encoding='utf-8')
		lines = f.readlines()		
		lines = [pre_process(l) for l in lines]
		fw = open(cfg.modi_file, 'w',encoding="utf-8")
		for l in lines:
			fw.write(l)
	
	lines = [l.split() for l in lines]
	for i in range(len(lines)):
		lines[i] = [j.split('/') for j in lines[i]]
	return lines


def add_one_on_dict(dct, ky):
	if ky in dct.keys():
		dct[ky] += 1
	else :
		dct[ky] = 1


def get_prob_fun(lines, cfg):
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
	
	# merge all sentence into one sentence
	super_line_w = [] 
	super_line_t = [] 
	for l in lines:
		super_line_w.append('<BOS>')
		super_line_t.append('bos')
		for i in l:
			super_line_w.append(i[0])
			super_line_t.append(i[-1])
		super_line_w.append('<EOS>')
		super_line_t.append('eos')


	#calc t/w_numN
	for i in range(0, len(super_line_t)-N+1):
		#[i,...,j]
		j = i + N - 1 
		tmp_w = " ".join(super_line_w[i:j+1])
		tmp_t = " ".join(super_line_t[i:j+1])
		if tmp_w.find("<BOS> <EOS>") != -1 :
			continue
		add_one_on_dict(t_numN, tmp_t)
		add_one_on_dict(w_numN, tmp_w)
	
	#calc t/w_numN_
	for i in range(0, len(super_line_t)-(N-1)+1):
		#[i,...,j]
		j = i + N-1 - 1 
		tmp_w = " ".join(super_line_w[i:j+1])
		tmp_t = " ".join(super_line_t[i:j+1])
		if tmp_w.find("<BOS> <EOS>") != -1 :
			continue
		add_one_on_dict(t_numN_, tmp_t)
		add_one_on_dict(w_numN_, tmp_w)

	# calc t/w_num1
	for i in range(0, len(super_line_w)):
		add_one_on_dict(t_num1, super_line_t[i])
		add_one_on_dict(w_num1, super_line_w[i])

	# calc p1: p(w|t) 
	for t in t_num1.keys():
		for l in lines:
			for i in l:
				if i[-1] == t:
					key = i[0] + ' ' + t
					if key in p1.keys():
						p1[key] += 1 / t_num1[t]
					else:
						p1[key] = 1 / t_num1[t]
	
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



def dfs(e, cnt, ans, pro):
	if cnt == len(e)-1:
		pro.append(ans)
		return
	for i in range(cnt+1, len(e)):
		if e[cnt][i] == 1:
			dfs(e, i, ans+str(i)+" ", pro)

def gene_proposal(sent, dict_set):
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
	e = [[0]*(n+1) for i in range(n+1)]
	for i in range(n):
		e[i][i+1] = 1
	for i in range(n-1):
		for j in range(i+1, n):
			word = sent[i:j+1]
			if word in dict_set:
				e[i][j+1] = 1 				
	proposals = []
	dfs(e,0,"", proposals)
	ret = []
	for p in proposals:
		p = p.split()
		last = 0
		tmp = ""
		for i in range(len(p)):
			tmp += sent[last: int(p[i])] + " "
			last = int(p[i])
		tmp = tmp[:-1]
		ret.append(tmp)
	return ret


if __name__ == '__main__':
	# cfg = Config()
	# lines = readfile(cfg)
	# fs = get_prob_fun(lines, cfg)
	dict_set = set()
	dict_set.add("充满希望")
	dict_set.add("希望的新世纪")
	s = "迈向充满希望的新世纪一九九八新年讲话"
	pro = gene_proposal(s, dict_set)
	print(pro)

