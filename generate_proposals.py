import heapq
import itertools
import re

def dfs(e, u, k, dis, nxt, vis):
	n = len(e)
	if vis[u] == True: return
	vis[u] = True
	if u == n - 1:
		dis[u][0] = 0
		return
	
	q = []
	for v in range(u+1, n):
		if e[u][v] == 1:
			if vis[v] == False: dfs(e,v,k,dis,nxt,vis)
			for j in range(k):
				if dis[v][j] == 10000: break
				if len(q) < k:
					heapq.heappush(q, (-(dis[v][j] + 1), (v, j)))
				elif -dis[v][j]-1 > q[0][0]:
					_ = heapq.heappop(q)
					heapq.heappush(q, (-(dis[v][j] + 1), (v, j)))
	i = len(q)-1
	while q:
		d, (v, j) = heapq.heappop(q)
		d = -d 
		dis[u][i] = d 
		nxt[u][i][0] = v
		nxt[u][i][1] = j
		i -= 1

def get_path(u, k, nxt, p):
	p.append(u)
	# print(u, end=" ")
	if u == len(nxt)-1: return
	get_path(nxt[u][k][0], nxt[u][k][1], nxt, p)

def get_all_path(u,K,dis,nxt):
	props = []
	for i in range(K):
		if dis[u][i] == 10000: break
		p = []
		get_path(u,i,nxt,p)
		props.append(p)
	return props

def gene_proposal(sent, dict, cfg, data_structure):
	n = len(sent)
	if n == 0: return []
	e = [[0]*(n+1) for i in range(n+1)]
	for i in range(n):
		e[i][i+1] = 1
	if data_structure == "set":	
		for i in range(n-1):
			for j in range(i+1, n):
				if j-i+1 > cfg.word_max_len: break
				word = sent[i:j+1]
				if word in dict:
					e[i][j+1] = 1
	elif data_structure == "ac":
		have = list(dict.iter(sent))
		for end, (_, word) in have:
			e[end-len(word)+1][end+1] = 1
	
	vis = [False] * (n+1)
	dis = [[10000] * (cfg.subseq_num) for i in range(n+1)]
	nxt = [[[-1] * 2 for i in range(cfg.subseq_num)] for j in range(n+1)]
	dfs(e, 0, cfg.subseq_num, dis, nxt, vis)

	proposals = get_all_path(0, cfg.subseq_num, dis, nxt)

	ret = []
	for p in proposals:
		tmp = ""
		for i in range(len(p)-1):			
			tmp += sent[p[i]:p[i+1]] + ' '
		tmp = tmp[:-1]
		ret.append(tmp)
	return ret

def union(old, new):
	if len(old) == 0: return new 
	old =  list(itertools.product(old, new))
	for i in range(len(old)):
		old[i] = old[i][0] + ' ' + old[i][1]
	
	old.sort(key=lambda x: len(x.split()))
	
	if len(old) >= 10:
		return old[:10]
	else:
		return old

def get_proposals(sent, dict, cfg):
	sent = sent.replace(' ', '')
	digit = re.findall(r"\d+\.?\d*",sent)
	english = r = re.findall(r"[a-zA-Z]+",sent)

	sent = re.sub(r"\d+\.?\d*", "0", sent)		
	sent = re.sub(r"[a-zA-Z]+", "1", sent)


	chinese_punc = set(["，", "。", "？", "！", "；", "《", "》", "“", "”", "、"])
	props = []
	
	last = 0
	for i in range(len(sent)):
		if sent[i] in chinese_punc or i == len(sent)-1:
			tmp = gene_proposal(sent[last:i+1], dict.A, cfg, dict.data_structure)
			last = i+1
			props = union(props, tmp)

	props.sort(key=lambda x: len(x.split()))

	return digit, english, props