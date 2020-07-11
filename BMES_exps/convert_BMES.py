import re
import sys

path_from = "BMES_corpus/msr_training.utf8"
path_to = "BMES_corpus/msr_BMES_nonum.txt"

f = open(path_from, 'r', encoding='utf-8')
g = open(path_to, 'w', encoding='utf-8')

for line in f:
	rstr = ""
	for uchar in line:
		unic=ord(uchar)
		if unic == 12288:
			unic = 32
		elif (65296 <= unic <= 65305) or (65345 <= unic <= 65370) or (65313 <= unic <= 65338):
			unic -= 65248
		rstr += chr(unic)
	rstr = re.sub(r"\d+\.?\d*", "0", rstr)		
	rstr = re.sub(r"[a-zA-Z]+", "1", rstr)

	l = rstr.split()
	for x in l:
		if len(x) == 1:
			g.write(x[0] + " " + "S\n")
		else:
			g.write(x[0] + " " + "B\n")
			for y in x[1:-1]:
				g.write(y + " " + "M\n")
			g.write(x[-1] + " " + "E\n")
	g.write("\n")