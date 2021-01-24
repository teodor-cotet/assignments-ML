import subprocess

def construct_class_file(nr_class, file_name):

	fout = open(file_name + str(nr_class), 'w')

	with open(file_name) as fin:
		for line in fin:
			if line[0] == str(nr_class):
				line = str(nr_class) + line[1:]
			else:
				line = "10" + line[1:]
			fout.write(line)

if __name__ == "__main__":

	for i in range(10):
		construct_class_file(i, 'pendigits')
		subprocess.run(["./svm-train", "-b", "1", "-t", "1", "-d", "3", "-g",  "0.0001",   "pendigits" + str(i)])
		subprocess.run(["./svm-predict", "-b", "1", "pendigits.t", "pendigits" + str(i) + ".model", 'out' + str(i)])
	predicts = {}
	
	for i in range(10):
		jj = -1

		with open("out" + str(i)) as fin:
			for line in fin:
				jj += 1
				if jj == 0:
					print(line)
					continue
				L = line.split(' ')
				L[0] = L[0].strip('\n')
				L[1] = L[1].strip('\n')
				L[2] = L[2].strip('\n')

				if i != 8:
					neg = float(L[1])
					pos = float(L[2])
				else:
					neg = float(L[2])
					pos = float(L[1])

				p = int(L[0])
				#print(L, pos)
				
				if jj not in predicts:
					predicts[jj] = (i, pos)

				if predicts[jj][1] < pos:
					predicts[jj] = (i, pos)
	
	correct = 0
	total = 0
	with open("pendigits.t") as fin:
		jj = 0
		for line in fin:
			p = int(line[0])
			jj += 1
			if p == predicts[jj][0]:
				total += 1
				correct += 1
			else:
				total += 1
	print(correct / total)




