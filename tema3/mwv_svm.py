import subprocess

def construct_class_file(nr_class1, nr_class2, file_name):

	fout = open(file_name + str(nr_class1) + str(nr_class2), 'w')

	with open(file_name) as fin:
		for line in fin:
			if line[0] == str(nr_class1) or line[0] == str(nr_class2):
				fout.write(line)


if __name__ == "__main__":

	for i in range(10):
		for j in range(i):
			construct_class_file(i, j, "pendigits")
			subprocess.run(["./svm-train", "-b", "1", "-t", "1",  "-d", "7", "-c", "100", "-g", "1", "pendigits" + str(i) + str(j), "pendigits.model" + str(i) + str(j)])
			subprocess.run(["./svm-predict", "-b", "1", "pendigits.t", "pendigits.model" + str(i) + str(j), 'out' + str(i) + str(j)])
	predicts = {k: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for k in range(1, 3499)}

	for i in range(10):
		for j in range(i):
			jj = -1
			el_pos = None
			el_neg = None
			with open("out" + str(i) + str(j)) as fin:
				for line in fin:
					jj += 1
					if jj == 0:
						L = line.split(' ')
						L[1] = L[1].strip('\n')
						L[2] = L[2].strip('\n')
						el_pos = int(L[1])
						el_neg = int(L[2])
						continue

					L = line.split(' ')
					L[0] = L[0].strip('\n')
					L[1] = L[1].strip('\n')
					L[2] = L[2].strip('\n')

					neg = float(L[2])
					pos = float(L[1])
					p = int(L[0])
					if pos > 0.5:
						predicts[jj][el_pos] += 1
					else:
						predicts[jj][el_neg] += 1

	
	correct = 0
	total = 0

	with open("pendigits.t") as fin:
		jj = 0
		for line in fin:
			
			jj += 1
			pred_jj = 0
			for x in range(len(predicts[jj])):
				if predicts[jj][x] > predicts[jj][pred_jj]:
					pred_jj = x

			p = int(line[0])
			if p == pred_jj:
				total += 1
				correct += 1
			else:
				total += 1

	print(correct / total)




