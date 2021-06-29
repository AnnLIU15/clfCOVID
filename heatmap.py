import numpy as np 
import re
if __name__=='__main__':
	text_path='./output/clfResult/vgg19/confusion_matrix.txt'

	with open(text_path,'r') as f:
		text_data=f.read()
	arr=np.array([float(s) for s in re.findall(r'\b\d+\b', text_data)] ).reshape(3,3)
	a=arr.sum(axis=0)
	print(arr.astype(np.int))
	for i in range(3):
		arr[:,i]/=a[i]
	print(arr)

