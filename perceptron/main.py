#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

class1 = [[1, 6],[7, 2],[8, 9],[9, 9],[4, 8],[8, 5]]
xcord1=[1,7,8,9,4,8]
ycord1=[6,2,9,9,8,5]

class2 = [[2, 1],[3, 3],[2, 4],[7, 1],[1, 3],[5, 2]]
xcord2=[2,3,2,7,1,5]
ycord2=[1,3,4,1,3,2]

class1nm = [[1, 6, 1],[7, 2, 1],[8, 9, 1],[9, 9, 1],[4, 8, 1],[8, 5, 1]]

class2nm = [[-2, -1, -1],[-3, -3, -1],[-2, -4, -1],[-7, -1, -1],[-1, -3, -1],[-5, -2, -1]]

dataset=np.array(class1nm+class2nm)

def plotfunc(listvector,id):
	linex=[]
	liney=[]
	for x in range (0,10,1):
		y=(-listvector[0]*x - listvector[2])/listvector[1]
		liney.append(y)
		linex.append(x)
	if(id==1):
		plt.plot(linex,liney,'-',color='black',label="ssp")

	elif(id==2):
		plt.plot(linex,liney,'-',color='red',label='sspwm')
	elif(id==3):
		plt.plot(linex,liney,'-',color='blue',label='rel')
	else:
		plt.plot(linex,liney,'-',color='green',label='woh')
	#plt.show()


def single_sample_perceptron():
	alpha = np.array([1 , 1 , 1])
	eta=.5
	b=2
	size_data = len(dataset)
	counter=0
	while(1):
		flag=0
		counter=0
		for i in range(size_data):
			dist=np.dot(alpha,dataset[i])
			#print dist
			if(dist<=0):
				flag=1
				counter=counter+1
				alpha=alpha+eta*dataset[i]
		#print alpha		
		if counter==0 :
			break
	
	plotfunc(alpha,1)

def Single_sample_perceptron_with_margin():
	alpha = np.array([1 , 1 , 1])
	eta=.5
	b=2
	size_data = len(dataset)
	counter=0
	while(1):
		counter=0
		for i in range(size_data):
			dist=np.dot(alpha,dataset[i])
			#print dist
			if(dist<=b):
				flag=1
				counter=counter+1
				alpha=alpha+eta*dataset[i]
		#print alpha		
		if counter==0 :
			break
	plotfunc(alpha,2)

def Relaxation_algorithm_with_margin():
	alpha=np.random.rand(3)
	eta=2
	b=10
	size_data = len(dataset)
	counter=0
	while(1):
		counter=0
		for i in range(size_data):
			dist=np.dot(alpha,dataset[i])
			value=np.dot(dataset[i],dataset[i])
			#print dist
			if dist <=b :
				#print dist
				counter=counter+1
				alpha=alpha+eta*dataset[i]*((b - dist)/(value))
		
		if counter==0 :
			break
	plotfunc(alpha,3)


def Widrow_Hoff_or_Least_Mean_Squared_Rule():
	alpha=np.random.rand(3)
	eta=.05
	b=2
	size_data = len(dataset)
	counter=0
	#counter1=0
	while(1):
		counter=0
		for i in range(size_data):
			dist=np.dot(alpha,dataset[i])
			value=np.dot(dataset[i],dataset[i])
			#print dist
			if dist <=b :
				#print dist
				counter=counter+1
				alpha=alpha+eta*dataset[i]*(b - dist)
		#print alpha	
		#counter1=counter1+1
		if counter==0 :
			break
	
	plotfunc(alpha,4)

def main():
	
	axes = plt.gca()
	axes.set_xlim([0,12])
	axes.set_ylim([0,12])
	plt.plot(xcord1,ycord1,".r",label='class1',marker="o")
	plt.plot(xcord2,ycord2,".b",label='class2',marker="x")
	#plt.legend()
	single_sample_perceptron()
	Single_sample_perceptron_with_margin()
	Relaxation_algorithm_with_margin()
	Widrow_Hoff_or_Least_Mean_Squared_Rule()
	plt.legend()
	plt.show()
	

main()