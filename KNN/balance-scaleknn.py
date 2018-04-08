#!/usr/bin/python

import sys
import random
import csv
import operator
import math

trainigset=[]
testset=[]
testsetlength=0
trainigsetlength=0

def nearestneighbour(x,k):
	distancelist=[]
	length=len(testset[x])
	neighbour=[]
	for i in range(trainigsetlength):
		distance=0
		for j in range(1,length):
			distance=distance+pow((testset[x][j]-trainigset[i][j]),2)
		distance=math.sqrt(distance)
		distancelist.append((trainigset[i],distance))
	distancelist.sort(key=operator.itemgetter(1))
	for i in range(k):
		neighbour.append(distancelist[i][0])
	classvalue={}
	for i in range(len(neighbour)):
		value = neighbour[i][0]
		if value in classvalue:
			classvalue[value]+=1
		else:
			classvalue[value]=1
	finalclassvalue=sorted(classvalue.iteritems(),key=operator.itemgetter(1),reverse=True)
	return finalclassvalue[0][0]

f= open('balance-scale.data','rb')
#xcord=[]
#ycord=[]
lines=csv.reader(f)
data=list(lines)
for i in range(len(data)):
	for j in range(1,5):
		data[i][j]=float(data[i][j])
	if random.random() < .5:
		trainigset.append(data[i])
	else:
		testset.append(data[i])
#for loop in range(0,10):
#	random.shuffle(data)
#	accuracy1[:]=[]
#	accuracy2[:]=[]
#	for i in range(0,5):
#		trainigset[:]=[]
#		testset[:]=[]
#		for j in range(i*len(data)/5,(i+1)*len(data)/5):
#			trainigset.append(data[j])
#		for k in range(0,i*len(data)/5):
#			testset.append(data[k])
#		for k in range((i+1)*len(data)/5,len(data)):
#			testset.append(data[k])
#

trainigsetlength= len(trainigset)
testsetlength=len(testset)
inputcase=[1,3]
for k in inputcase: 
	prediction=[]
	for i in range(testsetlength):
		result= nearestneighbour(i,k)
		prediction.append(result)
	correct=0
	confusionmatrix=[[0,0,0],[0,0,0],[0,0,0]]
	for i in range(testsetlength):
		if(testset[i][0]=='L'):
			x=0
		elif(testset[i][0]=='B'):
			x=1
		else:
			x=2
		if(prediction[i]=='L'):
			y=0
		elif(prediction[i]=='B'):
			y=1
		else:
			y=2
		confusionmatrix[x][y]+=1
		if testset[i][0]==prediction[i]:
			correct+=1
	print "confusionmatrix for k=%d" % k
	for n in range(3):
		for m in range(3):
			print confusionmatrix[n][m],'\t',
		print " "		
	output=(correct/float(testsetlength))*100
	print "Accuracy for k=%d: %f" % (k,output)