import csv
import random
import math

d1={} #yes
d2={} #no
counter_yes=0
counter_no=0
trainingSet_length=0
pridiction_list=[]
output_count=0
accuracy_list=[]

def loadCsv(filename):
	skiplist=[1,2,3,4,5,6,7,8,9,14,20]
	newdataset=[]
	with open(filename) as csvfile:
		lines = csv.reader(csvfile, delimiter=';', quotechar='"')
		dataset = list(lines)
		for i in range(1,len(dataset)):
			templist=[]
			if (dataset[i][1]!='unknown' and dataset[i][2]!='unknown' and dataset[i][3]!='unknown' and dataset[i][4]!='unknown' and dataset[i][5]!='unknown' and dataset[i][6]!='unknown' and dataset[i][7]!='unknown' and dataset[i][8]!='unknown' and dataset[i][9]!='unknown' and dataset[i][14]!='unknown'):
				for x in skiplist:
					templist.append(dataset[i][x])
				newdataset.append(templist)
		return newdataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def trainig(trainingSet):
	global counter_yes
	global counter_no
	for i in range (0,10):
		d1[i]={}
		d2[i]={}
	for i in range (len(trainingSet)):
		for j in range(0,10):
			if(trainingSet[i][-1]=='yes'):
				counter_yes=counter_yes+1
				if(d1[j].has_key(trainingSet[i][j])== True ):
					temp=d1[j].get(trainingSet[i][j])
					d1[j][trainingSet[i][j]]=temp+1
				else:
					d1[j][trainingSet[i][j]]=1
			else:
				counter_no=counter_no+1
				if(d2[j].has_key(trainingSet[i][j])== True):
					temp=d2[j].get(trainingSet[i][j])
					d2[j][trainingSet[i][j]]=temp+1
				else:
					d2[j][trainingSet[i][j]]=1
	#print d1
	#print d2

def calpro(testSet):
	cal_proyes=0
	cal_prono=0
	global counter_yes
	global counter_no
	global d1
	global d2
	proyes=counter_yes/(1.0*trainingSet_length)
	prono=counter_no/(1.0*trainingSet_length)
	global output_count
	for i in range(len(testSet)):
		cal_prono=0
		cal_proyes=0
		for j in range(len(testSet[i])-1):
			if testSet[i][j] in d1[j]:
				cal_proyes=float(cal_proyes+math.log(d1[j][testSet[i][j]]/proyes))
			if testSet[i][j] in d2[j]:
				cal_prono=float(cal_prono+math.log(d2[j][testSet[i][j]]/prono))
		cal_proyes=cal_proyes*proyes
		cal_prono=cal_prono*prono
		if cal_proyes >= cal_prono:
			output='yes'
		else:
				output='no'
		pridiction_list.append(output)
		if testSet[i][-1]==output:
			output_count=output_count+1
	#print "output_count=",output_count
	accuracy=output_count/(1.0*len(testSet))*100
	accuracy_list.append(accuracy)
	print "accuracy=",accuracy
	print "confusion matrix"
	x=0
	y=0
	twodl=[[0,0],[0,0]]
	for i in range (len(testSet)):
		if testSet[i][-1]=='yes':
			x=0
		if testSet[i][-1]=='no':
			x=1
		if pridiction_list[i]=='yes':
			y=0
		if pridiction_list[i]=='no':
			y=1
		twodl[x][y]=twodl[x][y]+1
	for i in range(2):
		for j in range(2):
			print twodl[i][j], '\t',
		print " "

def main():
	filename = 'bank-additional-full.csv'
	dataset = loadCsv(filename)
	splitRatio = 0.5
	global counter_yes
	global counter_no
	global trainingSet_length
	#global pridiction_list=[]
	global output_count
	for i in range(0,10):
		d1.clear()
		d2.clear()
		counter_yes=0
		counter_no=0
		trainingSet_length=0
		del pridiction_list[:]
		output_count=0
		print "iteration", i+1
		trainingSet, testSet = splitDataset(dataset, splitRatio)
		global trainingSet_length
		trainingSet_length=len(trainingSet)
		#print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
		trainig(trainingSet)
		calpro(testSet)
	meanacuuracy=sum(accuracy_list)/float(len(accuracy_list))
	print "average accuracy=",meanacuuracy
	temp=0
 	for i in range (0,10):
 		temp=temp+math.pow(accuracy_list[i]-meanacuuracy,2)
 	temp=math.pow(temp/float(10),.5)
 	print "standard deviation=",temp
main()