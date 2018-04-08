from array import array as pyarray
from numpy import append, array, int8, uint8, zeros, dtype
import os, struct
import sys
import timeit
import pickle

import numpy as np
from operator import countOf

class AlphabetRecog(object):

	def __init__(self, inputeUnits, hiddenUnits, outputUnits, pathToDatasetFolder, enableWeightDecay=False):
		self.nh = hiddenUnits
		self.no = outputUnits
		self.ni = inputeUnits
		self.enableWeightDecay = enableWeightDecay
		self.gama = 0.01
		self.etha = 0.01
		
		print("Loading the images...")
		self.images, self.labels = self.load_mnist(path=pathToDatasetFolder)
		self.images = 1.0 * self.images / np.amax(self.images.ravel())
#         self.images = self.images[:1000]
#         self.labels = self.labels[:1000]
		print("done")
				
		self.reset_weights()
	
	def reset_weights(self):
		self.w1 = np.random.randn(self.nh, self.ni + 1)
		self.w1 = self.w1 / np.max(self.w1);
		self.w2 = np.random.randn(self.no, self.nh + 1)
		self.w2 = self.w2 / np.max(self.w2);


	def load_mnist(self, dataset="training", digits=np.arange(10), path="."):
		'''
		Load the training data-set and the labels
		'''
		if dataset == "training":
			fname_img = os.path.join(path, 'train-images.idx3-ubyte')
			fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
		elif dataset == "testing":
			fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
			fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
		else:
			raise ValueError("dataset must be 'testing' or 'training'")
	
		flbl = open(fname_lbl, 'rb')
		magic_nr, size = struct.unpack(">II", flbl.read(8))
		lbl = pyarray("b", flbl.read())
		flbl.close()
	
		fimg = open(fname_img, 'rb')
		magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
		img = pyarray("B", fimg.read())
		fimg.close()
	
		ind = [ k for k in range(size) if lbl[k] in digits ]
		N = len(ind)
	
		images = zeros((N, rows, cols), dtype=uint8)
		labels = zeros((N, 1), dtype=int8)
		for i in range(len(ind)):
			images[i] = array(img[ ind[i] * rows * cols : (ind[i] + 1) * rows * cols ]).reshape((rows, cols))
			labels[i] = lbl[ind[i]]
	
		return images, labels      

	def signum(self, x):
		res = 1 / (1 + np.exp(-x))
		res[ np.isnan(res)] = 0;
		return res;
	
	def signum_prime(self, x):
		return self.signum(x) * (1 - self.signum(x));
			
	def get_update_rule_for_out_gradient(self, etha, s_out, y):
		return etha * np.dot(s_out , y.T)
	
	def get_update_rule_for_hidd_gradient(self, etha, s_hidd, x):
		return etha * np.dot(s_hidd, x.T)
	
	def update_weights_bias(self, grad_in_hidd, grad_hidd_out):
		self.w1 = self.w1 + grad_in_hidd + (self.etha * self.gama * self.w1 if self.enableWeightDecay else 0)
		self.w2 = self.w2 + grad_hidd_out + (self.etha * self.gama * self.w2 if self.enableWeightDecay else 0)
		
	def get_prediction(self, x):
		'''
		Predict the digital number that is represented by x
		:param x:  pixel values of 28x28 image, 
		'''
		x = x.ravel()  
		x = np.array([ 0 if val == 0 else 1 for val in x ])
		x_a = np.insert(x, 0, values=1, axis=0) 
		net_hidd = np.dot(self.w1, x_a)
		y = self.signum(net_hidd)
		y_a = np.insert(y, 0, values=1, axis=0)
		net_out = np.dot(self.w2, y_a)
		z = self.signum(net_out)
		return np.argmax(z, axis=0)
		
	def loadDatasetWithNFold(self, startIndexForTestData, endIndexForTestData):
		imagesTrainSet1 = self.images[:startIndexForTestData]
		imagesTrainSet2 = self.images[endIndexForTestData:]
		imagesTrainSet = np.vstack((imagesTrainSet1, imagesTrainSet2))
		
		labelsTrainSet1 = self.labels[:startIndexForTestData]
		labelsTrainSet2 = self.labels[endIndexForTestData:]
		labelsTrainSet = np.vstack((labelsTrainSet1, labelsTrainSet2))
		
		imagesTestSet = self.images[startIndexForTestData:endIndexForTestData]
		labelsTestSet = self.labels[startIndexForTestData:endIndexForTestData]
		
		print("Trainset size: " + str(imagesTrainSet.shape))
		print("Test size: " + str(imagesTestSet.shape))
		
		return (imagesTrainSet, labelsTrainSet, imagesTestSet, labelsTestSet)


	def trainNN(self, imagesTrainSet, labelsTrainSet, etha):
		self.reset_weights()
		trainingSetSize = labelsTrainSet.shape[0];
		j = 0
		while j < 30:
			i = 0
			# print("Round: " + str(j + 1))
			while i < trainingSetSize :
				x = imagesTrainSet[i].ravel()  # Convert 28x28 pixel image into a (784,) vector
				x = np.array([ 0 if val == 0 else 1 for val in x ])
				x_a = np.insert(x, 0, values=1, axis=0)  # Augmented Feature vector
				net_hidd = np.dot(self.w1, x_a)
				y = self.signum(net_hidd)
				y_a = np.insert(y, 0, values=1, axis=0)  # Augmented Feature vector
				
				net_out = np.dot(self.w2, y_a)
				z = self.signum(net_out)
				lab = np.array([ 1 if k == self.labels[i] else 0 for k in range(10) ])
				
				J = z - lab;
				J = np.sum(0.5 * J * J);
				if J < 1 and self.enableWeightDecay:
					break;
				out_sensitivity = (lab - z) * self.signum_prime(net_out)
				net_hidd_prime = self.signum_prime(net_hidd) 
				hid_sensitivity = np.dot(self.w2.T, out_sensitivity) * np.insert(net_hidd_prime, 0, 1)
				
				grad_hidd_out = etha * np.outer(out_sensitivity, y_a.T)
				grad_in_hidd = etha * np.outer(hid_sensitivity[1:] , x_a.T) 
				
				self.update_weights_bias(grad_in_hidd, grad_hidd_out)
				i += 1
			j += 1
			
		return self.w1, self.w2
	
	def getAccuracy(self, imagesTrainSet, labelsTrainSet, imagesTestSet, labelsTestSet):
				
		totalCount = imagesTestSet.shape[0]
		confusionMatrix = np.zeros(shape=(10, 10))
		
		for i in range(totalCount):
			prediction = self.get_prediction(imagesTestSet[i])
			confusionMatrix[ labelsTestSet[i], prediction ] += 1
	
		return confusionMatrix;

	
	def kFoldCrossValidationAccuracy(self, k):
		countOfImages = 1000  # self.images.shape[0];
		foldSize = countOfImages / k
		currentFoldNum = 0;
		squared_sum = 0;
		overall_sum = 0
		overall_mean = 0
		
		for i in range(0, countOfImages, foldSize):
			print("++ " + str(i), str(i + foldSize - 1))
			(imagesTrainSet, labelsTrainSet, imagesTestSet, labelsTestSet) = self.loadDatasetWithNFold(int(i), int(i + foldSize - 1));
			w1, w2 = self.trainNN(imagesTrainSet, labelsTrainSet, etha=self.etha)
			confusionMatrix = self.getAccuracy(imagesTrainSet, labelsTestSet, imagesTestSet, labelsTestSet)
			sum = np.sum([ confusionMatrix[i][i] for i in range(confusionMatrix.shape[0]) ])
			squared_sum += (sum * sum)
			mean = sum / np.sum(confusionMatrix.ravel())
			overall_mean += mean
			overall_sum += sum
			print("For Fold-" + str(currentFoldNum))
			print("Confusion Matrix:")
			print(confusionMatrix)
			print ("+ Mean accuracy-" + str(mean))
			precesion = {}
			specifity = {}
			sensitivity = {}
			for digit in range(10):
				tp = np.sum(confusionMatrix[digit, digit])
				fn = np.sum(confusionMatrix[digit, :]) - confusionMatrix[digit, digit]
				fp = np.sum(confusionMatrix[:, digit]) - confusionMatrix[digit, digit]
				tn = np.sum(confusionMatrix)
				sensitivity[digit] = tp / (tp + fn)
				precesion[digit] = tp / (tp + fp)
				specifity[digit] = tn / (tn + fp)
			print ("+ sensitivity-" + str(sensitivity[digit]))
			print ("+ precesion-" + str(precesion[digit]))
			print ("+ specifity-" + str(specifity[digit]))
			print("\n\n\n")
			currentFoldNum += 1
		overall_mean = overall_mean * 1.0 / k;
		import math as m
		return (overall_mean, m.sqrt(1.0 * squared_sum / k - overall_mean))


def main(path):
	start = timeit.default_timer()
	
	ar = AlphabetRecog(inputeUnits=28 * 28, hiddenUnits=100, outputUnits=10,
					   pathToDatasetFolder=path);
	etha = 0.01
	i = 0;
	trainingSetSize = ar.labels.shape[0];
	
	
	# 5-Fold Cross validation
	print("5-Fold Cross validation Accuracy: ")
	mean, variance = ar.kFoldCrossValidationAccuracy(k=5)
	print("Average Mean-Accuracy:" + str(mean))
	print("Variance in accuracy:" + str(variance))
	
	f = open('w1_w2_kfold', 'wb+')
	print("Saving the Weights to file")
	pickle.dump((ar.w1, ar.w2), f)
	f.close()
	print("Done.")
	
	stop = timeit.default_timer()
	print(str(stop - start) + " secs")
	
		
if __name__ == "__main__":
	
	path = sys.argv[1];
	main(path)