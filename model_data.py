import quandl
import math
import numpy as np
import pandas as pd
import random

class preprocessData(object):


	def __init__(self,pandas_frame=None,single_column=False,window_size=1,num_steps=5,normalized=True):
		self.pandas_frame = pandas_frame
		self.single_column = single_column
		self.window_size=window_size
		self.num_steps = num_steps
		self.normalized = normalized


	def generate_frame(self):
		quandl.ApiConfig.api_key = 'upvv8dx3pwLpm_Rxi8iP'
		stock_exchange = quandl.get('XBOM/500010', start_date='200-01-20', end_date='2018-07-02')
		vectorized_input = stock_exchange.iloc[:,3]

		return vectorized_input


	def generate_full_frame(self):
		quandl.ApiConfig.api_key = 'upvv8dx3pwLpm_Rxi8iP'
		stock_exchange = quandl.get('XBOM/500010', start_date='2000-01-20', end_date='2018-07-02')
		vectorized_input = stock_exchange.iloc[:,0:4]

		return vectorized_input


	def to_list(self):
		if self.single_column:
			pandas_frame = self.generate_frame()
		else:
			pandas_frame = self.generate_full_frame()
		pandas_list = pandas_frame.values.tolist()

		return pandas_list


	def prepare_data(self,seq):

		seq = [np.array(seq[i * self.window_size: (i + 1) * self.window_size])
               for i in range(len(seq) // self.window_size)]

		if self.normalized:
			divisor = [seq[i][0][-1] for i in range(len(seq))]
			seq[0][0] = np.log(seq[0][0]/seq[0][0][0])
			for i in range(1,len(seq)):

				seq[i][0] = np.log(seq[i]/divisor[i-1])

		X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
		if not self.single_column:
			X = X.reshape((X.shape[0],X.shape[1],X.shape[3])) 
			#print(X)

			list_arr_x = []
			for i in range(self.num_steps):
				a = X[:,i,:]
				list_arr_x.append(a)

			X = np.stack(list_arr_x,axis=0)


		#X = np.transpose(X,(0,2,1))
		if not self.single_column:
			Y = np.array([seq[i+self.num_steps][0] for i in range(len(seq) - self.num_steps)])
			Y = Y.reshape((Y.shape[0],4))
		else:
			Y = np.array([seq[i+self.num_steps] for i in range(len(seq) - self.num_steps)])


		return seq,X,Y


	@staticmethod
	def minibatches(X, Y, mini_batch_size):
	    m = X.shape[1]
	    mini_batches = []
	    permutation = list(np.random.permutation(m))
	    shuffled_X = X[:, permutation, :]
	    shuffled_Y = Y[permutation, :]
	    num_complete_minibatches = math.floor(m / mini_batch_size)
	    num_minibatches = num_complete_minibatches

	    for k in range(0, num_complete_minibatches):
	        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size, :]
	        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]

	        mini_batch = (mini_batch_X, mini_batch_Y)
	        mini_batches.append(mini_batch)

	    # handling last minibatch
	    if m % mini_batch_size != 0:
	        mini_batch_X = shuffled_X[:, 0:(m - mini_batch_size * num_complete_minibatches), :]
	        mini_batch_Y = shuffled_Y[0:(m - mini_batch_size * num_complete_minibatches), :]
	        mini_batch = (mini_batch_X, mini_batch_Y)
	        mini_batches.append(mini_batch)
	        num_minibatches += 1

	    return mini_batches, num_minibatches
	@staticmethod
	def print_mini_batch_sizes(mini_batches, num_minibatches):
	    for i in range(num_minibatches):
	        print ("shape of mini_batch" + str(i) + ": " + str(mini_batches[i][0].shape))

data = preprocessData()

#print(data.generate_full_frame())
seq = data.to_list()
seq,X,Y = data.prepare_data(seq)
#print(seq)
#print(seq[0][0][-1])
mini_batches,num_minibatches = data.minibatches(X,Y,10)
#print(X.shape)






