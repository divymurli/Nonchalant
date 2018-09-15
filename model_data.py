import quandl
import math
import numpy as np
import pandas as pd
import random

class preprocessData(object):


	def __init__(self,test_ratio=0.05,seed=42,pandas_frame=None,single_column=False,window_size=1,num_steps=4,normalized=True):
		self.pandas_frame = pandas_frame
		self.single_column = single_column
		self.window_size=window_size
		self.num_steps = num_steps
		self.normalized = normalized
		self.test_ratio = test_ratio
		self.seed = seed


	def generate_frame(self):
		quandl.ApiConfig.api_key = 'upvv8dx3pwLpm_Rxi8iP'
		stock_exchange = quandl.get('XBOM/500010', start_date='2000-01-20', end_date='2018-07-02')
		vectorized_input = stock_exchange.iloc[:,3]

		return vectorized_input


	#one stock
	def generate_full_frame(self):
		quandl.ApiConfig.api_key = 'upvv8dx3pwLpm_Rxi8iP'
		stock_exchange = quandl.get('XBOM/500010', start_date='2000-01-20', end_date='2018-09-12')
		vectorized_input = stock_exchange.iloc[:,0:4]

		return vectorized_input

	#kwargs dict of data frames to plug in
	def generate_concatenate_multiple_frames(self,stock_dict, start_date, end_date):
		quandl.ApiConfig.api_key = 'upvv8dx3pwLpm_Rxi8iP'

		vectorized_input_dict = {}

		for key in stock_dict:
			vectorized_input_dict[key] = quandl.get(stock_dict[key], start_date=start_date, end_date=end_date).iloc[:,0:4]
			vectorized_input_dict[key] = vectorized_input_dict[key].dropna(axis=1)

		output_frame = pd.concat(vectorized_input_dict,axis=1,ignore_index=False)
	

		return output_frame, vectorized_input_dict




	def to_list(self):
		if self.single_column:
			pandas_frame = self.generate_frame()
		else:
			pandas_frame = self.generate_full_frame()
		pandas_list = pandas_frame.values.tolist()

		return pandas_list


	def stocks_to_list(self,vectorized_input_dict):

		stocks_to_list = {}
		seq = {}

		for key in vectorized_input_dict:
			stocks_to_list[key] = vectorized_input_dict[key].values.tolist()

			seq[key] = [np.array(stocks_to_list[key][i * self.window_size: (i + 1) * self.window_size])
			   for i in range(len(stocks_to_list[key]) // self.window_size)]



		return stocks_to_list, seq



	def prepare_data(self,seq):

		seq = [np.array(seq[i * self.window_size: (i + 1) * self.window_size])
			   for i in range(len(seq) // self.window_size)]

		#print("divisors: " + str([seq[i][0][-1] for i in range(len(seq))]))

		seq_std = 1

		#change it so first day is normalized by price of previous day
		if self.normalized:
			divisor = [seq[i][0][-1] for i in range(len(seq))]
			seq[0][0] = np.log(seq[0][0]/seq[0][0][0])
			for i in range(1,len(seq)):

				seq[i][0] = np.log(seq[i]/divisor[i-1])
				
				#seq[i][0] = seq[i][0] 

			seq_std = np.std(seq)
			
			seq = seq / seq_std

			#return seq_std

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



		return seq,X,Y, seq_std

	def prepare_baseline_data(self, seq):
		seq = [np.array(seq[i * self.window_size: (i + 1) * self.window_size])
			   for i in range(len(seq) // self.window_size)]

		#normalization part
		seq_std = 1

		if self.normalized:
			divisor = [seq[i][0][-1] for i in range(len(seq))]
			seq[0][0] = np.log(seq[0][0]/seq[0][0][0])
			for i in range(1,len(seq)):

				seq[i][0] = np.log(seq[i]/divisor[i-1])
				
				#seq[i][0] = seq[i][0] 

			seq_std = np.std(seq)
			
			seq = seq / seq_std

		X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
		X = X.reshape((X.shape[0],X.shape[1],X.shape[3]))

		list_arr_x = []
		for i in range(self.num_steps):
			a = X[:,i,:]
			list_arr_x.append(a)

		X = np.stack(list_arr_x,axis=0)

		Y_row = []
		for i in range(X.shape[1]):
			Y_row.append(np.repeat(X[-1][i][-1],4))
			#Y_row.append(X[-1][i])
			#Y_row[i][1] = X[-1][i][1]
			#Y_row[i][2] = X[-1][i][2]
		Y = np.stack(Y_row,axis=0)

		return seq, X, Y  

	def compute_loss_baseline_data(self,Y_pred,Y):
		return np.mean( np.square( np.subtract(Y_pred, Y)  ) )

	def compute_loss_open_baseline_data(self,Y_pred,Y):
		return np.mean( np.square( np.subtract(Y_pred[:,0], Y[:,0])  ) )


	def split_data(self,X,Y):
		
		train_size = int(X.shape[1] * (1.0 - self.test_ratio))

		m = X.shape[1]
		np.random.seed(self.seed)
		permutation_array = np.random.permutation(m)
		permutation = list(permutation_array)
		#print("permutation_used: " + str(permutation))
		shuffled_X = X[:, permutation,:]
		shuffled_Y = Y[permutation, :]

		#print("Shuffled_X:")
		#print(shuffled_X)

		train_X, test_X = shuffled_X[:,:train_size,:], shuffled_X[:,train_size:,:]
		train_Y, test_Y = shuffled_Y[:train_size,:], shuffled_Y[train_size:,:]

		return train_X, train_Y, test_X, test_Y,permutation_array

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




data = preprocessData(test_ratio = 0.2,normalized=True)
"""
stock_dict = {"stock_1": "XBOM/500010", "stock_2": "FSE/ZO1_X", "stock_3": "FSE/EON_X"}

output_frame, vectorized_input_dict = data.generate_concatenate_multiple_frames(stock_dict,'2018-08-20','2018-09-01')

stocks_to_list,seq = data.stocks_to_list(vectorized_input_dict)

#print(stocks_to_list)
print(stocks_to_list['stock_3'])
print(seq['stock_2'])

print(output_frame)
"""


#print(data.generate_full_frame())
seq_1 = data.to_list()
#print(seq_1)
seq,X,Y,seq_std = data.prepare_data(seq_1)
#print(X)

#print(seq)


_,X_base, Y_base = data.prepare_baseline_data(seq_1)
#print(X_base)
#print(Y)
#print(Y_base)
print("Baseline cost: " + str(data.compute_loss_baseline_data(Y,Y_base)))
print("Baseline cost open: " + str(data.compute_loss_open_baseline_data(Y,Y_base)))



"""
print(seq_std)
train_X,train_Y,test_X,test_Y = data.split_data(X,Y)
print(test_X)
print(test_Y)
print(test_X[3][-1][-1])
print(test_X.shape)
"""

#print(seq)
#print(X)
#print(Y)
#print(seq)
#print(np.std(seq))
#print(seq[0][0][-1])
#mini_batches,num_minibatches = data.minibatches(X,Y,10)
#print(X.shape)

#print(Y.shape)
#print(train_X.shape)
#print(train_Y.shape)
#print(test_X.shape)
#print(test_Y.shape)







