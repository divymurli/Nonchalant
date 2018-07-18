import quandl
import math
import numpy as np
import pandas as pd
import random

class preprocessData(object):

	def __init__(self,pandas_frame=None,single_column=False,window_size=1,num_steps=5,normalized=False):
		self.pandas_frame = pandas_frame
		self.single_column = single_column
		self.window_size=window_size
		self.num_steps = num_steps
		self.normalized = normalized


	def generate_frame(self):
		quandl.ApiConfig.api_key = 'upvv8dx3pwLpm_Rxi8iP'
		stock_exchange = quandl.get('XBOM/500010', start_date='2018-04-20', end_date='2018-07-02')
		vectorized_input = stock_exchange.iloc[:,3]

		return vectorized_input

	def generate_full_frame(self):
		quandl.ApiConfig.api_key = 'upvv8dx3pwLpm_Rxi8iP'
		stock_exchange = quandl.get('XBOM/500010', start_date='2018-05-20', end_date='2018-07-02')
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
			divisor = [seq[i][-1] for i in range(len(seq))]
			seq[0] = seq[0]/seq[0][0] - 1.0
			for i in range(1,len(seq)):

				seq[i] = seq[i]/divisor[i-1] - 1.0

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
			Y = np.array([seq[i+self.num_steps][0][0] for i in range(len(seq) - self.num_steps)])
			Y = Y.reshape((Y.shape[0],1))
		else:
			Y = np.array([seq[i+self.num_steps] for i in range(len(seq) - self.num_steps)])


		return seq,X,Y



data = preprocessData()

dataframe = data.generate_full_frame()

seq = data.to_list()

seq,X,Y = data.prepare_data(seq)
print(seq)
print(X)
#print(X[:,0,:])
print(X.shape)


"""
divisor = [seq[i][-1] for i in range(len(seq))]
print(divisor)
for i in range(1,len(seq)):

	seq[i] = seq[i]/divisor[i-1]

print(seq)
"""




