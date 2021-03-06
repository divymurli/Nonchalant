import quandl
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import random
from model_data import preprocessData
# get data
quandl.ApiConfig.api_key = 'upvv8dx3pwLpm_Rxi8iP'
frankfurt_data = quandl.get('FSE/EON_X', start_date='2012-05-09', end_date='2018-06-22')
scalar_input = quandl.get('FSE/EON_X', start_date='2012-05-09', end_date='2018-06-22', column_index='3',
						  returns='numpy')
bombay_stock_exchange = quandl.get('XBOM/500010', start_date='2000-05-20', end_date='2018-07-02')
hongkong_stock_exchange = quandl.get('XHKG/00005', start_date = '2000-05-20',end_date = '2018-07-02')
vectorized_input = bombay_stock_exchange.iloc[:,0:4]
#print(vectorized_input)

def naive_normalization(pandas_frame):

	mean = pandas_frame.mean().mean()
	std = pandas_frame.stack().std()

	normalized_frame = (pandas_frame - mean)/std
	#normalized_frame = (pandas_frame)/std
	#normalized_frame = pandas_frame - mean

	return normalized_frame

def convert_to_numpy(pandas_frame, begin_column, end_column):
	numpy_frame = np.asarray(pandas_frame.iloc[:, begin_column:end_column], dtype=np.float32)

	return numpy_frame


def generateData(N):
	a1 = [a[i][2] for i in range(a.shape[0] - a.shape[0] % N)]
	x = np.array([a1[i:i + N - 1] for i in range(0, len(a1), N)])
	x = x.T.reshape((N - 1, x.shape[0], 1))
	y = np.array([[a1[i] for i in range(len(a1)) if (i + 1) % N == 0]]).T

	return x, y


def generateData2(N, input_frame):
	M = input_frame.shape[0] - input_frame.shape[0] % N
	a1 = input_frame[0:M, :]
	mask1 = np.remainder(np.arange(M), N)
	a_x = a1[mask1 != N - 1, :]
	a_y = a1[mask1 == N - 1, 0]
	a_y = a_y.reshape(a_y.shape[0], 1)
	# print(a_x)
	mask = np.remainder(np.arange(int(a_x.shape[0])), N - 1)
	list_arr_x = []
	for i in range(int(a_x.shape[0] / a_y.shape[0])):
		a = a_x[mask == i, :]
		list_arr_x.append(a)

	array_x = np.stack(list_arr_x, axis=0)

	return array_x, a_y


def initialize_parameters(n_a, n_x, n_y):
	parameters = {}
	parameters["W"] = tf.get_variable("W", shape=[n_a + n_x, n_a], initializer=tf.contrib.layers.xavier_initializer())
	parameters["b"] = tf.get_variable("b", shape=[1, n_a], initializer=tf.zeros_initializer())
	parameters["Wy"] = tf.get_variable("Wy", shape=[n_a, n_y], initializer=tf.contrib.layers.xavier_initializer())
	parameters["by"] = tf.get_variable("by", shape=[1, n_y], initializer=tf.zeros_initializer())

	return parameters


# All encoder cells
def rnn_cell_forward(xt, a_prev, parameters):
	W = parameters['W']
	b = parameters['b']

	concatenated = tf.concat([a_prev, xt], 1)

	a_next = tf.nn.relu(tf.matmul(concatenated, W) + b)

	return a_next


def rnn_forward(inputs_series, init_state, parameters):
	# x: shape (m,n_x)
	# a0: shape (m,n_a)

	current_state = init_state
	states_series = []

	for current_input in inputs_series:  # loop over time steps
		# current_input = tf.reshape(current_input,[batch_size,1])
		next_state = rnn_cell_forward(current_input, current_state, parameters)
		states_series.append(next_state)

		# states_series.append(current_state)
		current_state = next_state

	prediction = tf.matmul(current_state, parameters['Wy']) + parameters['by']
	

	return prediction, states_series, current_state


# Cost function
def compute_cost(prediction, label):
	loss = tf.reduce_mean(tf.square(tf.subtract(prediction, label)))
	#loss =  tf.square(tf.subtract(prediction, label))
	return loss

def compute_cost_column(prediction,label):
	prediction = tf.transpose(prediction)
	label = tf.transpose(label)
	#print(label.shape)
	prediction_open = tf.gather_nd(prediction,[0])
	label_open = tf.gather_nd(label,[0])
	#print(label_open.shape)
	loss = tf.reduce_mean(tf.square(tf.subtract(prediction_open, label_open)))

	return loss

#computes the cost for each row separately
def compute_cost_row(prediction,label):
	loss = tf.reduce_mean(tf.square(tf.subtract(prediction, label)),axis=1)

	return loss


# Create placeholders for input
def create_placeholders(n_x, n_y, n_a, T_x):
	X = tf.placeholder(tf.float32, shape=(T_x, None, n_x), name="X")
	Y = tf.placeholder(tf.float32, shape=(None, n_y), name="Y")
	a0 = tf.placeholder(tf.float32, [None, n_a])

	return X, Y, a0

#Load data
perm = random.getrandbits(32)
print("seed: " + str(perm))
data = preprocessData(test_ratio=0.1,seed=perm,num_steps=4)
data_unnorm = preprocessData(test_ratio=0.5,seed=perm,num_steps=4,normalized=False)
seq = data.to_list()
seq_unnorm = data_unnorm.to_list()

_,x,y,seq_std = data.prepare_data(seq)

#print("seq_std: " +str(seq_std))

#data splitting and print attributes in command line
#for one stock fed in
_,x_unnorm, y_unnorm,_ = data_unnorm.prepare_data(seq_unnorm)
train_x,train_y,test_x,test_y,perm_norm = data.split_data(x,y)
_,_,test_x_unnorm, test_y_unnorm,perm_unnorm = data.split_data(x_unnorm,y_unnorm)
print("train_x_shape: " + str(train_x.shape))
print("train_y_shape: " + str(train_y.shape))
print("test_x_shape: " + str(test_x.shape))
print("test_y_shape: " + str(test_y.shape))


#print("Test set unnormalized: ")
#print(test_x_unnorm)
print("permutations used are equal: " + str(np.array_equal(perm_norm,perm_unnorm)))

print("Ground truth: ")

row = -5 #which row to display
print("Multiplier factor: " + str(test_x_unnorm[-1][row][-1]))
for i in range(0,test_x_unnorm.shape[0]):
	print(test_x_unnorm[i][row])

print("Ground truth predicted: ")
print(test_y_unnorm[row])

_,num_minibatches_ = data.minibatches(x, y, 20)
print("num_minibatches: " + str(num_minibatches_))

#used for testing shapes

"""
n_x = x.shape[2]
m = x.shape[1]
n_y = y.shape[1]
X, Y, a0 = create_placeholders(n_x, n_y, 5, 4)
inputs_series = tf.unstack(X, axis=0)
parameters = initialize_parameters(5, n_x, n_y)
prediction, _, _ = rnn_forward(inputs_series, a0, parameters)
print(prediction.shape)
#cost_row = compute_cost_row(prediction,Y)
cost_column = compute_cost_column(prediction,Y)
print(cost_column.shape)
#print(cost_row.shape)
"""


def model_1(X_train, Y_train, X_test, Y_test, state_size, mini_batch_size, num_epochs, print_train_cost=True,print_val_cost=True):
	# input dimensions
	n_x = X_train.shape[2]
	m = X_train.shape[1]
	n_y = Y_train.shape[1]
	parameters = initialize_parameters(state_size, n_x, n_y)

	# computation graph
	X, Y, a0 = create_placeholders(n_x, n_y, state_size, 4)
	inputs_series = tf.unstack(X, axis=0)
	prediction, _, _ = rnn_forward(inputs_series, a0, parameters)
	total_loss_row = compute_cost_row(prediction,Y)
	print(total_loss_row.shape)
	total_loss_column = compute_cost_column(prediction,Y)
	total_loss = compute_cost(prediction, Y)

	# optimizer
	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = 0.001
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
										   2000, 0.9, staircase=True)

	optimizer = tf.train.AdamOptimizer(starter_learning_rate)
	train_step = optimizer.minimize(total_loss,global_step=global_step)

	final_cost = -1

	with tf.Session() as sess2:

		sess2.run(tf.global_variables_initializer())

		mini_batches, num_minibatches = data.minibatches(X_train, Y_train, mini_batch_size)
		print("num_train_minibatches: " + str(num_minibatches))
		pre_epoch_cost = 0

		for minibatch in mini_batches:
			(minibatch_X, minibatch_Y) = minibatch
			# print(minibatch_X.shape)
			batch_size = minibatch_X.shape[1]
			minibatch_cost = sess2.run([total_loss],
											  feed_dict={X: minibatch_X, Y: minibatch_Y,
														 a0: np.zeros((batch_size, state_size))})

			#print(minibatch_cost)
			pre_epoch_cost += minibatch_cost[0] / num_minibatches

		print("Pre-epoch cost: "+str(pre_epoch_cost))
			

		for epoch in range(num_epochs):

			mini_batches, num_minibatches = data.minibatches(X_train, Y_train, mini_batch_size)
			epoch_cost = 0

			for minibatch in mini_batches:
				(minibatch_X, minibatch_Y) = minibatch
				#print(minibatch_X.shape)
				batch_size = minibatch_X.shape[1]
				_, minibatch_cost = sess2.run([train_step, total_loss],
											  feed_dict={X: minibatch_X, Y: minibatch_Y,
														 a0: np.zeros((batch_size, state_size))})

				epoch_cost += minibatch_cost / num_minibatches

			final_cost = epoch_cost

			if print_train_cost == True:
				print("Cost after epoch %i: %f" % (epoch, epoch_cost))

			if print_val_cost == True and epoch%10 == True:

				prediction_val_norm, total_loss_val,total_loss_val_row,total_loss_val_column = sess2.run([prediction,total_loss,total_loss_row,total_loss_column],feed_dict={X: X_test, Y: Y_test,
					a0: np.zeros((Y_test.shape[0],state_size))})

				#check this
				#prediction = np.exp(prediction_val_norm[-1])*test_y_unnorm[-2][-1]
				print("Validation_loss: " + str(total_loss_val))
				#print("Individual validation losses: " + str(total_loss_val_row))
				#print(total_loss_val_row.shape)
				#print("Average of ind val losses: " + str(np.average(total_loss_val_row)))
				print("Total loss open: " + str(total_loss_val_column))


				print("Uncorrected prediction: " + str(prediction_val_norm[row] ) )
				print("Corrected_prediction: " + str(sess2.run( tf.multiply(  tf.exp( tf.multiply( prediction_val_norm[row], seq_std ) ), test_x_unnorm[-1][row][-1]    )      )   )  )

				#print(prediction_val_norm.shape)
				#print("Example prediction: "+ str(sess2.run( tf.multiply( tf.exp(prediction_val_norm[row]),test_x_unnorm[4][row][-1]  ) ) ) )
				


				#perf_str = "Validation_prediction: %.2f, Validation_loss: %.2f" % (prediction_val_norm, total_loss_val)
				#print(perf_str)

				#print("Validation cost: " + str(sess2.run([total_loss],feed_dict={X: X_test, Y: Y_test,
					#a0: np.zeros((Y_test.shape[0],state_size))  } ) ))

		

	return parameters, final_cost


model_1(train_x, train_y, test_x, test_y, 5, 20, 25)

def print_success(trial_count):

	success_count = 0

	for i in range(trial_count):

		_, final_cost = model_1(x, y, 5, 30, 30,print_cost=False)
		tf.reset_default_graph()

		if final_cost < 400:
			success_count+=1
			print("Model converged!: " + str(final_cost))
		else:
			print("Diverged!: " + str(final_cost))

	print(success_count/trial_count)

#print_success(50)

# problem with multiple stocks being fed in?
# use regularization?
# minibatches do seem to be taking care of randomization of training data


"""
#random tests
sess = tf.Session()

A = sess.run( tf.concat( [tf.constant([[2,3,4],[5,6,7]]), tf.constant([[1,2,3],[9,8,7]])], 1 ) )
#print(A.shape)
#print(A)
#print(A.shape)
B = tf.constant(  [[ [1,2,5], [3,4,5]  ],   [ [5,6,6], [7,8,9]  ] ]   )
#print(B.shape)

C = sess.run(tf.constant([[  [1],[2],[3]  ],[  [4],[5],[6]   ]]))
print(C)
print(C.shape) 

E = tf.unstack(C, axis=0)
print(sess.run(E))

for state in E:
	print(sess.run(state))
	print(state.shape)

"""
