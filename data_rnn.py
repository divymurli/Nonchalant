import quandl
import tensorflow as tf
import math
import numpy as np
import pandas as pd

quandl.ApiConfig.api_key = 'upvv8dx3pwLpm_Rxi8iP'
b = quandl.get('FSE/EON_X', start_date='2016-05-09', end_date='2018-06-22')
a = quandl.get('FSE/EON_X', start_date='2012-05-09', end_date='2018-06-22',column_index='3',returns='numpy')


#print(b)

print(a.shape[0] - a.shape[0]%5)



b = [a[i][1] for i in range(a.shape[0])]
#print(b)

b1 = [a[i][1] for i in range(a.shape[0] - a.shape[0]%5)]
#print(b1)

c = [b1[i:i + 5] for i in range(0, len(b1), 5)]
#print(c)

d = [b1[i:i + 4] for i in range(0, len(b1), 5)]
#print(d)

e = [b1[i] for i in range(len(b)) if (i+1)%5==0]
#print(e)

def generateData(T_x):
	a1 = [a[i][1] for i in range(a.shape[0] - a.shape[0]%T_x)]
	x = np.array([a1[i:i+T_x-1] for i in range(0,len(a1),T_x)])
	x = x.T.reshape((T_x-1,x.shape[0],1))
	y = np.array([[a1[i] for i in range(len(a1)) if (i+1)%T_x==0]]).T


	return x, y

x1,y1 = generateData(5)

#print(x1)
#print(y)

#perm = list(np.random.permutation(55))
#print(x[perm,:])



def minibatches(X,Y,mini_batch_size):

	m = X.shape[1]
	mini_batches = []
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:,permutation,:]
	shuffled_Y = Y[permutation,:]
	num_complete_minibatches = math.floor(m/mini_batch_size)
	num_minibatches = num_complete_minibatches

	for k in range(0, num_complete_minibatches):
        
		mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size,:]
		mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
      
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	#handling last minibatch
	if m % mini_batch_size != 0:
        
		mini_batch_X = shuffled_X[:,0:(m - mini_batch_size*num_complete_minibatches),:]
		mini_batch_Y = shuffled_Y[0:(m - mini_batch_size*num_complete_minibatches),:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
		num_minibatches +=1

	return mini_batches, num_minibatches


#mini_batches, num_minibatches = minibatches(x1,y1,mini_batch_size = 20)
#print(num_minibatches)

#print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
#print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
#print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
#print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
#print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
#print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))

def initialize_parameters(n_a,n_x,n_y):

	parameters = {}
	parameters["W"] = tf.get_variable("W",shape=[n_a+n_x,n_a],initializer = tf.contrib.layers.xavier_initializer())
	parameters["b"] = tf.get_variable("b", shape=[1,n_a],initializer=tf.zeros_initializer())
	parameters["Wy"] = tf.get_variable("Wy",shape=[n_a,n_y], initializer = tf.contrib.layers.xavier_initializer())
	parameters["by"] = tf.get_variable("by", shape=[1,n_y],initializer=tf.zeros_initializer())

	return parameters

#All encoder cells
def rnn_cell_forward(xt,a_prev,parameters):

	W = parameters['W']
	b = parameters['b']
	
	concatenated = tf.concat([a_prev,xt],1)

	a_next = tf.nn.tanh(tf.matmul(concatenated,W)+b)

	return a_next

def rnn_forward(inputs_series,init_state,parameters):

	#x: shape (m,n_x)
	#a0: shape (m,n_a)


	current_state = init_state
	states_series = []
	
	for current_input in inputs_series: #loop over time steps
		#current_input = tf.reshape(current_input,[batch_size,1])
		next_state = rnn_cell_forward(current_input,current_state,parameters)
		states_series.append(next_state)
		
		#states_series.append(current_state)
		current_state = next_state


	logit = tf.matmul(next_state, parameters['Wy']) + parameters['by']
	prediction = tf.nn.relu(logit)

	return prediction, states_series, current_state

#Cost function
def compute_cost(prediction,label):

	loss = tf.reduce_mean(tf.square(tf.subtract(prediction,label)))

	return loss

#Create placeholders for input

def create_placeholders(n_x,n_y,n_a,T_x):

	X = tf.placeholder(tf.float32,shape=(T_x,None,n_x),name="X")
	Y = tf.placeholder(tf.float32,shape=(None,n_y),name="Y")
	a0 =  tf.placeholder(tf.float32,[None,n_a])

	return X, Y, a0

X, Y, a0 = create_placeholders(1,1,5,4)
state_size = 5

inputs_series = tf.unstack(X,axis=0)


#Build computation graph
parameters = initialize_parameters(5,1,1)
print(parameters["W"].shape)
print(a0.shape)
print(X.shape)
#print(inputs_series)


prediction,_,_ = rnn_forward(inputs_series,a0,parameters)
print("prediction: " + str(prediction.shape))

total_loss = compute_cost(prediction,Y)

optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(total_loss)

#Do training
with tf.Session() as sess2:

	sess2.run(tf.global_variables_initializer())

	x,y = generateData(5)
	m = x.shape[1]

	for epoch in range(500):

		print("Epoch: " + str(epoch))
		mini_batches, num_minibatches = minibatches(x,y,mini_batch_size = 20)
		epoch_cost = 0

		for minibatch in mini_batches:

			(minibatch_X, minibatch_Y) = minibatch
			#print(minibatch_X.shape)
			batch_size = minibatch_X.shape[1] 
			_,minibatch_cost = sess2.run([train_step,total_loss],
				feed_dict={X:minibatch_X, Y: minibatch_Y,a0: np.zeros((batch_size,state_size))})

			epoch_cost+= minibatch_cost/num_minibatches

		print ("Cost after epoch %i: %f" % (epoch, epoch_cost))




















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














