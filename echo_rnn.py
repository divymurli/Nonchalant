import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Adapted from https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

#print(num_batches)

#Generate some toy data set
def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    #print(x)
    y = np.roll(x, echo_step)
    #print(y)

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))
    #print(x)
    #print(y)

    return (x, y)

#Initialize the parameters of this RNN
def initialize_parameters(n_a,n_x,n_y):

	parameters = {}
	parameters["W"] = tf.get_variable("W",shape=[n_a+n_x,n_a],initializer = tf.contrib.layers.xavier_initializer())
	parameters["b"] = tf.get_variable("b", shape=[1,n_a],initializer=tf.zeros_initializer())
	parameters["Wy"] = tf.get_variable("Wy",shape=[n_a,n_y], initializer = tf.contrib.layers.xavier_initializer())
	parameters["by"] = tf.get_variable("by", shape=[1,n_y],initializer=tf.zeros_initializer())

	return parameters


#Single RNN cell
def rnn_cell_forward(xt,a_prev,parameters):

	W = parameters['W']
	b = parameters['b']
	Wy = parameters['Wy']
	by = parameters['by']

	concatenated = tf.concat([a_prev,xt],1)

	a_next = tf.nn.tanh(tf.matmul(concatenated,W)+b)

	#yt_pred = tf.nn.softmax(tf.matmul(a_next,Wy)+by)

	return a_next

#RNN forward pass
def rnn_forward(inputs_series,init_state,parameters):

	#x: shape (m,n_x)
	#a0: shape (m,n_a)


	current_state = init_state
	states_series = []
	
	for current_input in inputs_series: #loop over time steps
		current_input = tf.reshape(current_input,[batch_size,1])
		next_state = rnn_cell_forward(current_input,current_state,parameters)
		states_series.append(next_state)
		
		#states_series.append(current_state)
		current_state = next_state


	logits_series = [tf.matmul(state, parameters['Wy']) + parameters['by'] for state in states_series]
	predictions_series = [tf.nn.softmax(logits) for logits in logits_series]


	return logits_series, states_series, current_state


#Cost function
def compute_cost(logits_series,label_series):

	losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels) for logits, labels in zip(logits_series,labels_series)]
	total_loss = tf.reduce_mean(losses)

	return total_loss

#Create placeholders for minibatch input
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

inputs_series = tf.unstack(batchX_placeholder,axis=1)
labels_series = tf.unstack(batchY_placeholder,axis=1)

init_state = tf.placeholder(tf.float32, [batch_size,state_size])

##Build computation graph and do training

parameters = initialize_parameters(state_size,1,num_classes)

logits_series,_,current_state = rnn_forward(inputs_series,init_state,parameters)

total_loss = compute_cost(logits_series,labels_series)

optimizer = tf.train.AdagradOptimizer(0.3)
train_step = optimizer.minimize(total_loss)

with tf.Session() as sess2:

	sess2.run(tf.global_variables_initializer())

	loss_list = []

	for epoch in range(20):

		x,y = generateData()

		print("Epoch: "+str(epoch))

		_current_state = np.zeros((batch_size, state_size))

		for batch in range(num_batches):
			start_idx = batch*truncated_backprop_length
			end_idx = start_idx + truncated_backprop_length

			batchX = x[:,start_idx:end_idx]
			batchY = y[:,start_idx:end_idx]

			_total_loss, _current_state, _ = sess2.run([total_loss, current_state,train_step],
				feed_dict = {batchX_placeholder:batchX,batchY_placeholder:batchY,init_state:_current_state})

			loss_list.append(_total_loss)

			if batch%100 == 0:
				print("Step: "+str(batch)+" Loss: " +str(_total_loss))
	#plt.plot(loss_list)
	#plt.show()





	




#testing grounds

"""

init = tf.global_variables_initializer()
sess = tf.Session()

A = sess.run( tf.concat( [tf.constant([[2,3,4],[5,6,7]]), tf.constant([[1,2,3],[9,8,7]])], 1 ) )
#print(A.shape)
print(A)
B = tf.constant(  [[ [1,2,5], [3,4,5]  ],   [ [5,6,6], [7,8,9]  ] ]   )
#print(B.shape)

E = tf.unstack(A, axis=1)
print(sess.run(E))

for state in E:
	state = tf.reshape(state,[2,1])
	print(sess.run(state))




indices = [1]

C = tf.gather_nd(B,indices)
#print(C)

D = tf.constant([[1],[0],[0]])





#E = tf.get_variable("E",shape=[2,1,1],initializer=tf.zeros_initializer())
#F = tf.Variable(tf.zeros([2,1,1]))
#sess.run(init)
#sess.run(E.initializer)











#print(E.eval(sess))
#print(sess.run(tf.gather_nd(E,indices))

#sess.run(tf.assign(tf.gather_nd(E,indices),[[1]]))







x = tf.Variable(0)
y = tf.assign(x, 1)
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    print(sess2.run(x))
    print(sess2.run(y))
    #print sess2.run(x)

x = tf.Variable(
    [], # A list of scalar
    dtype=tf.int32,
    validate_shape=False, # By "shape-able", i mean we don't validate the shape so we can change it
    trainable=False
)
# I build a new shape and assign it to x
concat = tf.concat([x, [1]], 0)
assign_op = tf.assign(x, concat, validate_shape=False) # We force TF, to skip the shape validation step

with tf.Session() as sess3:
  sess3.run(tf.global_variables_initializer())
    
  for i in range(5):
    print('x:', sess3.run(x), 'shape:', sess3.run(tf.shape(x)))
    sess3.run(assign_op)




#print(sess.run(tf.gather_nd(B,indices)))
#print(C.shape)

#print(sess.run(tf.matmul(C,D)))

#print(A)

"""
	







