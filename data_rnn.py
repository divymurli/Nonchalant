import quandl
import tensorflow as tf
import math
import numpy as np
import pandas as pd

# get data
quandl.ApiConfig.api_key = 'upvv8dx3pwLpm_Rxi8iP'
frankfurt_data = quandl.get('FSE/EON_X', start_date='2012-05-09', end_date='2018-06-22')
scalar_input = quandl.get('FSE/EON_X', start_date='2012-05-09', end_date='2018-06-22', column_index='3',
                          returns='numpy')
bombay_stock_exchange = quandl.get('XBOM/500010', start_date='2018-05-20', end_date='2018-07-02')
hongkong_stock_exchange = quandl.get('XHKG/00005', start_date = '2000-05-20',end_date = '2018-07-02')
vectorized_input = bombay_stock_exchange.iloc[:,0:4]
print(vectorized_input)

def naive_normalization(pandas_frame):

	mean = pandas_frame.mean().mean()
	std = pandas_frame.stack().std()

	normalized_frame = (pandas_frame - mean)/std
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


def print_mini_batch_sizes(mini_batches, num_minibatches):
    for i in range(num_minibatches):
        print ("shape of mini_batch" + str(i) + ": " + str(mini_batches[i][0].shape))


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

    logit = tf.matmul(next_state, parameters['Wy']) + parameters['by']
    prediction = tf.nn.relu(logit)

    return prediction, states_series, current_state


# Cost function
def compute_cost(prediction, label):
    loss = tf.reduce_mean(tf.square(tf.subtract(prediction, label)))
    #loss =  tf.square(tf.subtract(prediction, label))

    return loss


# Create placeholders for input
def create_placeholders(n_x, n_y, n_a, T_x):
    X = tf.placeholder(tf.float32, shape=(T_x, None, n_x), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, n_y), name="Y")
    a0 = tf.placeholder(tf.float32, [None, n_a])

    return X, Y, a0


# Do training

training_frame = convert_to_numpy(vectorized_input, 0, 4)
print(training_frame.shape)

x, y = generateData2(5, training_frame)

print(x)
print(y)

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
"""

def model_1(X_train, Y_train, state_size, mini_batch_size, num_epochs, print_cost=True):
    # input dimensions
    n_x = X_train.shape[2]
    m = X_train.shape[1]
    n_y = Y_train.shape[1]
    parameters = initialize_parameters(state_size, n_x, n_y)

    # computation graph
    X, Y, a0 = create_placeholders(n_x, n_y, state_size, 4)
    inputs_series = tf.unstack(X, axis=0)
    prediction, _, _ = rnn_forward(inputs_series, a0, parameters)
    total_loss = compute_cost(prediction, Y)

    # optimizer
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(total_loss)

    final_cost = -1

    with tf.Session() as sess2:

        sess2.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):

            mini_batches, num_minibatches = minibatches(X_train, Y_train, mini_batch_size)
            epoch_cost = 0

            for minibatch in mini_batches:
                (minibatch_X, minibatch_Y) = minibatch
                # print(minibatch_X.shape)
                batch_size = minibatch_X.shape[1]
                _, minibatch_cost = sess2.run([train_step, total_loss],
                                              feed_dict={X: minibatch_X, Y: minibatch_Y,
                                                         a0: np.zeros((batch_size, state_size))})

            epoch_cost += minibatch_cost / num_minibatches

            final_cost = epoch_cost

            if print_cost == True:
            	print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
 		

    return parameters, final_cost


#model_1(x, y, 4, 30, 10)

def print_success(trial_count):

	success_count = 0

	for i in range(trial_count):

		_, final_cost = model_1(x, y, 5, 30, 10,print_cost=False)
		tf.reset_default_graph()

		if final_cost < 100:
			success_count+=1
			print("Model converged!: " + str(final_cost))
		else:
			print("Diverged!: " + str(final_cost))

	print(success_count/trial_count)


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
