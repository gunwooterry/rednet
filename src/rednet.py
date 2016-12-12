import tensorflow as tf
import numpy as np
import random
from random import shuffle


def init_weight(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def init_bias(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def show_var(v):
	sess = tf.InteractiveSession()
	v.initializer.run()
	temp = v.eval()
	sess.close()
	return temp


def shuffle_two(x, y):
	listx = []
	listy = []
	index = range(len(x))
	shuffle(index)
	for i in index:
		listx.append(x[i])
		listy.append(y[i])
	return listx, listy


def sample(num, x):
	x_test = []
	for i in range(num):
		r = random.randrange(0, len(x))
		x_test.append(x[r])
		x.pop(r)
	return x_test


def main():
	x_all = np.load("../input/dataX.npy")
	length = x_all.shape[0]
	input_dim = x_all.shape[1]
	output_dim = 2
	layer1 = 512
	layer2 = 512
	learning_rate = 1e-3
	total_rep = 10000
	batch_size = 10
	reg_term = 0
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	print("Input Dimension: " + str(input_dim))

	x1 = tf.placeholder("float", [None, input_dim])
	x2 = tf.placeholder("float", [None, input_dim])
	w1 = init_weight([input_dim, layer1])
	w2 = init_weight([layer1, layer2])
	v = init_weight([layer2, output_size])
	b1 = init_bias([layer1])
	b2 = init_bias([layer2])

	h1 = tf.nn.relu(tf.matmul(x1, w1)+b1)
	h2 = tf.nn.relu(tf.matmul(x2, w1)+b1)
	g1 = tf.nn.relu(tf.matmul(h1, w2)+b2)
	g2 = tf.nn.relu(tf.matmul(h2, w2)+b2)
	o1 = tf.matmul(g1, v)
	o2 = tf.matmul(g2, v)
	y = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(o1, o2)), reduction_indices=1))
	y_ = tf.placeholder("float", [None])
	loss = tf.reduce_sum(tf.square(tf.sub(y, y_)))+reg_term*(tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(v)+tf.nn.l2_loss(b1))
	train_step = optimizer.minimize(loss)
	init = tf.initialize_all_variables()

	# saver = tf.train.Saver()
	sess = tf.Session()
	sess.run(init)
	print("Sample constructed")

	for rep in range(total_rep):
		x1_t = []
		x2_t = []
		y_t = []
		for _ in range(batch_size):
			i = np.random.randint(0, length)
			j = np.random.randint(0, length)
			if i == j:
				continue
			x1_ = x_all[i]
			x2_ = x_all[j]
			x1_t.append(x_all[i])
			x2_t.append(x_all[j])
			y_t.append(np.linalg.norm(x1_-x2_))
		sess.run(train_step, feed_dict={x1: x1_t, x2: x2_t, y_: y_t})
		if rep % int(total_rep/10) == 0 and rep != 0:
			print("%d / %d" % (rep, total_rep))

	print("Training done")
	print("%d repetitions, regularization = %d" % (total_rep, reg_term))
	np.savetxt('../output/weights1.csv', show_var(w1))
	np.savetxt('../output/weights2.csv', show_var(w2))
	np.savetxt('../output/weights3.csv', show_var(v))
	np.savetxt('../output/bias1.csv', show_var(b1))
	np.savetxt('../output/bias2.csv', show_var(b2))
	np.savetxt('../output/dataY.csv', sess.run(o1, feed_dict={x1: x_all}), delimiter=',')


if __name__ == "__main__":
	main()
