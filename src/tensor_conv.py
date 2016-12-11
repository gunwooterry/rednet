from pymongo import MongoClient
import tensorflow as tf
import numpy as np
import random
import itertools
from random import shuffle

def normalize(v) :
	norm = np.linalg.norm(v)
	if norm==0:
		return v
	return v/norm

def weight_variable(shape) :
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
	
def bias_variable(shape) :
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W) :
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')
	
def max_pool_2x2(x) :
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	
def next_batch(num) :
	x_sample = []
	y_sample = []
	for i in range(num) :
		x_sample.append(x_all.pop())
		y_sample.append(y_all.pop())
	return x_sample, y_sample

def next_batch_rand(num) :
	x_sample = []
	y_sample = []
	for i in range(num) :
		r = random.randrange(0,len(x_all))
		x_sample.append(x_all[r])
		y_sample.append(y_all[r])
	
	return x_sample, y_sample

def shuffle2(x,y) :
	listx = []
	listy = []
	index = range(len(x))
	shuffle(index)
	for i in index :
		listx.append(x[i])
		listy.append(y[i])
	return listx, listy

def jsonToTensorX(json, index) :
	data = json['data']
	x_temp = np.zeros(260)
	for j in data :	
		x_temp[index[j['BSSID']]] = np.float(1+(35+j['level'])/64.0)
	return x_temp
	
def jsonToTensorY(json, zones) :
	data = json['zone']
	y_temp = np.zeros(28)
	y_temp[zones[data]] = np.float(1)
	return y_temp
	
def takeTestSamples(num, x) :
	x_test = []
	for i in range(num) : 
		r = random.randrange(0,len(x_all))
		x_test.append(x_all[r])
		x_all.pop(r)
	return x_test

client = MongoClient()
db = client.test
wifi = db.wifi

BSSID = set()

num = 0

for i in wifi.find() :
	for j in i['data'] :
		BSSID.add(j['BSSID'])
		
index = {}
zones = {00:0,01:1,02:2,03:3,10:4,13:5,20:6,23:7,40:8,41:9,42:10,43:11,50:12,53:13,60:14,63:15,70:16,71:17,72:18,73:19,80:20,83:21,90:22,93:23,100:24,101:25,102:26,103:27}

for i in range(len(BSSID)) :
	index[BSSID.pop()] = i
		
for i in wifi.find() :
	for j in i['data'] :
		BSSID.add(j['BSSID'])
		num+=1

varI = len(BSSID)
varO = 2
var1 = 256
var3 = 10000
K = 12

print ("Input Dim : "+str(len(BSSID)))

x1 = tf.placeholder("float", [None, varI])
x2 = tf.placeholder("float", [None, varI])
w = weight_variable([varI,var1])
v = weight_variable([var1,varO])
b1 = bias_variable([var1])
b2 = bias_variable([varO])
h1 = tf.nn.relu(tf.matmul(x1, w)+b1)
h2 = tf.nn.relu(tf.matmul(x2, w)+b1)
o1 = tf.matmul(h1, v)+b2
o2 = tf.matmul(h2, v)+b2
y = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(o1,o2)),reduction_indices=1))
y_ = tf.placeholder("float")
train_step = tf.train.AdamOptimizer(1e-5).minimize(tf.square(tf.sub(y,y_)))
init = tf.initialize_all_variables()

saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)

print ("sample constructed")

x_all = []
KNN = []
KNN_toSave = []
for i in wifi.find() :
	x_all.append(jsonToTensorX(i, index))
	
x_all = takeTestSamples(1000,x_all)

cnt = 0
for x in x_all :
	temp = sorted(x_all,key=lambda val:np.linalg.norm(x-val))
	KNN+=[[x,t] for t in temp[1:K+1]]
	for t in temp :
		KNN_toSave.append(x)
		KNN_toSave.append(t)
	cnt += 1
	print ("KNN constructing : %d / %d" % (cnt, len(x_all)))

np.save('KNN.csv',KNN_toSave)
	
shuffle(KNN)

for i in range(var3) :
	size = 16
	length = len(KNN)
	x1_t = []
	x2_t = []
	y_t = []
	for j in range(size) :
		x1_ = KNN[(i*size+j)%(length)][0]
		x2_ = KNN[(i*size+j)%(length)][1]
		x1_t.append(x1_)
		x2_t.append(x2_)
		y_t.append(np.linalg.norm(x1_-x2_))
	sess.run(train_step, feed_dict={x1: x1_t,x2: x2_t, y_: y_t})
	print ("%d / %d" % (i,var3))

print ("Machine Learning Done")
np.savetxt('dataY.csv',sess.run(o1, feed_dict={x1: x_all}),delimiter=',')
