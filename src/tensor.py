from pymongo import MongoClient
import tensorflow as tf
import numpy as np
import random
from random import shuffle


def showVariable(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    temp = v.eval()
    sess.close()
    return temp


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def next_batch(num):
    x_sample = []
    y_sample = []
    for i in range(num):
        x_sample.append(x_all.pop())
        y_sample.append(y_all.pop())
    return x_sample, y_sample


def next_batch_rand(num):
    x_sample = []
    y_sample = []
    for i in range(num):
        r = random.randrange(0, len(x_all))
        x_sample.append(x_all[r])
        y_sample.append(y_all[r])
    return x_sample, y_sample


def shuffle2(x, y):
    listx = []
    listy = []
    index = range(len(x))
    shuffle(index)
    for i in index:
        listx.append(x[i])
        listy.append(y[i])
    return listx, listy


def jsonToTensorX(json, index):
    data = json['data']
    x_temp = np.zeros(260)
    for j in data:
        x_temp[index[j['BSSID']]] = np.float(1+(35+j['level'])/64.0)
    return x_temp


def jsonToTensorY(json, zones):
    data = json['zone']
    y_temp = np.zeros(28)
    y_temp[zones[data]] = np.float(1)
    return y_temp


def takeTestSamples(num, x):
    x_test = []
    for i in range(num):
        r = random.randrange(0, len(x_all))
        x_test.append(x_all[r])
        x_all.pop(r)
    return x_test

client = MongoClient()
db = client.test
wifi = db.wifi
BSSID = set()
num = 0

for i in wifi.find():
    for j in i['data']:
        BSSID.add(j['BSSID'])

index = {}
zones = {0: 0, 1: 1, 2: 2, 3: 3, 10: 4, 13: 5, 20: 6, 23: 7, 40: 8, 41: 9,
         42: 10, 43: 11, 50: 12, 53: 13, 60: 14, 63: 15, 70: 16, 71: 17,
         72: 18, 73: 19, 80: 20, 83: 21, 90: 22, 93: 23, 100: 24, 101: 25,
         102: 26, 103: 27}

for i in range(len(BSSID)):
    index[BSSID.pop()] = i

for i in wifi.find():
    for j in i['data']:
        BSSID.add(j['BSSID'])
        num += 1

varI = len(BSSID)
varO = 2
var1 = 512
var2 = 512
K = 12

print("Input Dim : "+str(len(BSSID)))

x1 = tf.placeholder("float", [None, varI])
x2 = tf.placeholder("float", [None, varI])
w = weight_variable([varI, var1])
w2 = weight_variable([var1, var2])
v = weight_variable([var2, varO])
b1 = bias_variable([var1])
b2 = bias_variable([var2])

h1 = tf.nn.relu(tf.matmul(x1, w)+b1)
h2 = tf.nn.relu(tf.matmul(x2, w)+b1)
g1 = tf.nn.relu(tf.matmul(h1, w2)+b2)
g2 = tf.nn.relu(tf.matmul(h2, w2)+b2)
o1 = tf.matmul(g1, v)
o2 = tf.matmul(g2, v)
y = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(o1, o2)), reduction_indices=1))
y_ = tf.placeholder("float", [None])
reg_term = 0
loss = tf.reduce_sum(tf.square(tf.sub(y, y_)))+reg_term*(tf.nn.l2_loss(w)+tf.nn.l2_loss(v)+tf.nn.l2_loss(b1))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
init = tf.initialize_all_variables()

saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)

print("Sample constructed")

x_all = []
for i in wifi.find():
        x_all.append(jsonToTensorX(i, index))

#sample_size = 5000
#x_all = takeTestSamples(sample_size, x_all)

sample_size = len(x_all)

total_rep = 10000
for rep in range(total_rep):
    length = len(x_all)
    size = 10
    x1_t = []
    x2_t = []
    y_t = []
    for _ in range(size):
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
    if rep % 1000 == 0 and rep != 0:
        print("%d / %d" % (rep, total_rep))

# for i in range(1000):
#     size = 10
#     for j in range(100):
#         x1_t = []
#         x2_t = []
#         y_t = []
#         for k in range(size):
#             if i == j*size+k:
#                 continue
#             x1_ = x_all[i]
#             x2_ = x_all[j*size+k]
#             x1_t.append(x1_)
#             x2_t.append(x2_)
#             y_t.append(np.linalg.norm(x1_-x2_))
#         sess.run(train_step, feed_dict={x1: x1_t, x2: x2_t, y_: y_t})
#     if i % 10 == 0:
#         print("%d / %d" % (i*100, 100000))

print("Training done")
print("rep%d-reg%f-s%d-h%d" % (total_rep, reg_term, sample_size, var1))
print(sess.run(o1, feed_dict={x1: x_all}))
np.savetxt('weights1.csv', showVariable(w))
np.savetxt('weights2.csv', showVariable(w2))
np.savetxt('weights3.csv', showVariable(v))
np.savetxt('bias1.csv', showVariable(b1))
np.savetxt('bias2.csv', showVariable(b2))
np.savetxt('dataY.csv', sess.run(o1, feed_dict={x1: x_all}), delimiter=',')
