import tensorflow as tf
import gensim
import numpy as np
import pandas as pd
import dataparse

from gensim.models.keyedvectors import KeyedVectors
wordModel = KeyedVectors.load('model/w2v_model')
featureModel = KeyedVectors.load('model/f2v_model')

import dataparse as ps
#train data parsing
data_path = 'data/train.txt'
parser = ps.Parser(data_path)
parser.parse()
sentences = parser.sentences

category = parser.catagory
labels = parser.labels


#train set
input_data = []
entity_data = []

n_class = len(category)  #number of labels

max_len = 336
vec_size = 50
learning_rate = 0.01
empty = 'empty'
label_dic = {
    'O' : [1,0,0,0,0,0,0],
    'B_OG' : [0,1,0,0,0,0,0],
    'B_DT' : [0,0,1,0,0,0,0],
    'B_PS' : [0,0,0,1,0,0,0],
    'B_LC' : [0,0,0,0,1,0,0],
    'B_TI': [0,0,0,0,0,1,0],
    'I': [0,0,0,0,0,0,1],
    'empty':[0,0,0,0,0,0,0]
}
for line in sentences:
    w2c = []
    if len(line) < max_len:
        for i in range(max_len - len(line)):
            line.append(empty)
    for i in line:
        if empty == i:
            w2c.append(np.zeros(vec_size))
        else:
            w2c.append(wordModel[i])
    input_data.append(w2c)

for line in labels:
    l = []
    if len(line) < max_len:
        for i in range(max_len-len(line)):
            line.append(empty)
    for i in line:
        l.append(label_dic[i])
    entity_data.append(l)

x_data = np.array(input_data)
y_data = np.array(entity_data)

print(x_data.shape)
print(y_data.shape)

category = np.array(category)
# ['O' 'B_OG' 'I' 'B_DT' 'B_PS' 'B_LC' 'B_TI']


X = tf.placeholder(tf.float32, [None, max_len, 50])  # X data
Y = tf.placeholder(tf.float32, [None, max_len, n_class])  # Y label




cell = tf.contrib.rnn.BasicLSTMCell(max_len, state_is_tuple=True)
#cell = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
initial_state = cell.zero_state(1, tf.float32)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(outputs, n_class, activation_fn=None)

print("Y_predic looks like %s" % (Y_pred))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_pred))
optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
corr = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
accr = tf.reduce_mean(tf.cast(corr, tf.float32))


#Saver
save_dir = "model/"
saver = tf.train.Saver(max_to_keep=3)

# INITIALIZER
init = tf.global_variables_initializer()
print ("FUNCTIONS READY")

iteration = 50
save_step = 2
batch_size = 20
n_train = x_data.shape[0]
show_step = 2
# LAUNCH THE GRAPH
sess = tf.Session()
sess.run(init)

print ("Training start")
for iter in range(iteration):
    total_batch = int(n_train / batch_size)
    randpermlist = np.random.permutation(n_train)
    sum_cost = 0
    for i in range(total_batch):
        randidx = randpermlist[i * batch_size:min((i + 1) * batch_size, n_train - 1)]
        batch_xs = x_data[randidx, :]
        batch_ys = y_data[randidx, :]
        feeds = {X: batch_xs, Y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        sum_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = sum_cost / total_batch

    if iter % show_step == 0:
        print("[%d/%d] iterations" % (iter, iteration))
        print("cost {0}".format(avg_cost))

    if iter % save_step == 0:
        saver.save(sess, save_dir+"/ner.ckpt-"+str(iter))

print("finish training")