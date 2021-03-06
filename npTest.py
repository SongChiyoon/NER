import tensorflow as tf
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
wordModel = KeyedVectors.load('model/w2v_model')
featureModel = KeyedVectors.load('model/f2v_model')



import dataparse as ps
#train data parsing
data_path = 'data/train.txt'
parser = ps.Parser(data_path)
parser.parse()
sentences = parser.sentences
print(sentences[:1])
category = parser.catagory
labels = parser.labels


#train set
input_data = []
entity_data = []
sequence_length = []
batch_size = 20
n_class = len(category)+1  #number of labels

max_len = 336
vec_size = 16
learning_rate = 0.05
empty = 'empty'
label_dic = {
    'O' : [1,0,0,0,0,0,0,0],
    'B_OG' : [0,1,0,0,0,0,0,0],
    'B_DT' : [0,0,1,0,0,0,0,0],
    'B_PS' : [0,0,0,1,0,0,0,0],
    'B_LC' : [0,0,0,0,1,0,0,0],
    'B_TI': [0,0,0,0,0,1,0,0],
    'I': [0,0,0,0,0,0,1,0],
    'empty':[0,0,0,0,0,0,0,1]   #zero-padding's label
}
v2l = {
    0 : 'O',
    1 :'B_OG',
    2 : 'B_DT',
    3 : 'B_PS',
    4 : 'B_LC',
    5 : 'B_TI',
    6 : 'I',
    7 : 'empty'
}

# add zero padding using empty
# convert word to vector
for line in sentences:
    w2c = []
    sequence_length.append(len(line))
    if len(line) < max_len:
        for i in range(max_len - len(line)):
            line.append(empty)
    for i in line:
        if empty == i:
            w2c.append(np.zeros(vec_size))
        else:
            w2c.append(wordModel[i])
    input_data.append(w2c)

# remove zero padding
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

'''print(x_data.shape)
print(y_data.shape)
k = np.array([[[1,0,0,0,0], [0,0,0,0,1]]])
print(k)
print(np.argmax(k, axis=2))

print(k.shape)

l = np.argmax(y_data[:1], axis=2)
print(v2l[l])'''


category = np.array(category)
# ['O' 'B_OG' 'I' 'B_DT' 'B_PS' 'B_LC' 'B_TI']


X = tf.placeholder(tf.float32, [None, None, vec_size])  # X data
Y = tf.placeholder(tf.float32, [None, None, n_class])  # Y label
seq_len = tf.placeholder(tf.int64, [None])


#cell = tf.contrib.rnn.BasicLSTMCell(max_len, state_is_tuple=True)
cell_size = 100
lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(cell_size,state_is_tuple=True)
lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(cell_size, state_is_tuple=True)

#initial_state = cell.zero_state(1, tf.float32)
lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * 1)
lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * 1)

# bidirectional_ rnn
outputs, _  = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw,
            lstm_cell_bw,
            inputs=X,
            dtype=tf.float32,
            sequence_length=seq_len
        )
#outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, sequence_length=seq_len)
outputs_forward, outputs_backward = outputs
outputs = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')


Y_pred = tf.contrib.layers.fully_connected(outputs, n_class, activation_fn=tf.tanh)

print("Y_predic looks like %s" % (Y_pred))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_pred))
optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
corr = tf.equal(tf.argmax(Y_pred, 2), tf.argmax(Y, 2))
accr = tf.reduce_mean(tf.cast(corr, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Saver
save_dir = "model_2/"
saver = tf.train.Saver(max_to_keep=3)
#sess = tf.Session()
#saver.restore(sess, save_dir+"ner.ckpt-28")
# INITIALIZER
init = tf.global_variables_initializer()
print ("FUNCTIONS READY")

iteration = 5
save_step = 2
n_train = x_data.shape[0]
show_step = 2
pretrained = True
# LAUNCH THE GRAPH

#outputs2 = tf.reshape(tf.concat(1, outputs), [-1, max_len])

sess = tf.Session()
sess.run(init)
sequence_length = np.array(sequence_length)

for iter in range(iteration):
    total_batch = int(n_train / batch_size)
    randpermlist = np.random.permutation(n_train)
    sum_cost = 0
    for i in range(total_batch):
        randidx = randpermlist[i * batch_size:min((i + 1) * batch_size, n_train - 1)]
        batch_xs = x_data[randidx, :]
        batch_ys = y_data[randidx, :]
        batch_seq = sequence_length[randidx]
        sess.run(optm, feed_dict={X: batch_xs, Y: batch_ys, seq_len: batch_seq})

        if i % 20==0:
            of, ob = sess.run([outputs_forward, outputs_backward],
                              feed_dict={X:batch_xs, Y:batch_ys, seq_len:batch_seq} )
            out =  sess.run(outputs, feed_dict={X:batch_xs, Y:batch_ys, seq_len:batch_seq})
            Ypre = sess.run(Y_pred, feed_dict={X:x_data[:1], Y:y_data[:1], seq_len:sequence_length[:1]})
            print(of.shape)
            print(ob.shape)
            print(out.shape)
            Ypre = np.argmax(Ypre, axis=2)
            print(Ypre)
            print(np.argmax(y_data[:1], axis=2))

            #print(np.equal(np.argmax(Ypre, axis=0), np.argmax(y_data[:1], axis=0) ))


'''
print ("Training start")
for iter in range(iteration):
    sess.run(optm, feed_dict={X:x_data, Y:y_data})
    if iter % 5 == 0:
        print(sess.run(cost, feed_dict={X:x_data, Y:y_data}))'''

'''for iter in range(iteration):
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
'''


