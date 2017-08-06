import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

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
sequence_length = []
batch_size = 20
n_class = len(category)+1  #number of labels
n_hidden = n_class
#n_hidden = 100
max_len = 336
vec_size = 50
learning_rate = 0.1
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

def getData(sentences):
    # convert word to vector
    input_data = []
    entity_data = []
    sequence_length = []
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
            for i in range(max_len - len(line)):
                line.append(empty)
        for i in line:
            l.append(np.argmax(label_dic[i], axis=0))
        entity_data.append(l)

    x_data = np.array(input_data)
    y_data = np.array(entity_data)
    print(x_data.shape)
    print(y_data.shape)
    return x_data, y_data, sequence_length

x_data, y_data, sequence_length = getData(sentences)

X = tf.placeholder(
    tf.float32, [None, max_len, vec_size])  # X one-hot
Y = tf.placeholder(tf.int32, [None, max_len])  # Y label

def RNN_with_Fully(X):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(
        cell, X, initial_state=initial_state, dtype=tf.float32)

    # FC layer
    X_for_fc = tf.reshape(outputs, [-1, n_hidden])
    # fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
    # fc_b = tf.get_variable("fc_b", [num_classes])
    # outputs = tf.matmul(X_for_fc, fc_w) + fc_b
    outputs = tf.contrib.layers.fully_connected(
        inputs=X_for_fc, num_outputs=n_class, activation_fn=None)
    return outputs

# reshape out for sequence_loss
outputs = RNN_with_Fully(X)

outputs = tf.reshape(outputs, [batch_size, max_len, n_class])

weights = tf.ones([batch_size, max_len])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

#train params
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print ("FUNCTIONS READY")

iteration = 10
save_step = 2
n_train = x_data.shape[0]
show_step = 1

for iter in range(iteration):
    total_batch = int(n_train / batch_size)
    randpermlist = np.random.permutation(n_train)
    sum_cost = 0
    for i in range(total_batch):
        randidx = randpermlist[i * batch_size:min((i + 1) * batch_size, n_train - 1)]
        batch_xs = x_data[randidx, :]
        batch_ys = y_data[randidx, :]
        #batch_seq = sequence_length[randidx]
        _, show_loss = sess.run([train, loss], feed_dict={X: batch_xs, Y: batch_ys})

    if iter % show_step == 0:
        print("loss :",show_loss)
        '''y_pre1,= sess.run(prediction, feed_dict={X:x_data[:1], Y:y_data[:1]} )
        print("predict1")
        print(y_pre1)
        print("label")
        print(y_data[:1])'''


