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
useBatch = True
batch_size = 64
n_class = len(category)+1  #number of labels
n_hidden = n_class
#n_hidden = 100
max_len = 336
vec_size = 50
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
    sequence_length = np.array(sequence_length)
    return x_data, y_data, sequence_length

x_data, y_data, sequence_length = getData(sentences)

print("x data format", x_data[1:2, :].shape)
X = tf.placeholder(
    tf.float32, [None, max_len, vec_size])  # X one-hot
Y = tf.placeholder(tf.int32, [None, max_len])  # Y label
seq_len = tf.placeholder(tf.int32, [None])

def RNN_with_Fully(X):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(
        cell, X, initial_state=initial_state, dtype=tf.float32)

    # FC layer
    X_for_fc = tf.reshape(outputs, [-1, n_hidden])

    outputs = tf.contrib.layers.fully_connected(
        inputs=X_for_fc, num_outputs=n_class, activation_fn=None)
    return outputs

def BiRNN(X, seq_len):
    lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
    lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)

    # initial_state_fw = lstm_cell_fw.zero_state(batch_size, tf.float32)
    # initial_state_bw = lstm_cell_bw.zero_state(batch_size, tf.float32)
    lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw] * 1)
    lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw] * 1)

    # bidirectional_ rnn
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        lstm_cell_fw,
        lstm_cell_bw,
        inputs=X,
        dtype=tf.float32,
        sequence_length=seq_len
    )
#8/7 concat부터

    outputs_forward, outputs_backward = outputs
    #outputs = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
    outputs = tf.add(outputs_forward, outputs_backward)
     # FC layer

    X_for_fc = tf.reshape(outputs, [-1, n_hidden])

    outputs = tf.contrib.layers.fully_connected(
        inputs=X_for_fc, num_outputs=n_class, activation_fn=None)

    return outputs

# reshape out for sequence_loss
#outputs = RNN_with_Fully(X)
outputs = BiRNN(X, seq_len)
if useBatch:
    outputs = tf.reshape(outputs, [batch_size, max_len, n_class])
    weights = tf.ones([batch_size, max_len])
else:
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

iteration = 30
save_step = 2
n_train = x_data.shape[0]
show_step = 1

for iter in range(iteration):
    if useBatch:
        total_batch = int(n_train / batch_size)
        randpermlist = np.random.permutation(n_train)
        sum_cost = 0
        for i in range(total_batch):
            randidx = randpermlist[i * batch_size:min((i + 1) * batch_size, n_train - 1)]
            batch_xs = x_data[randidx, :]
            batch_ys = y_data[randidx, :]
            batch_seq = sequence_length[randidx]
            _, show_loss = sess.run([train, loss], feed_dict={X: batch_xs, Y: batch_ys, seq_len:batch_seq})

        if iter % show_step == 0:
            print("[{}/{}] loss : {}".format (iter,iteration, show_loss))
            predic = sess.run(prediction, feed_dict={X:x_data, Y:y_data, seq_len:sequence_length})
            print(predic.shape)
            '''print(batch_xs.shape)
            print(batch_xs[:1].shape)
            y_pre1= sess.run(prediction, feed_dict={X:batch_xs, Y:batch_ys, seq_len:batch_seq} )
            print("predict1")
            print(y_pre1[:1])
            print("label")
            print(batch_ys[:1])'''
    else:
        _, show_loss = sess.run([train, loss], feed_dict={X: x_data, Y: y_data, seq_len: sequence_length})


'''
tf.add
[21/30] loss : 0.054476987570524216
[22/30] loss : 0.047998007386922836
[23/30] loss : 0.042630262672901154
[24/30] loss : 0.05896882340312004
[25/30] loss : 0.05155254527926445
[26/30] loss : 0.03887255862355232
[27/30] loss : 0.04661808907985687
[28/30] loss : 0.04571153596043587
[29/30] loss : 0.03673136979341507
…
only use fw cell 
[20/30] loss : 0.046440403908491135
[21/30] loss : 0.04254528880119324
[22/30] loss : 0.03595905005931854
[23/30] loss : 0.04015754908323288
[24/30] loss : 0.046795379370450974
[25/30] loss : 0.045131634920835495
[26/30] loss : 0.04626455903053284
[27/30] loss : 0.045871902257204056
[28/30] loss : 0.04608023166656494
[29/30] loss : 0.0387987419962883

basic RNN
[20/30] loss : 0.04517918452620506
[21/30] loss : 0.06283189356327057
[22/30] loss : 0.0440068356692791
[23/30] loss : 0.044367264956235886
[24/30] loss : 0.041829898953437805
[25/30] loss : 0.050393570214509964
[26/30] loss : 0.04674462974071503
[27/30] loss : 0.04105803370475769
[28/30] loss : 0.054583840072155

'''