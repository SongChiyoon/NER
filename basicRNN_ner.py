import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

from gensim.models.keyedvectors import KeyedVectors
wordModel = KeyedVectors.load('model/w2v_model')
featureModel = KeyedVectors.load('model/f2v_model')

import dataparse as ps
#trained data parsing
data_path = 'data/train.txt'
parser = ps.Parser(data_path)
parser.parse()
sentences = parser.sentences
category = parser.catagory
labels = parser.labels



#train set
batch_size = 32 #batch_size
n_class = 12  #number of labels
n_hidden = 25
max_len = 336
vec_size = 25
learning_rate = 0.005
drop_rate = 0.5
empty = 'empty'
label_dic = {
    'O' : [1,0,0,0,0,0,0,0,0,0,0,0],     #0
    'B_OG' : [0,1,0,0,0,0,0,0,0,0,0,0],  #1
    'I_OG' : [0,0,1,0,0,0,0,0,0,0,0,0],  #2
    'B_DT' : [0,0,0,1,0,0,0,0,0,0,0,0],  #3
    'I_DT' : [0,0,0,0,1,0,0,0,0,0,0,0],  #4
    'B_PS' : [0,0,0,0,0,1,0,0,0,0,0,0],  #5
    'I_PS' : [0,0,0,0,0,0,1,0,0,0,0,0],  #6
    'B_LC' : [0,0,0,0,0,0,0,1,0,0,0,0],  #7
    'I_LC' : [0,0,0,0,0,0,0,0,1,0,0,0],  #8
    'B_TI': [0,0,0,0,0,0,0,0,0,1,0,0],   #9
    'I_TI': [0,0,0,0,0,0,0,0,0,0,1,0],   #10
    'empty':[0,0,0,0,0,0,0,0,0,0,0,1]   #zero-padding's label
}
index2entity = {
    0 : '0',
    1 : 'B-OG',
    2 : 'I-OG',
    3 : 'B-DT',
    4 : 'I-DT',
    5 : 'B-PS',
    6 : 'I-PS',
    7 : 'B-LC',
    8 : 'I-LC',
    9 : 'B-TI',
    10 : 'I-TI',
    11 : 'empty'
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
        prev = None
        if len(line) < max_len:
            for i in range(max_len - len(line)):
                line.append(empty)
        for i in line:
            if i != "O" and i != "I" and i != "empty":
                prev = i.split("_")[1]
            if i == 'I':
                i = i+"_"+prev
            l.append(label_dic[i])
        entity_data.append(l)

    x_data = np.array(input_data)
    y_data = np.array(entity_data)
    sequence_length = np.array(sequence_length)
    return x_data, y_data, sequence_length

# x_data format : [None, 336, 50]
# y_data format : [None, 336, n_class]
x_data, y_data, sequence_length = getData(sentences)


X = tf.placeholder(
    tf.float32, [None, max_len, vec_size])  # X
Y = tf.placeholder(tf.int32, [None, max_len, n_class])  # Y label
seq_len = tf.placeholder(tf.int32, [None]) # sequeance length
dropoout_rate = tf.placeholder(tf.float32)

# Bidirectional LSTM with Softmax Layer
def BiRNN_with_Softmax(X, seq_len, dropoout_rate):
    lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
    lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
    lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, input_keep_prob=dropoout_rate)
    lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, input_keep_prob=dropoout_rate)

    lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw] * 1)
    lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw] * 1)

    # bidirectional_ rnn
    bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        lstm_cell_fw,
        lstm_cell_bw,
        inputs=X,
        dtype=tf.float32,
        sequence_length=seq_len
    )
    outputs_forward, outputs_backward = bi_outputs
    bi_outputs = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
    outputs = bi_outputs

    # Softmax Layer   -> convert from lstm's output to [batch_size, max_length, n_class]
    X_for_softmax = tf.reshape(outputs, [-1, 2*n_hidden])
    softmax_w = tf.get_variable("softmax_w",[2*n_hidden, n_class], initializer=tf.contrib.layers.xavier_initializer())
    softmax_b = tf.Variable(tf.random_normal([n_class]))
    outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
    outputs = tf.reshape(outputs, [batch_size, max_len, n_class])

    return outputs
# Basic LSTM with Softmax
def RNN_with_Softmax(X):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(
        cell, X, initial_state=initial_state, dtype=tf.float32)

    # Softmax layer
    X_for_softmax = tf.reshape(outputs, [-1, n_hidden])
    softmax_w = tf.get_variable("w",[n_hidden, n_class])
    softmax_b = tf.get_variable("b",[n_class])
    outputs = tf.matmul(X_for_softmax, softmax_w)+ softmax_b
    outputs = tf.reshape(outputs, [batch_size, max_len, n_class])

    return outputs

outputs = BiRNN_with_Softmax(X, seq_len, dropoout_rate)

# Softmax cost
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=Y)

cost = tf.reduce_mean(cost_i)

train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#train = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(outputs, axis=2)

# evaluation method
def make_tag_idx (ans_seq) :
    B_num = 0
    all_answer_start = []
    all_answer_end = []
    all_answer_tag = []

    for sent in ans_seq:
        start_idx = []
        end_idx = []
        tag_set = []
        tag_num = 0
        flag = 0

        for tag in sent:
            if flag == 1 and tag[0] != 'I':
                end_idx.append(tag_num - 1)
                flag = 0

            if tag[0] == 'B':
                B_num = B_num + 1
                start_idx.append(tag_num)
                tag_set.append(tag[2:4])
                flag = 1
            tag_num = tag_num + 1

        if flag == 1:
            end_idx.append(tag_num - 1)

        all_answer_start.append(start_idx)
        all_answer_end.append(end_idx)
        all_answer_tag.append(tag_set)

    return (all_answer_start, all_answer_end, all_answer_tag, B_num)


def eval(ans_seq, pred_seq) :
    correct_num = 0

    (all_answer_start, all_answer_end, all_answer_tag, answer_num) =  make_tag_idx (ans_seq)
    (all_pred_start, all_pred_end, all_pred_tag, pred_num) =  make_tag_idx (pred_seq)

    for i in range(0,len(ans_seq)) :
        for j in range (0,len(all_pred_start[i])) :
            for k in range (0,len(all_answer_start[i])) :
                if all_pred_start[i][j] == all_answer_start[i][k] and all_pred_end[i][j] == all_answer_end[i][k] and all_pred_tag[i][j] == all_answer_tag[i][k] : correct_num = correct_num +1

    return (correct_num, pred_num, answer_num)

# train params
init = tf.global_variables_initializer()
sess = tf.Session()
print ("FUNCTIONS READY")

iteration = 30
save_step = 2
n_train = x_data.shape[0]
show_step = 5

 # train start
print("Train Start")
sess.run(init)
for iter in range(100):
    total_batch = int(n_train / batch_size)
    randpermlist = np.random.permutation(n_train)
    sum_cost = 0

    precision_corr = 0
    precision_total = 0
    recall_corr = 0
    recall_total = 0
    y_pred = []
    y_test = []
    for i in range(total_batch):
        randidx = randpermlist[i * batch_size:min((i + 1) * batch_size, n_train - 1)]
        batch_xs = x_data[randidx, :]
        batch_ys = y_data[randidx, :]
        batch_seq = sequence_length[randidx]

        _, c= sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys, seq_len:batch_seq, dropoout_rate:drop_rate})

        if iter % show_step == 0:
            predict = sess.run(prediction, feed_dict={X: batch_xs, Y: batch_ys, seq_len:batch_seq, dropoout_rate:1})
            for batch in range(batch_size):
                pred = []
                test = []
                for a in range(max_len):
                    label = np.argmax(batch_ys[batch, a])
                    if label == 11:
                        break
                    test.append(index2entity[label])
                    pred.append(index2entity[predict[batch, a]])
                y_test.append(test)
                y_pred.append(pred)


    if iter % show_step == 0:
        print("[{}/{}] cost : {}".format(iter, 100, c))
        correct, pred, ans = eval(y_test, y_pred)
        precision = correct/pred
        recall = correct/ans
        score = (2*precision*recall)/(precision+recall)
        print("[{}/{}] precision : {}".format(iter, 100, precision))
        print("[{}/{}] recall : {}".format(iter, 100, recall))
        print("[{}/{}] score : {}\n".format(iter, 100, score))
        '''print("prediction")
        print(predict[:1])
        print("label")
        print(np.argmax(batch_ys[:1], axis=2))

        print("total num : {} correct_num {}".format(precision_total, precision_corr))
        print("recall : total num:{} correct_num:{}".format(recall_total, recall_corr))
        precision = precision_corr / precision_total
        recall = recall_corr / recall_total
        score = 2 * precision * recall / (precision + recall)
        print("[{}/{}] precision : {}".format(iter, iteration, precision))
        print("[{}/{}] recall : {}".format(iter, iteration, recall))
        print("[{}/{}] score : {}\n".format(iter, iteration, score))'''
