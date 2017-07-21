import tensorflow as tf
import gensim
import numpy as np
import pandas as pd
import dataparse

from gensim.models.keyedvectors import KeyedVectors
wordModel = KeyedVectors.load('model/w2v_model')
featureModel = KeyedVectors.load('model/f2v_model')

import dataparse as ps

data_path = 'data/train.txt'
parser = ps.Parser(data_path)
parser.parse()
sentences = parser.sentences

category = parser.catagory
labels = parser.labels

input_data = []
for line in sentences:
    w2c = []
    for i in line:
        w2c.append(wordModel[i])
    input_data.append(w2c)

x_data = np.array(input_data)
y_data = np.array(labels)

n_class = len(category)  #number of labels
category = np.array(category)
# ['O' 'B_OG' 'I' 'B_DT' 'B_PS' 'B_LC' 'B_TI']


X = tf.placeholder(tf.float32, [None, None, 200])  # X data
Y = tf.placeholder(tf.float32, [None, n_class])  # Y label

onehot_labels = tf.one_hot(n_class-1, 5)



