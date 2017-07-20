import tensorflow as tf
import gensim
import numpy as np
import pandas as pd
import dataparse

from gensim.models.keyedvectors import KeyedVectors
wordModel = KeyedVectors.load_word2vec_format('model/w2v_model', binary=False)
featureModel = KeyedVectors.load_word2vec_format('model/f2v_model', binary=False)

import dataparse as ps

data_path = 'data/train.txt'
parser = ps.Parser(data_path)
parser.parse()
sentences = parser.sentencs


labels = parser.labels

x_data = np.array(sentences)
y_data = np.array(labels)

print(y_data.shape)
print(x_data.shape)

X = tf.placeholder(tf.int32, [None, None])  # X data
Y = tf.placeholder(tf.int32, [None, None])  # Y label


#y_one_hot = tf.one_hot(Y, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0