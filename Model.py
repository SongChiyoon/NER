import tensorflow as tf
import gensim
import numpy as np
import pandas as pd
import dataparse

from gensim.models.keyedvectors import KeyedVectors
wordModel = KeyedVectors.load_word2vec_format('model/w2v_model', binary=False)
featureModel = KeyedVectors.load_word2vec_format('model/featureEmbedding', binary=False)

import dataparse as ps




