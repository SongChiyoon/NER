import gensim
import numpy as np
from gensim import corpora
model_name = "model/w2v_model"
model = gensim.models.Word2Vec.load(model_name)
print(model)
#print(model.wv['챔피언스'])

data = np.array([])

model.save_as_text('data/w2v.txt', True)
#np.savetxt('data/w2v.txt', data)

#print(data)