from gensim.models import word2vec
data = word2vec.Text8Corpus("wiki.wakati")
model = word2vec.Word2Vec(data, size=64)
model.save("model/wiki_model")
print("ok")