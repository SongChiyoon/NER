import dataparse

'''text2Object = []
words = []
fe = []
sentencs = []
features = []
appending = False
text = None
datapath = "data/train.txt"
with open(datapath, 'r') as f:
    for line in f:

        if ";" in line and "." in line:
            if not appending:
                print("start")
            appending = True
            if appending and text is not None:
                text2Object.append(text)
                sentencs.append(words)
                features.append(fe)
                fe = []
                words = []
            text = dataparse.textObject()
            text.setLine(line[2:])
            continue
        if "$" in line and "." in line:
            text.setResult(line[1:])
            continue
        if appending:
            line = line.replace("\n", "")
            if "" == line:
                continue
            splits = line.split("\t")
            #print(len(splits))
            if len(splits) == 4:
                text.addSplits(splits[0], splits[1], splits[2], splits[3])
                words.append(splits[1])
                fe.append(splits[2])
            continue'''

import gensim
import dataparse as ps
datapath = "data/train.txt"
parser = ps.Parser(datapath)
parser.parse()
sentences = parser.sentencs
features = parser.features

model_name = "model/w2v_model"
model = gensim.models.Word2Vec(sentences)
model.wv.save_word2vec_format(model_name, binary=False)

fModel_name = "model/f2v_model"
model = gensim.models.Word2Vec(features)
model.wv.save_word2vec_format(fModel_name, binary=False)



