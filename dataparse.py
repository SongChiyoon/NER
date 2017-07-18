class textObject(object):
    def __init__(self):
        self.positions = []
        self.words = []
        self.features = []
        self.labels = []

    def setLine(self, line):
        self.line = line

    def getLine(self):
        return self.line


    def setResult(self, l):
        self.result = l

    def getResult(self):
        return self.result

    def addSplits(self, position, word, feature, label):
        self.positions.append(position)
        self.words.append(word)
        self.features.append(feature)
        self.labels.append(label)
        #print(self.words)

    def getWord(self):
        return self.words

text2Object = []
words = []
sentencs = []
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
                words = []
            text = textObject()
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
            continue
#print(sentencs)
import gensim
model_name = "model/w2v_model"
model = gensim.models.Word2Vec(sentencs)
#model.save(model_name)
model.wv.save_word2vec_format("model.txt", binary=False)




#with open('word2vec.txt', 'w') as f:

