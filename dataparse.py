class textObject(object):
    def __init__(self, path):
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

class Parser(object):
    def __init__(self, path):
        self.path = path
        self.text2Object = []
        self.sentences = []
        self.features = []
        self.labels = []
        self.catagory = []

    def sentences(self):
        return self.sentences

    def features(self):
        return self.features

    def labels(self):
        return self.labels

    def category(self):
        return self.catagory

    def parse(self):
        text = None
        appending = False
        words = []
        fe = []
        label = []
        a = 0
        with open(self.path, 'r') as f:
            for line in f:
                if ";" in line and "." in line:
                    if not appending:
                        print("start")
                        appending = True

                    if appending and text is not None:
                        self.text2Object.append(text)
                        self.sentences.append(words)
                        self.features.append(fe)
                        self.labels.append(label)
                        label = []
                        fe = []
                        words = []
                        a += 1
                    text = textObject(object)
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
                    # print(len(splits))
                    if len(splits) == 4:
                        text.addSplits(splits[0], splits[1], splits[2], splits[3])
                        words.append(splits[1])
                        fe.append(splits[2])
                        label.append(splits[3])
                        if splits[3] not in self.catagory:
                            self.catagory.append(splits[3])
                    continue
            print('len :{0}'.format(a))