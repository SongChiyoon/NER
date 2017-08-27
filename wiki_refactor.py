# encoding: utf-8
import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitter
from gensim.models import Word2Vec

#파일열기
readFile = codecs.open("data/wiki.txt", "r", encoding="utf-8")
wk_file = "wiki.wakati"
writeFile = open(wk_file, "w", encoding="utf-8")

twitter = Twitter()
i = 0

while True:
    line = readFile.readline()
    if not line : break
    if i % 20000 == 0:
        print("current - " + str(i))
    i+=1

    malist = twitter.pos(line, norm=True, stem=True)

    r = []
    for word in malist:
        if not word[1] in ["Josa", "Eomi", "Punctuation"]:
            writeFile.write(word[0] + " ")
writeFile.close()