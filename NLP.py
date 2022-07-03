from nltk import NaiveBayesClassifier as nbc
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
import pickle

# เขียนโดย นาย วรรณพงษ์  ภัททิยไพบูลย์
# ใช้ประกอบบทความใน python3.wannaphong.com
# cc-by 3.0 Thai Sentiment Text https://github.com/wannaphongcom/lexicon-thai/tree/master/ข้อความ/
# อ่านบทความได้ที่ https://python3.wannaphong.com/2017/02/ทำ-sentiment-analysis-ภาษาไทยใน-python.html
from nltk import NaiveBayesClassifier as nbc
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
# pos.txt
with codecs.open('pos.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listpos=[e.strip() for e in lines]
del lines
f.close() # ปิดไฟล์

# neg.txt
with codecs.open('neg.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listneg=[e.strip() for e in lines]
f.close() # ปิดไฟล์

# neutral.txt
with codecs.open('neutral.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listneutral=[e.strip() for e in lines]
del lines
f.close() # ปิดไฟล์

pos1=['pos']*len(listpos)
neg1=['neg']*len(listneg)
neutral1=['neutral']*len(listneutral)

training_data = list(zip(listpos,pos1)) + list(zip(listneg,neg1)) + list(zip(listneutral,neutral1)) 
vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))
feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in training_data]
classifier = nbc.train(feature_set)

# # save file
f = open('test_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

# load file
f = open('test_classifier.pickle', 'rb')
classifier = pickle.load(f)
#f.close()

from io import open
import os

for filename in os.listdir('ข้อความ/'):
    f = open(f'message/{filename}', mode="r", encoding="utf-8")
    test_sentence = f.read()
    f.close()
    
    featurized_test_sentence = {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary}
    print("test_sent:",test_sentence)
    print("tag:",classifier.classify(featurized_test_sentence)) # ใช้โมเดลที่ train ประมวลผล
