import os
from collections import Counter
import enchant
from nltk.util import ngrams
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm

dataDir = '../data/'
data = []

#labels and input words
labels = []
words = []
telWords = {}

#English Dictionary
eng = enchant.Dict('en_US')

def checkDict(word):
    # if Telugu
    if word in telWords:
        return 2
    # if English 
    if eng.check(word.lower()):
        return 1
    # Not found in Dictionaries
    return 0

def generateNgrams(word):
    ngram = []
    for i in ngrams('   ' + word + '   ',3):
        ngram.append(''.join(i).strip())
    return ngram[0], ngram[1], ngram[2], ngram[-1], ngram[-2], ngram[-3]
def generateFeatures(words):
    features = []
    for i in words:
        temp = {}
        temp['suf1'], temp['suf2'], temp['suf3'], temp['pre1'], temp['pre2'], temp['pre3'] = generateNgrams(i)
        temp['len'] = 1 if len(i) > 5 else 0
        temp['dict'] = checkDict(i)
        temp['capital'] = 0 if i[0] == i[0].lower() else 1
        features.append(temp)
    return features

for i in os.listdir(dataDir):
    data.append(open(dataDir + i).read())


for i in data:
    for j in i.split('\n\n'):
        for k in j.split('\n'):
            temp = k.split()
            if len(temp)>1:
                labels.append(temp[1])
                if temp[1] == 'te':
                    telWords[temp[1]] = temp[1]
                words.append(temp[0])

features = generateFeatures(words)

X_train = generateFeatures(words[:9000])
X_test = generateFeatures(words[9000:])
Y_train = labels[:9000]
Y_test = labels[9000:]

clf = Pipeline([
	('vectorizer', DictVectorizer(sparse=False)),
	('classifier',linear_model.LogisticRegression())
	])
clf.fit(X_train,Y_train)

'''
for i in X_test:
    print(i['word'],clf.predict(i))

'''
print('Accuracy:',clf.score(X_test,Y_test))





