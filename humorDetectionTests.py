from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import spacy
import numpy
import io,os,sys
import dill
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier


trainText = []
humour = []
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\dataset.csv"), encoding="utf-8") as file:
    next(file)
    for line in file:
        ln = line.rstrip().split(",")
        trainText.append(ln[0])
        humour.append(ln[1])

text_features = dill.load(open(os.path.expanduser("humour.dill"),"rb"))

X_train,X_test,y_train,y_test=train_test_split(text_features,humour,test_size=0.3)


classifierKNN = KNeighborsClassifier(n_neighbors=3)
classifierKNN.fit(X_train, y_train)
knnTestPred = classifierKNN.predict(X_test)
npYtest = numpy.array(y_test)
print("K-Nearest Neighbour 3" + " Test set score: {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=5)
classifierKNN.fit(X_train, y_train)
knnTestPred = classifierKNN.predict(X_test)
npYtest = numpy.array(y_test)
print("K-Nearest Neighbour 5" + " Test set score: {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=2)
classifierKNN.fit(X_train, y_train)
knnTestPred = classifierKNN.predict(X_test)
npYtest = numpy.array(y_test)
print("K-Nearest Neighbour 2" + " Test set score: {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(X_train, y_train)
knnTestPred = classifierKNN.predict(X_test)
knnTestPred = numpy.array(knnTestPred)
npYtest = numpy.array(y_test)
print("K-Nearest Neighbour 7" + " Test set score: {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=9)
classifierKNN.fit(X_train, y_train)
knnTestPred = classifierKNN.predict(X_test)
npYtest = numpy.array(y_test)
print("K-Nearest Neighbour 9" + " Test set score: {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierDTree = DecisionTreeClassifier()
classifierDTree.fit(X_train, y_train)
classifierDTreeTestPred = classifierDTree.predict(X_test)
npYtest = numpy.array(y_test)
print("Decision Tree" + " Test set score: {:.2f}".format(numpy.mean(classifierDTreeTestPred == npYtest)))

classifierMultinomialNB = MultinomialNB()
classifierMultinomialNB.fit(X_train,y_train)
classifierMultinomialNBTestPred = classifierMultinomialNB.predict(X_test)
npYtest = numpy.array(y_test)
print("MultinomialNB " + " Test set score: {:.2f}".format(numpy.mean(classifierMultinomialNBTestPred == npYtest)))

classifierBernoulliNB = BernoulliNB()
classifierBernoulliNB.fit(X_train,y_train)
classifierBernoulliNBTestPred = classifierBernoulliNB.predict(X_test)
npYtest = numpy.array(y_test)
print("BernoulliNB " + " Test set score: {:.2f}".format(numpy.mean(classifierBernoulliNBTestPred == npYtest)))

classifierSGD = SGDClassifier(max_iter=(10))
classifierSGD.fit(X_train, y_train)
classifierSGDTestPred = classifierSGD.predict(X_test)
npYtest = numpy.array(y_test)
print("SGD " + " Test set score: {:.2f}".format(numpy.mean(classifierSGDTestPred == npYtest)))

classifierSGD = SGDClassifier()
classifierSGD.fit(X_train, y_train)
classifierSGDTestPred = classifierSGD.predict(X_test)
npYtest = numpy.array(y_test)
print("SGD " + " Test set score: {:.2f}".format(numpy.mean(classifierSGDTestPred == npYtest)))

classifierPassiveAggressive = PassiveAggressiveClassifier()
classifierPassiveAggressive.fit(X_train, y_train)
classifierPassiveAggressiveTestPred = classifierPassiveAggressive.predict(X_test)
npYtest = numpy.array(y_test)
print("PassiveAggressive " + " Test set score: {:.2f}".format(numpy.mean(classifierPassiveAggressiveTestPred == npYtest)))
