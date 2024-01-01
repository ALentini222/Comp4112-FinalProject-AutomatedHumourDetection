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

trainText = []
humour = []
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\dataset.csv"), encoding="utf-8") as file:
    next(file)
    for line in file:
        ln = line.rstrip().split(",")
        trainText.append(ln[0])
        humour.append(ln[1])

f1Features = dill.load(open(os.path.expanduser("f1.dill"),"rb"))
f2Features = dill.load(open(os.path.expanduser("f2.dill"),"rb"))
f3Features = dill.load(open(os.path.expanduser("f3.dill"),"rb"))
f4Features = dill.load(open(os.path.expanduser("f4.dill"),"rb"))
f5Features = dill.load(open(os.path.expanduser("f5.dill"),"rb"))
f6Features =dill.load(open(os.path.expanduser("f6.dill"),"rb"))
f7Features =  dill.load(open(os.path.expanduser("f7.dill"),"rb"))
f8Features = dill.load(open(os.path.expanduser("f8.dill"),"rb"))
f9Features = dill.load(open(os.path.expanduser("f9.dill"),"rb")) 
f10Features = dill.load(open(os.path.expanduser("f10.dill"),"rb"))
f11Features = dill.load(open(os.path.expanduser("f11.dill"),"rb"))
f12Features = dill.load(open(os.path.expanduser("f12.dill"),"rb"))
f13Features =dill.load(open(os.path.expanduser("f13.dill"),"rb"))

f1_X_train,f1_X_test,f1_y_train,f1_y_test=train_test_split(f1Features,humour,test_size=0.3)
f2_X_train,f2_X_test,f2_y_train,f2_y_test=train_test_split(f2Features,humour,test_size=0.3)
f3_X_train,f3_X_test,f3_y_train,f3_y_test=train_test_split(f3Features,humour,test_size=0.3)
f4_X_train,f4_X_test,f4_y_train,f4_y_test=train_test_split(f4Features,humour,test_size=0.3)
f5_X_train,f5_X_test,f5_y_train,f5_y_test=train_test_split(f5Features,humour,test_size=0.3)
f6_X_train,f6_X_test,f6_y_train,f6_y_test=train_test_split(f6Features,humour,test_size=0.3)
f7_X_train,f7_X_test,f7_y_train,f7_y_test=train_test_split(f7Features,humour,test_size=0.3)
f8_X_train,f8_X_test,f8_y_train,f8_y_test=train_test_split(f8Features,humour,test_size=0.3)
f9_X_train,f9_X_test,f9_y_train,f9_y_test=train_test_split(f9Features,humour,test_size=0.3)
f10_X_train,f10_X_test,f10_y_train,f10_y_test=train_test_split(f10Features,humour,test_size=0.3)
f11_X_train,f11_X_test,f11_y_train,f11_y_test=train_test_split(f11Features,humour,test_size=0.3)
f12_X_train,f12_X_test,f12_y_train,f12_y_test=train_test_split(f12Features,humour,test_size=0.3)
f13_X_train,f13_X_test,f13_y_train,f13_y_test=train_test_split(f13Features,humour,test_size=0.3)

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f1_X_train, f1_y_train)
knnTestPred = classifierKNN.predict(f1_X_test)
npYtest = numpy.array(f1_y_test)
print("K-Nearest Neighbour " + "F1 Test set score(?): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f2_X_train, f2_y_train)
knnTestPred = classifierKNN.predict(f2_X_test)
npYtest = numpy.array(f2_y_test)
print("K-Nearest Neighbour " + "F2 Test set score(;): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f3_X_train, f3_y_train)
knnTestPred = classifierKNN.predict(f3_X_test)
npYtest = numpy.array(f3_y_test)
print("K-Nearest Neighbour " + "F3 Test set score(:): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f4_X_train, f4_y_train)
knnTestPred = classifierKNN.predict(f4_X_test)
npYtest = numpy.array(f4_y_test)
print("K-Nearest Neighbour " + "F4 Test set score(...): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f5_X_train, f5_y_train)
knnTestPred = classifierKNN.predict(f5_X_test)
npYtest = numpy.array(f5_y_test)
print("K-Nearest Neighbour " + "F5 Test set score(!): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f6_X_train, f6_y_train)
knnTestPred = classifierKNN.predict(f6_X_test)
npYtest = numpy.array(f6_y_test)
print("K-Nearest Neighbour " + "F6 Test set score(common words): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f7_X_train, f7_y_train)
knnTestPred = classifierKNN.predict(f7_X_test)
npYtest = numpy.array(f7_y_test)
print("K-Nearest Neighbour " + "F7 Test set score(noun): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f8_X_train, f8_y_train)
knnTestPred = classifierKNN.predict(f8_X_test)
npYtest = numpy.array(f8_y_test)
print("K-Nearest Neighbour " + "F8 Test set score(verbs): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f9_X_train, f9_y_train)
knnTestPred = classifierKNN.predict(f9_X_test)
npYtest = numpy.array(f9_y_test)
print("K-Nearest Neighbour " + "F9 Test set score(word count): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f10_X_train, f10_y_train)
knnTestPred = classifierKNN.predict(f10_X_test)
npYtest = numpy.array(f10_y_test)
print("K-Nearest Neighbour " + "F10 Test set score(avg word length): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f11_X_train, f11_y_train)
knnTestPred = classifierKNN.predict(f11_X_test)
npYtest = numpy.array(f11_y_test)
print("K-Nearest Neighbour " + "F11 Test set score(unique words): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f12_X_train, f12_y_train)
knnTestPred = classifierKNN.predict(f12_X_test)
npYtest = numpy.array(f12_y_test)
print("K-Nearest Neighbour " + "F12 Test set score(bigrams): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))

classifierKNN = KNeighborsClassifier(n_neighbors=7)
classifierKNN.fit(f13_X_train, f13_y_train)
knnTestPred = classifierKNN.predict(f13_X_test)
npYtest = numpy.array(f13_y_test)
print("K-Nearest Neighbour " + "F13 Test set score(sentiment analysis): {:.2f}".format(numpy.mean(knnTestPred == npYtest)))