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

from nltk.stem import WordNetLemmatizer
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()
#purely looking for single symbols and words
def checkQuestionMarks(text):
    questionCheck = 0
    if "?" in text:
        questionCheck += 1
    return questionCheck

def checkSemiColon(text):
    semmicolonCheck = 0
    if ";" in text:
        semmicolonCheck += 1
    return semmicolonCheck

def checkColon(text):
    colonCheck = 0
    if ":" in text:
        colonCheck += 1
    return colonCheck

def checkElipsis(text):
    elipsisCheck = 0
    if "..." in text:
        elipsisCheck += 1
    return elipsisCheck

def checkExclaim(text):
    exclaimCheck = 0
    if "!" in text:
        exclaimCheck += 1
    return exclaimCheck

def containsCommonWords(text):
    commonWords = ["who", "when", "how", "why","you","do","what"]
    tokens = text.split()
    commonWordsCount = 0
    for token in tokens:
        for commonWord in commonWords:
            if token.lower() == commonWord.lower():
                commonWordsCount = commonWordsCount + 1
    return commonWordsCount

#looking at the structure and more information
def countWords(text):
    tokens = text.split()
    return len(tokens)

def avgWordLength(text):
    length = 0
    count = 0
    for word in text.split():
        length = len(text)
        count += 1
    wordLength = length / count
    return wordLength

def countUniqueWords(text):
    token = text.split()
    uniqueWords = set(token)
    lenUniqueWords = len(uniqueWords)
    return lenUniqueWords

def findNouns(text):
    doc = nlp(text)
    nouns= 0
    for token in doc:
        if token.tag_ == "NN" or token.tag_ == "NNP":
            nouns += 1
    return nouns

def findVerbs(text):
    doc = nlp(text)
    verbTotal = 0
    for token in doc:
        if token.tag_ == "VBZ" or token.tag_ == "VBG":
            verbTotal += 1
    return verbTotal

def bigramAvgLength(text):
    doc = nlp(text)
    bigramTotal = 0
    bigramAmt = 1
    bigram = []
    for token in doc:
        if len(bigram) == 2:
            tmpTotal = 0
            for tkn in bigram:
                tmpTotal = tmpTotal + len(tkn)
            bigramTotal = bigramTotal + tmpTotal
            bigram = []
            bigramAmt = bigramAmt + 1
        else:
            bigram.append(token.text)
            
    return bigramTotal / bigramAmt

def sentimentAnalysis(text):
    sentiment = 0
    tokens = word_tokenize(text.lower()) 
    for token in tokens:
        if token not in stopwords.words('english'):
            filtered_tokens = [token]
    lemmatizer = WordNetLemmatizer()
    for token in filtered_tokens:
        lemmatized_tokens = [lemmatizer.lemmatize(token)]
    processed_text = ' '.join(lemmatized_tokens)
    scores = analyzer.polarity_scores(processed_text)
    sentiment += scores['compound'] + scores['neu'] + scores['neg'] + scores['pos']
    return sentiment

trainText = []
humour = []
text_features = []
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\dataset.csv"), encoding="utf-8") as file:
    next(file)
    for line in file:
        ln = line.rstrip().split(",")
        trainText.append(ln[0])
        humour.append(ln[1])

f1Features = []
f2Features = []
f3Features = []
f4Features = []
f5Features = []
f6Features = []
f7Features = []
f8Features = []
f9Features = []
f10Features = []
f11Features = []
f12Features = []
f13Features = []

for text in trainText:

    f1 = checkQuestionMarks(text) 
    f2 = checkSemiColon(text)
    f3 = checkColon(text)
    f4 = checkElipsis(text)
    f5 = checkExclaim(text)
    f6 = containsCommonWords(text)
    f7 = findNouns(text)
    f8 = findVerbs(text)
    f9 = countWords(text)
    f10 = avgWordLength(text)
    f11 = countUniqueWords(text)
    f12 = bigramAvgLength(text)
    f13 = sentimentAnalysis(text)

    f1_text_feature = [f1]
    f2_text_feature = [f2]
    f3_text_feature = [f3]
    f4_text_feature = [f4]
    f5_text_feature = [f5]
    f6_text_feature = [f6]
    f7_text_feature = [f7]
    f8_text_feature = [f8]
    f9_text_feature = [f9]
    f10_text_feature = [f10]
    f11_text_feature = [f11]
    f12_text_feeature = [f12]
    f13_text_feature = [f13]

    f1Features.append(f1_text_feature)
    f2Features.append(f2_text_feature)
    f3Features.append(f3_text_feature)
    f4Features.append(f4_text_feature)
    f5Features.append(f5_text_feature)
    f6Features.append(f6_text_feature)
    f7Features.append(f7_text_feature)
    f8Features.append(f8_text_feature)
    f9Features.append(f9_text_feature)
    f10Features.append(f10_text_feature)
    f11Features.append(f11_text_feature)
    f12Features.append(f12_text_feeature)
    f13Features.append(f13_text_feature)

with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f1.dill"), "wb") as f:
    dill.dump(f1Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f2.dill"), "wb") as f:
    dill.dump(f2Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f3.dill"), "wb") as f:
    dill.dump(f3Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f4.dill"), "wb") as f:
    dill.dump(f4Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f5.dill"), "wb") as f:
    dill.dump(f5Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f6.dill"), "wb") as f:
    dill.dump(f6Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f7.dill"), "wb") as f:
    dill.dump(f7Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f8.dill"), "wb") as f:
    dill.dump(f8Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f9.dill"), "wb") as f:
    dill.dump(f9Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f10.dill"), "wb") as f:
    dill.dump(f10Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f11.dill"), "wb") as f:
    dill.dump(f11Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f12.dill"), "wb") as f:
    dill.dump(f12Features, f)
with open(os.path.join(os.path.dirname(sys.argv[0]) + "\\f13.dill"), "wb") as f:
    dill.dump(f13Features, f)



