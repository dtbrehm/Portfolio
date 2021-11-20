# -*- coding: utf-8 -*-
"""
IMDB Sentiment Analysis.

@author: David Brehm
"""

import pandas as pd
import numpy as np
import os
import re
import statistics
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#%% Datapath.
datapath = r'D:\School\680\Project 3\Data\aclImdb\\'

#%% Get training data
train_texts = []
train_labels = []
for reviewType in ['pos', 'neg']:
    train_path = os.path.join(datapath, 'train', reviewType)
    for f in sorted(os.listdir(train_path)):
        if f.endswith('.txt'):
            with open(os.path.join(train_path, f), encoding='utf8') as fn:
                train_texts.append(fn.read())
            train_labels.append(0 if reviewType == 'neg' else 1)

#%% Get test data.
test_texts = []
test_labels = []
for reviewType in ['pos', 'neg']:
    test_path = os.path.join(datapath, 'test', reviewType)
    for f in sorted(os.listdir(test_path)):
        if f.endswith('.txt'):
            with open(os.path.join(test_path, f), encoding='utf8') as fn:
                test_texts.append(fn.read())
            test_labels.append(0 if reviewType == 'neg' else 1)

#%% Clean data. Remove punctuation, set to lowercase, and remove breaks.
remove_punc = re.compile("[.;:!\'?,\"()\[\]]")
remove_breaks = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

train_texts = [remove_punc.sub("", t.lower()) for t in train_texts]
train_texts = [remove_breaks.sub(" ", t) for t in train_texts]

test_texts = [remove_punc.sub("", t.lower()) for t in test_texts]
test_texts = [remove_breaks.sub(" ", t) for t in test_texts]

#%% Basic review insights. 
#   Training number of words.

wordCountTrain = [len(t.split()) for t in train_texts]
print(statistics.mean(wordCountTrain))
print(statistics.median(wordCountTrain))
bins = np.arange(0, 1100, 50)

plt.hist(wordCountTrain, bins=bins)
plt.title('Distribution of Training Reviews Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Number of Reviews')

#%% Test number of words.
wordCountTest = [len(t.split()) for t in test_texts]
bins = np.arange(0, 1100, 50)

plt.hist(wordCountTest, bins=bins)
plt.title('Distribution of Test Reviews Word Counts')
plt.xlabel('Word Count')


#%% Vectorize reviews for machine learning input.
cv = CountVectorizer(binary=True, stop_words='english')
cv.fit(train_texts)
X = cv.transform(train_texts)
X_test = cv.transform(test_texts)

#%% Split training data into validation.
X_train, X_val, y_train, y_val = train_test_split(X, train_labels, train_size = 0.7)

#%% Find the best inverse lambda for the Logistic Regression model.
for c in [0.001, 0.01, 0.025, 0.05, 0.1, 0.2]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print("Accuracy for C = {}: {}".format(str(c), str(round(accuracy_score(y_val, lr.predict(X_val)),3))))
    
#%% Train model with most optimal C while using the entire training set. 
lr_0pt1 = LogisticRegression(C=0.1)
lr_0pt1.fit(X, train_labels)
print("Accuracy for C = 0.1: {}".format(str(round(accuracy_score(test_labels, lr_0pt1.predict(X_test)),3))))

#%% Look at the word coefficients. Plot 10 most positive and negative words. 
coefs =  pd.DataFrame(sorted(list(zip(cv.get_feature_names(), lr_0pt1.coef_[0])), key=lambda x:x[1]), columns=['Word', 'Coef'])

headtail = pd.concat([coefs.head(10), coefs.tail(10)])

plt.bar(headtail['Word'].head(10), headtail['Coef'].head(10), color='r')
plt.bar(headtail['Word'].tail(10), headtail['Coef'].tail(10), color='g')
plt.xticks(rotation=90)
plt.ylabel('Coefficient')
plt.title('Most Impactful Positive and Negative Words')    



#%%

#%% Model with bi-grams.
cv = CountVectorizer(ngram_range=(2,2), binary=True)
cv.fit(train_texts)
Xbi = cv.transform(train_texts)
Xbi_test = cv.transform(test_texts)

#%% Split training data into validation.
Xbi_train, Xbi_val, ybi_train, ybi_val = train_test_split(Xbi, train_labels, train_size = 0.7)

#%% Find the best inverse lambda for the Logistic Regression model.
for c in [0.001, 0.01, 0.025, 0.05, 0.1, 0.2]:
    lr = LogisticRegression(C=c)
    lr.fit(Xbi_train, ybi_train)
    print("Accuracy for C = {}: {}".format(str(c), str(round(accuracy_score(ybi_val, lr.predict(Xbi_val)),3))))
    
#%% Train model with most optimal C while using the entire training set. 
lrbi_0pt1 = LogisticRegression(C=0.1)
lrbi_0pt1.fit(Xbi, train_labels)
print("Accuracy for C = 0.1: {}".format(str(round(accuracy_score(test_labels, lrbi_0pt1.predict(Xbi_test)),3))))

#%% Look at the word coefficients. Plot 10 most positive and negative words. 
coefsbi =  pd.DataFrame(sorted(list(zip(cv.get_feature_names(), lrbi_0pt1.coef_[0])), key=lambda x:x[1]), columns=['Word', 'Coef'])

headtailbi = pd.concat([coefsbi.head(10), coefsbi.tail(10)])

plt.bar(headtailbi['Word'].head(10), headtailbi['Coef'].head(10), color='r')
plt.bar(headtailbi['Word'].tail(10), headtailbi['Coef'].tail(10), color='g')
plt.xticks(rotation=90)
plt.ylabel('Coefficient')
plt.title('Most Impactful Positive and Negative Bigrams')

#%% Model with bi-grams and without stopwords.
cv = CountVectorizer(ngram_range=(2,2), binary=True, stop_words='english')
cv.fit(train_texts)
Xbi = cv.transform(train_texts)
Xbi_test = cv.transform(test_texts)

#%% Split training data into validation.
Xbi_train, Xbi_val, ybi_train, ybi_val = train_test_split(Xbi, train_labels, train_size = 0.7)

#%% Find the best inverse lambda for the Logistic Regression model.
for c in [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 10, 100]:
    lr = LogisticRegression(C=c)
    lr.fit(Xbi_train, ybi_train)
    print("Accuracy for C = {}: {}".format(str(c), str(round(accuracy_score(ybi_val, lr.predict(Xbi_val)),3))))
    
#%% Train model with most optimal C while using the entire training set. 
lrbi_1 = LogisticRegression(C=1)
lrbi_1.fit(Xbi, train_labels)
print("Accuracy for C = 1: {}".format(str(round(accuracy_score(test_labels, lrbi_1.predict(Xbi_test)),3))))

#%% Look at the word coefficients. Plot 10 most positive and negative words. 
coefsbi =  pd.DataFrame(sorted(list(zip(cv.get_feature_names(), lrbi_1.coef_[0])), key=lambda x:x[1]), columns=['Word', 'Coef'])

headtailbi = pd.concat([coefsbi.head(10), coefsbi.tail(10)])

plt.bar(headtailbi['Word'].head(10), headtailbi['Coef'].head(10), color='r')
plt.bar(headtailbi['Word'].tail(10), headtailbi['Coef'].tail(10), color='g')
plt.xticks(rotation=90)
plt.ylabel('Coefficient')
plt.title('Most Impactful Positive and Negative Bigrams without Stopwords')    


#%%

#%% Model with tri-grams.
cv = CountVectorizer(ngram_range=(3,3), binary=True)
cv.fit(train_texts)
Xtri = cv.transform(train_texts)
Xtri_test = cv.transform(test_texts)

#%% Split training data into validation.
Xtri_train, Xtri_val, ytri_train, ytri_val = train_test_split(Xtri, train_labels, train_size = 0.7)

#%% Find the best inverse lambda for the Logistic Regression model.
for c in [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 10, 100]:
    lr = LogisticRegression(C=c)
    lr.fit(Xtri_train, ytri_train)
    print("Accuracy for C = {}: {}".format(str(c), str(round(accuracy_score(ytri_val, lr.predict(Xtri_val)),3))))
    
#%% Train model with most optimal C while using the entire training set. 
lrtri_10 = LogisticRegression(C=10)
lrtri_10.fit(Xtri, train_labels)
print("Accuracy for C = 10: {}".format(str(round(accuracy_score(test_labels, lrtri_10.predict(Xtri_test)),3))))

#%% Look at the word coefficients. Plot 10 most positive and negative words. 
coefstri =  pd.DataFrame(sorted(list(zip(cv.get_feature_names(), lrtri_10.coef_[0])), key=lambda x:x[1]), columns=['Word', 'Coef'])

headtailtri = pd.concat([coefstri.head(10), coefstri.tail(10)])

plt.bar(headtailtri['Word'].head(10), headtailtri['Coef'].head(10), color='r')
plt.bar(headtailtri['Word'].tail(10), headtailtri['Coef'].tail(10), color='g')
plt.xticks(rotation=90)
plt.ylabel('Coefficient')
plt.title('Most Impactful Positive and Negative Trigrams')    


#%%

#%% Model with uni-grams through tri-grams.
cv = CountVectorizer(ngram_range=(1,3), binary=True)
cv.fit(train_texts)
Xtri = cv.transform(train_texts)
Xtri_test = cv.transform(test_texts)

#%% Split training data into validation.
Xtri_train, Xtri_val, ytri_train, ytri_val = train_test_split(Xtri, train_labels, train_size = 0.7)

#%% Find the best inverse lambda for the Logistic Regression model.
for c in [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 10, 100]:
    lr = LogisticRegression(C=c)
    lr.fit(Xtri_train, ytri_train)
    print("Accuracy for C = {}: {}".format(str(c), str(round(accuracy_score(ytri_val, lr.predict(Xtri_val)),3))))
    
#%% Train model with most optimal C while using the entire training set. 
lrtri_10 = LogisticRegression(C=10)
lrtri_10.fit(Xtri, train_labels)
print("Accuracy for C = 10: {}".format(str(round(accuracy_score(test_labels, lrtri_10.predict(Xtri_test)),3))))

#%% Look at the word coefficients. Plot 10 most positive and negative words. 
coefstri =  pd.DataFrame(sorted(list(zip(cv.get_feature_names(), lrtri_10.coef_[0])), key=lambda x:x[1]), columns=['Word', 'Coef'])

headtailtri = pd.concat([coefstri.head(10), coefstri.tail(10)])

plt.bar(headtailtri['Word'].head(10), headtailtri['Coef'].head(10), color='r')
plt.bar(headtailtri['Word'].tail(10), headtailtri['Coef'].tail(10), color='g')
plt.xticks(rotation=90)
plt.ylabel('Coefficient')
plt.title('Most Impactful Positive and Negative Unigrams through Trigrams')   


#%% Model with uni-grams through tri-grams, TfidVectorizer.
tv = TfidfVectorizer(ngram_range=(1,3), binary=True)
tv.fit(train_texts)
Xtri = tv.transform(train_texts)
Xtri_test = tv.transform(test_texts)

#%% Split training data into validation.
Xtri_train, Xtri_val, ytri_train, ytri_val = train_test_split(Xtri, train_labels, train_size = 0.7)

#%% Find the best inverse lambda for the Logistic Regression model.
for c in [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 10, 100]:
    lr = LogisticRegression(C=c)
    lr.fit(Xtri_train, ytri_train)
    print("Accuracy for C = {}: {}".format(str(c), str(round(accuracy_score(ytri_val, lr.predict(Xtri_val)),3))))
    
#%% Train model with most optimal C while using the entire training set. 
lrtri_100 = LogisticRegression(C=100)
lrtri_100.fit(Xtri, train_labels)
print("Accuracy for C = 100: {}".format(str(round(accuracy_score(test_labels, lrtri_100.predict(Xtri_test)),3))))

#%% Look at the word coefficients. Plot 10 most positive and negative words. 
coefstri =  pd.DataFrame(sorted(list(zip(cv.get_feature_names(), lrtri_100.coef_[0])), key=lambda x:x[1]), columns=['Word', 'Coef'])

headtailtri = pd.concat([coefstri.head(10), coefstri.tail(10)])

plt.bar(headtailtri['Word'].head(10), headtailtri['Coef'].head(10), color='r')
plt.bar(headtailtri['Word'].tail(10), headtailtri['Coef'].tail(10), color='g')
plt.xticks(rotation=90)
plt.ylabel('Coefficient')
plt.title('Most Impactful Positive and Negative Unigrams through Trigrams')  
 
