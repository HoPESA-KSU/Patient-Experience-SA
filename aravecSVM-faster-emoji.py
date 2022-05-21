#!/usr/bin/env python
# coding: utf-8

# In[84]:


import gensim
import re
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import string
import nltk
from nltk.corpus import stopwords
from sklearn import metrics
import statistics
import matplotlib.pyplot as plt


# In[63]:


#load the model
model = gensim.models.Word2Vec.load('D:/download2/full_grams_sg_300_twitter/full_grams_sg_300_twitter.mdl')


# In[64]:


#set number of features(length of vector)
num_features = 300


# In[65]:


#load the dataset
df = pd.read_csv("finalDataset.csv")
#drop NaN rows 
df = df.dropna()

df.shape


# In[66]:


df.head()


# In[67]:


#split the dataset using Stratified split
X = df['tweet']
y = df['finalLabel']

split1 = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
for train_index, test_index in split1.split(X,y):
    X_train , X_test = X.iloc[train_index], X.iloc[test_index]
    y_train , y_test = y.iloc[train_index], y.iloc[test_index]
    
    
# split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
# for test_index, val_index in split2.split(X_validation_and_test, y_validation_and_test):
#     X_test, X_validation = X_validation_and_test.iloc[test_index], X_validation_and_test.iloc[val_index]
#     y_test, y_validation = y_validation_and_test.iloc[test_index], y_validation_and_test.iloc[val_index]


# # Preprocessing

# In[68]:


def replace_emoji(tweet):
    result = ' '
    for char in tweet:
        result += emoji_dict.get(char, char)
    return result


# In[69]:


emj_df = pd.read_csv("emoji.csv")
emj = emj_df['Emoji']
sent = emj_df['sentiment']

emoji_dict = {emj[i]: sent[i] for i in range(len(emj))}


# In[70]:


#function to clean and normalize each tweet
def preprocess(tweet): 
    
    #Replace emoji with their semantics
    tweet =  replace_emoji(tweet)
    
    #Remove http links and mention tags
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    
    #Remove punctuations
    eng_punct = string.punctuation
    arabic_punct = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    punct_list = eng_punct + arabic_punct
    tweet = re.sub('[%s]' % re.escape(punct_list), ' ', tweet)
    
    #Remove digits
    tweet = re.sub(r'[0-9]+', '', tweet)    
    
    #Remove English words
    tweet = re.sub(r'\s*[A-Za-z]+\b', '' , tweet)
    tweet = tweet.rstrip()
    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    tweet = re.sub(p_tashkeel,"", tweet)
    
    
    #Remove elongation (repeated letters)
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
 
    tweet = tweet.replace('وو', 'و')
    tweet = tweet.replace('يي', 'ي')
    tweet = tweet.replace('اا', 'ا')
    
    #Normalize letters
    tweet = re.sub("[إأآا]", "ا", tweet)
    tweet = re.sub("ى", "ي", tweet)
    tweet = re.sub("ة", "ه", tweet) 
    #tweet = re.sub("گ", "ك", tweet)
    
    #Tokenize tweets and remove stop words
    stopword_list = stopwords.words('arabic')
    tokens = nltk.word_tokenize(tweet)
    clean_tokens = [w for w in tokens if w not in stopword_list]
    tweet = clean_tokens
    
    return tweet


# In[71]:


#preprocess
#X_train = [preprocess(tweet) for tweet in X_train]
#X_test = [preprocess(tweet) for tweet in X_test]


# In[72]:


X = [preprocess(tweet) for tweet in X]


# # Word Embedding

# In[73]:


not_found = 0


# In[74]:


def word_vector(tokens, size, not_found):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model.wv[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            not_found = not_found +1
    if count != 0:
        vec /= count
    return vec, not_found


# In[75]:


trainvecs = np.zeros((len(X), 300)) 
for i in range(len(X)):
    if i%1000 == 0:
        print("Review %d of %d"%(i,len(X)))
    trainvecs[i,:], not_found = word_vector(X[i], 300, not_found)
train_df = pd.DataFrame(trainvecs)
train_df.shape


# In[76]:


not_found 


# In[77]:


train_df.head()


# In[54]:


#Create a svm Classifier
from sklearn.model_selection import cross_validate
clf = svm.SVC(kernel='rbf') # Gaussian Kernel


# In[55]:


scoresGauss = cross_validate(clf, trainvecs, y, cv=10, scoring=('accuracy','f1_micro', 'f1_macro'),return_train_score=True)


# In[56]:


acc = scoresGauss['test_accuracy']
print(acc)
print("mean: "+str(np.average(acc)))
print("min: "+str(min(acc)))
print("max: "+str(max(acc)))
print("std: "+str(statistics.stdev(acc)))


# In[57]:


micro = scoresGauss['test_f1_micro']
print(micro)
print("mean: "+str(np.average(micro)))
print("min: "+str(min(micro)))
print("max: "+str(max(micro)))
print("std: "+str(statistics.stdev(micro)))


# In[58]:


macro = scoresGauss['test_f1_macro']
print(macro)
print("mean: "+str(np.average(macro)))
print("min: "+str(min(macro)))
print("max: "+str(max(macro)))
print("std: "+str(statistics.stdev(macro)))


# In[78]:


from sklearn.model_selection import KFold
 
clf2 = svm.SVC(kernel='linear') # Linear Kernel

cv = KFold(n_splits=10, random_state=42, shuffle=True)
scores = cross_validate(clf2, trainvecs, y, scoring=('accuracy','f1_micro', 'f1_macro'),return_train_score=True, cv=cv, n_jobs=-1)


# In[80]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
# we get same results each run
y_pred = cross_val_predict(clf2, trainvecs, y, cv=cv)
conf_mat = confusion_matrix(y, y_pred)


# In[107]:


conf_mat


# In[101]:


font = {'weight' : 'bold',
        'size'   : 20}

plt.rc('font', **font)


# In[104]:


import seaborn as sns  
from sklearn.metrics import confusion_matrix  
def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array):  
  
    matrix = confusion_matrix(actual_classes, predicted_classes)  
      
    plt.figure(figsize=(12.8,6))  
    sns.heatmap(matrix, annot=True, cmap="Blues", fmt="g")  
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix SVM')  
  
    plt.show()


# In[105]:


plot_confusion_matrix(y, y_pred)


# In[79]:


acc = scores['test_accuracy']
print(acc)
print("mean: "+str(np.average(acc)))
print("min: "+str(min(acc)))
print("max: "+str(max(acc)))
print("std: "+str(statistics.stdev(acc)))


# In[61]:


micro = scores['test_f1_micro']
print(micro)
print("mean: "+str(np.average(micro)))
print("min: "+str(min(micro)))
print("max: "+str(max(micro)))
print("std: "+str(statistics.stdev(micro)))


# In[62]:


macro = scores['test_f1_macro']
print(macro)
print("mean: "+str(np.average(macro)))
print("min: "+str(min(macro)))
print("max: "+str(max(macro)))
print("std: "+str(statistics.stdev(macro)))

