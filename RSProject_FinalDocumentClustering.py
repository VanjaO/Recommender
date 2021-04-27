#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json ## JSON is a syntax for storing and exchanging data.JSON is text, written with JavaScript object notation.
import os ## This module provides a portable way of using operating system-dependent functionality
import pandas as pd
import numpy as np
import ExplicitMF as mf

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import nltk; print(nltk.__version__)
import nltk.tokenize
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer

import pandas as pd
from nltk.corpus import stopwords
stop_words = stopwords.words('norwegian')
from nltk.tokenize import word_tokenize
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
import numpy as np


# In[ ]:


def load_data(path):
    """
        Load events from files and convert to dataframe.
    """
    map_lst=[]
    for f in os.listdir(path):
        file_name=os.path.join(path,f)
        if os.path.isfile(file_name):
            for line in open(file_name):
                obj = json.loads(line.strip()) ## The result will be a Python dictionary.
                if not obj is None:
                    map_lst.append(obj)
    return pd.DataFrame(map_lst) ## Two-dimensional, size-mutable, potentially heterogeneous tabular data.


# In[ ]:


if __name__ == '__main__':
    df=load_data("C:/RSProject/Data_active1000")
mdf2= df[['userId','eventId','category','title','url','activeTime', 'documentId']] #.iloc[0:1000]  
mdf2 = mdf2[mdf2.documentId.notnull()] # Removing those events that documentId is null
print(mdf2)


# In[ ]:


print (mdf2.info())


# In[ ]:


mdf=mdf2
print(mdf["title"])


# In[ ]:


def replace_norwegian_char(my_text):
    """ This function replaces the Norwegian
    specific language alphabets with its ASCII
    unicode in the URL """
    my_text=my_text.lower()
    my_text=my_text.replace('%c3%a5', 'a')
    my_text=my_text.replace('%c3%a6', 'æ')
    my_text=my_text.replace('%c3%b8', 'o')
    my_text=my_text.replace('å', 'a')
    my_text=my_text.replace('ø', 'o')
    return (my_text)


# In[ ]:


# # delete stopwords for title
def make_lower_text(text):
    text=text.lower()
    return text
mdf["title"]=  mdf["title"].apply(replace_norwegian_char)
print(mdf["title"])
mdf["title"]=  mdf["title"].apply(make_lower_text)
print(mdf["title"])
mdf["title"]= mdf["title"].apply(word_tokenize)
print(mdf["title"])
mdf["title"]= mdf["title"].apply(lambda x: [item for item in x if item not in stop_words]) 
print(mdf["title"])


# In[ ]:


# delete tokenized words with lengths less than 2

mdf["title"]= mdf["title"].apply(lambda x: [item for item in x if len(item) >= 2])
print(mdf["title"])


# In[ ]:


url2=[]
for x in list((mdf['url'])):
    print(x)
    Is=x.rindex("/")
    Ie=x.rindex(".")
    x=x.replace("-", " ")
    x=x[Is+1:Ie]
    url2.append(x)
    
mdf['url2']=url2
print(mdf['url2'])


# In[ ]:


def clean_text(text):
    import re
    import string
    # remove numbers
    text_nonum = re.sub(r'\d+', ' ', text) #re.sub(pattern, repl, string) 
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation]) 
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return text_no_doublespace


mdf['url2']= mdf['url2'].apply(clean_text)
print(mdf['url2'])


# In[ ]:


mdf['url2']=mdf['url2'].apply(replace_norwegian_char)
print(mdf['url2'])


# In[ ]:


# delete stopwords for url2

mdf["url2"]= mdf["url2"].apply(word_tokenize)
mdf["url2"]= mdf["url2"].apply(lambda x: [item for item in x if item not in stop_words]) 
#print(mdf["url2"])

mdf["url2"]= mdf["url2"].apply(lambda x: [item for item in x if len(item) >= 2])
print(mdf["url2"])


# In[ ]:


# Creating "fcate" as a combined field of (title) and (url2)
mdf["fcate"]= mdf["title"] + mdf["url2"]
mdf['fcate'] = mdf['fcate'].apply(set).apply(list)
print(mdf['fcate'])


# In[ ]:


fcate = mdf['fcate']
print (fcate, type(fcate))


# In[ ]:


import gensim
from gensim.models import Word2Vec
import nltk
import numpy as np

from sklearn import cluster
from sklearn.cluster import KMeans

from sklearn import metrics
from sklearn.decomposition import PCA


# In[ ]:


m = Word2Vec(fcate, vector_size = 100, min_count=1, sg=1)


# In[ ]:


"""
We have created a model using Word2Vec which has min_count=1, min_count=1 is the window size for the seeking the words.
A function, vectorizer, is created which will help in addition to each word’s vector in a sentence and then dividing by
the total number of words in that particular sentence.
"""
def vectorizer(m, sent):
    vec=[]
    numw= 0
    for w in sent:
        try:
            if numw == 0:
                vec = m.wv[w]
            else:
                vec = np.add(vec, m.wv[w])
                #print(vec)
            numw+=1
        except:
            pass
    
    return np.asarray(vec)/ numw

l=[]
for i in fcate:
    l.append(vectorizer(m, i))
x = np.array(l)
#print(x)


# In[ ]:


import matplotlib.pyplot as plt
wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,10),wcss)
plt.title('The Elbow method')
plt.xlabel('Number of cluters')
plt.ylabel('wcss')
plt.show()


# In[ ]:


n_clusters = 10
clf = KMeans(n_clusters = n_clusters,
            max_iter = 100,
            init = 'k-means++',
            n_init = 1)
labels = clf.fit_predict(x)
print(labels)
for index, fcate in enumerate (fcate):
    print(str(labels[index])+ ":" + str(fcate))

