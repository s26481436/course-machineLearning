# -*- coding: utf-8 -*-
# import MongoClient from pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
import re
import pdb
import jieba
import jieba.analyse
import io
import pandas as pd
from gensim.models.word2vec import Word2Vec
import numpy
# Create a MongoDB client
mongoDBConnection   = MongoClient('address', 'port', username='userName',password='pwd')


# Get a collection instance
collection        = mongoDBConnection.TIPO.IPC_descriptions

#the database format
#description  |  ipcs
#  text       |   H01


field1 = "ipcs"
field2 = "description"
i = 0
sentences = []

text = []
stopWords = []
labels = []

#save collection because next step need to make sure
#about the dataset is not has another clss data
des = collection.find({},{"_id": 1,field2:1});


for cache in collection.find({},{"_id": 1,field1:1}):
    #preprocess.
    #this step is repleace some flag,
    #cause these maybe impact the classifier and model efficiency.
    
    cache = str(cache)

    cache = cache.strip("{'")
    cache = cache.strip(field1)
    cache = cache.strip(": '")
    cache = cache.strip("'}")
    cache = cache.strip("{'")
    cache = cache.strip("_id': ObjectId('")
    id = cache[0:24]
    cache = cache.replace(id,"")
    cache = cache.strip("'), 'ipcs': ")
    cache = re.sub('[，。、]+', "/", cache)
    #filter class and make sure there are all of H class
    #And catch the sub-class number
    if (cache[3:4] == "H"):
        labels.append(int(cache[5:6]))
        temp = des[i]
        temp = str(temp)
        temp = temp.strip("{'")
        temp = temp.strip(field2)
        temp = temp.strip(": '")
        temp = temp.strip("'}")
        temp = temp.strip("{'")
        temp = temp.strip("_id': ObjectId('")
        id = temp[0:24]
        temp = temp.replace(id,"")
        temp = temp.strip("'), 'description': ")
        temp = re.sub('[，。、]+', "/", temp)
        #this line is very important,
        #because some of html flag is found often, like hyperlink,
        #so we have to repleace these flags.
        results=re.compile(r'<a [a-zA-Z0-9.?/&=:^a-zA-Z0-9 a>]*',re.S)
        temp=results.sub("",temp)
        results=re.compile(r'</a>',re.S)
        temp=results.sub("",temp)
        results=re.compile(r'[0-9]*',re.S)
        temp=results.sub("",temp)

        #jieba has TFIDF function that can extrat top N words
        #we need configure the TFIDF setting
        #let it can't extrat word what is non-mean word
        jieba.analyse.set_stop_words("stopwords.txt")
        
        sentence = jieba.analyse.extract_tags(temp,topK=40)
        sentence = "|".join(sentence)
        
        sentences = sentence.split("|")
        text.append(sentences)
                    
        sentences = []

    print(i)
    i=i+1
    
    #set limit for test code has no error
    if len(text) >= 10000:
        break
    
    
#train data : test data => 9:1
train_x = text[0:int(len(text) / 10 * 9)]
train_y = labels[0:int(len(text) / 10 * 9)]
test_x = text[len(train_x) : ]
test_y = labels[len(train_x) : ]

#save the datasets as txt files
with io.open("train.txt",mode="a", encoding='utf-8') as rf:
    for i in range(len(train_x)):
        rf.write("%s |%d\n" % (train_x[i],train_y[i]))
with io.open("test.txt",mode="a", encoding='utf-8') as rf:
    for j in range(len(test_x)):
        rf.write("%s |%d\n" % (test_x[j],test_y[j]))


