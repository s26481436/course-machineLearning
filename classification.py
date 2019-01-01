# -*- coding: utf-8 -*-
# import MongoClient from pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
import re
import pdb
import jieba
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import naive_bayes
from gensim.models import FastText
import numpy as np
from sklearn import svm
from gensim.models.word2vec import Word2Vec
from sklearn import tree
import time


#load data and data label in dataset
def load_text(f):
    with io.open(f,mode="r", encoding='utf-8') as rf:
        lines=rf.readlines()
        words, lab=[], []
        for i in range(len(lines)):
            fn, label = lines[i].split(' |')
            fn = fn[2:-2].split("', '")
            if fn[0] != "":
                words.append(fn) 
                lab.append(int(label))
            
        words= np.asarray(words)
        lab= np.asarray(lab, np.int32)
        return words, lab 

# classification(string modelMode ,vectorModel model)
def classification(modelMode,model):
    xVec = []
    txVec = []
    
    #Transfor natual language word to vector via word2vec or fasttext model
    for i in range(len(x)):
        temp = [];
        for j in range(len(x[i])):
            #Here we need to check test word is in model or not
            #Because if word is not in model
            #Then we can't get the word vector (base on word2vec)
            if(x[i][j] in model):
                temp.append(model[x[i][j]])
            #Here is summarize word vectors
            #Now, we get a vector which is a column of words
        xVec.append(sum(temp) / len(temp))
    
    #Test data need to do the same steps like train data
    for i in range(len(tx)):
        temp = [];
        for j in range(len(tx[i])):
            if(tx[i][j] in model):
                temp.append(model[tx[i][j]])
        txVec.append(sum(temp) / len(temp))
    claTime = time.time()
    nnErr = 0.0
    
    #classification via nearst neighbor
    #Find out the nearst vector
    #And classificat
    for k in range(len(txVec)):
        dist = []
        for m in range(len(xVec)):
            dist.append(np.sum(np.abs(txVec[k] - xVec[m])))
        if(y[dist.index(min(dist))] != ty[k]):
            nnErr += 1
    nnTime = time.time()
    rfErr = 0.0
    
    
    clf = RandomForestClassifier(n_estimators=20, max_depth=30, random_state = 0)
    clf.fit(xVec,y)
    result = clf.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            rfErr +=1
    rfTime = time.time()

    svmErr = 0.0
    
    clf = svm.SVC(C=0.1)
    clf.fit(xVec,y)
    result = clf.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            svmErr +=1
    
    svmTime = time.time()

    gnb = naive_bayes.GaussianNB()
    gnbErr = 0.0
    gnb.fit(xVec,y)
    result = gnb.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            gnbErr +=1
    

    gnbTime = time.time()
    dctErr = 0.0
    clf = tree.DecisionTreeClassifier()
    clf.fit(xVec,y)
    result = clf.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            dctErr +=1
    dctTime = time.time()
    
        
    return ([modelMode, nnErr , rfErr , svmErr , gnbErr , dctErr, claTime , nnTime , rfTime , svmTime , gnbTime , dctTime])

x, y = load_text('train.txt')
tx,ty = load_text('test.txt')



#train a new model that base on word2vec
startTime = time.time()

model_w2v = Word2Vec(x,min_count=5,size=40,workers=4)

w2vTime = time.time()


#train a new model that base on fastText
model_FT = FastText(x, size = 40 , min_count = 5 ,workers=4)
FTTime = time.time()





w2vCla = classification("Word2Vec",model_w2v)




FFCla = classification("FastText",model_FT)

for i in range(1):
    print ("Training data: %d" %(len(y)))
    print ("Testing data: %d" %(len(ty)))
    print ("Word2Vec Training Time: %.2fs" % (w2vTime - startTime))
    print ("FastText Training Time: %.2fs" % (FTTime - w2vTime))
    print ("Base on %s: 1-NN accuracy rate: %.2f%% , cost time: %.2f(s)" % (w2vCla[0],100-( w2vCla[1] / len(ty) * 100),(w2vCla[7] - w2vCla[6])))
    print ("Base on %s: Random Forest accuracy rate: %.2f%% , cost time: %.2f(s)" % (w2vCla[0],100-( w2vCla[2] / len(ty) * 100),(w2vCla[8] - w2vCla[7])))
    print ("Base on %s: SVM accuracy rate: %.2f%% , cost time: %.2f(s)" % (w2vCla[0],100-( w2vCla[3] / len(ty) * 100),(w2vCla[9] - w2vCla[8])))
    print ("Base on %s: Gaussian Bayes accuracy rate: %.2f%% , cost time: %.2f(s)" % (w2vCla[0],100-( w2vCla[4] / len(ty) * 100),(w2vCla[10] - w2vCla[9])))
    print ("Base on %s: Decision Tree accuracy rate: %.2f%% , cost time: %.2f(s)" % (w2vCla[0],100-( w2vCla[5] / len(ty) * 100),(w2vCla[11] - w2vCla[10])))

    print ("Base on %s: 1-NN accuracy rate: %.2f%% , cost time: %.2f(s)" % (FFCla[0],100-( FFCla[1] / len(ty) * 100),(FFCla[7] - FFCla[6])))
    print ("Base on %s: Random Forest accuracy rate: %.2f%% , cost time: %.2f(s)" % (FFCla[0],100-( FFCla[2] / len(ty) * 100),(FFCla[8] - FFCla[7])))
    print ("Base on %s: SVM accuracy rate: %.2f%% , cost time: %.2f(s)" % (FFCla[0],100-( FFCla[3] / len(ty) * 100),(FFCla[9] - FFCla[8])))
    print ("Base on %s: Gaussian Bayes accuracy rate: %.2f%% , cost time: %.2f(s)" % (FFCla[0], 100-(FFCla[4] / len(ty) * 100),(FFCla[10] - FFCla[9])))
    print ("Base on %s: Decision Tree accuracy rate: %.2f%% , cost time: %.2f(s)" % (FFCla[0], 100-(FFCla[5] / len(ty) * 100),(FFCla[11] - FFCla[10])))
