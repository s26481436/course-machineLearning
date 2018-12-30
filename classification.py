# -*- coding: utf-8 -*-
# import MongoClient from pymongo
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
    
    rfErr = 0.0

    #classification via random forest
    clf = RandomForestClassifier(n_estimators=20, max_depth=30, random_state = 0)
    clf.fit(xVec,y)
    result = clf.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            rfErr +=1
    
    #classification via svm
    svmErr = 0.0
    clf = svm.SVC(C=0.1)
    clf.fit(xVec,y)
    result = clf.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            svmErr +=1
    

    #classification via Gaussian Naive Bayes
    gnb = naive_bayes.GaussianNB()
    gnbErr = 0.0
    gnb.fit(xVec,y)
    result = gnb.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            gnbErr +=1
    

    #classification via Decision Tree
    dctErr = 0.0
    clf = tree.DecisionTreeClassifier()
    clf.fit(xVec,y)
    result = clf.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            dctErr +=1
    

    
    return ([modelMode, nnErr , rfErr , svmErr , gnbErr , dctErr])

x, y = load_text('train.txt')
tx,ty = load_text('test.txt')


startTime = time.time()
#train a new model that base on word2vec
model_w2v = Word2Vec(x,min_count=5,size=40)

w2vTime = time.time()

#train a new model that base on fastText
model_FT = FastText(x, size = 40 , min_count = 5)

FFTime = time.time()

print ("Word2Vec model training Time : %s" %(str(w2vTime - startTime)))
print ("FastText model training Time : %s" %(str(FFTime - w2vTime)))


w2vCla = classification("Word2Vec",model_w2v)


FFCla = classification("FastText",model_FT)

print ("Training data: %d" %(len(y)))
print ("Testing data: %d" %(len(ty)))
print ("Base on %s: 1-NN err rate: %.2f%%" % (w2vCla[0], w2vCla[1] / len(ty) * 100))
print ("Base on %s: Random Forest err rate: %.2f%%" % (w2vCla[0], w2vCla[2] / len(ty) * 100))
print ("Base on %s: SVM err rate: %.2f%%" % (w2vCla[0], w2vCla[3] / len(ty) * 100))
print ("Base on %s: Gaussian Bayes err rate: %.2f%%" % (w2vCla[0], w2vCla[4] / len(ty) * 100))
print ("Base on %s: Decision Tree err rate: %.2f%%" % (w2vCla[0], w2vCla[5] / len(ty) * 100))

print ("Base on %s: 1-NN err rate: %.2f%%" % (FFCla[0], FFCla[1] / len(ty) * 100))
print ("Base on %s: Random Forest err rate: %.2f%%" % (FFCla[0], FFCla[2] / len(ty) * 100))
print ("Base on %s: SVM err rate: %.2f%%" % (FFCla[0], FFCla[3] / len(ty) * 100))
print ("Base on %s: Gaussian Bayes err rate: %.2f%%" % (FFCla[0], FFCla[4] / len(ty) * 100))
print ("Base on %s: Decision Tree err rate: %.2f%%" % (FFCla[0], FFCla[5] / len(ty) * 100))
