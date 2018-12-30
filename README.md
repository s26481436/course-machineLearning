# course:machineLearning
	This Repository is a project for npust machine learing course.
	Our target is classified patents and compare classifier effiency.

# python package
	jieba
	numpy
	sklearn
	pymongo
	gensim


Here has two parts of machine learning code.
	catchData.py , classification.py

# catchData.py
	This part of code which catch the database ,and run jieba,TF-IDF to extract keywords.

	What is jieba?
	Jieba is a module that is current the best module for chinese word segmentation.
	what can jieba do for us?
	Jieba can segment chinese words and use TF-IDF algorithm return several keywords.
	when catching has done, it will save the data as a .txt file for use this database next time, so we have to run this part first, if you have no database.

# classification.py
	You have to run catchData.py, then this part will load file "train.txt","test.txt" as arrays.
	This code will build two vector models. One is word2vec based model,the orther is fastText based model.

	We use these model to transfor train,test words to word vector line by line, and summarize these vectors, then we get data vector arrays now.

	Now, we can use these vector arrays to classification via Neart Neighbor, Random Forest, SVM etc.

	In this repository, we use Neart Neighbor, Random Forest, SVM, Gaussian Naive Bayes classifier.
