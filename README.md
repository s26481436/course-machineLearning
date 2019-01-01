# course: machineLearning
	This repository is a project for machine learning course.
	This project presents using different text classification approaches to classify the H section category of Chinese-language patent and figure out the precision of classification.

# catchData.py
	This is the part of code for data preprocessing. In this part, we will import the data from mongodb. Then, we use jieba for the Chinese words segmentation and remove all the stop words. We will use the Term Frequency-Inverse Document Frequency (TF-IDF) to analyze the weight of word in each document and extract the keywords.

# What is jieba?
	Jieba is a MIT-authorized Chinese segmentation open source in Python language and it is a module that is currently the best module for Chinese word segmentation.
	What can jieba do for us?
	Jieba can segment Chinese words and use TF-IDF algorithm to present the several keywords.

When the code has done, it will save the data as a .txt file automatically and it will be use for the next step. This is the important part for the classification, so we have to run this part first.

# Classification.py
	After run the catchData.py, this part will load the file “train.txt” and “test.txt” as arrays.
	This code will generate two vector models, one is word2vec and another one is fastText, this two model will be the based model.
	We use these model to represent the train and test words to word vector by line by line and summarize these vectors, then we will get the data vector arrays.
	Now, we can use these vector arrays to classification via Nearest Neighbor, Random Forest, SVM etc.
	In this repository, we use Nearest Neighbor, Random Forest, SVM, Gaussian Naive Bayes as our classifier.

# Train.txt Test.txt
	Text file for model training and test classification.
	Train.txt has 450 columns of text, test.txt has 50 columns of text.

# These files are only for education research, not for commercial use.
