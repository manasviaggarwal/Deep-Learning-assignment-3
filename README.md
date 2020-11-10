# Deep Learning assignment 3

This is the third assignment of Deep Learning course 2020. 

# Task 1: LOGISTIC REGRESSION

In this part of the assignment, I implement a Logistic regression using TF-IDF features in
python for the task of Natural Language Inference on SNLI dataset. Basic preprocessing is
carried out on the data such as tokenizing or removing stop words. I trained a logistic regression model using TF-IDF features which are formed by using TfidfVectorizer.

# Task 2: DEEP MODEL

In this task, I build a Deep model (Recurrent Neural
Network) specific for text for NLI (Natural Language Inference) i.e. to predict if the sentence pair constitutes entailment/contradiction/neutral for SNLI dataset. I implement a Recurrent Neural
Network for this task. For better performance of the model I used glove pretrained embeddings
to convert words to a d-dimension vectors. I also tested without these pretrained embeddings.


To test both the models, run main.py which will run tfidf_test.py and deep_model_test.py to test logistic regression and deep learning models respectively.
Also, tfidf_train.py is the code used for traning task 1 i.e. logistic regression model and deep_model_train.py is the code used to train deep learning (RNN) model.
To run these files, please add the dataset and the pretrained glove weights from the links given in the report. model directory contains both the trained models and tfidf.txt and deep_model.txt are the output files of logistic regression and deep learning (RNN) models respectively.


