import numpy as np
import pandas as pd
from keras.layers import concatenate
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
import keras
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from keras.utils import np_utils
import json
# from __future__ import print_function
from functools import reduce
import json
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from keras.layers.embeddings import Embedding
from sklearn.linear_model import LogisticRegression
from keras.models import Model
from scipy.sparse import coo_matrix, hstack

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import numpy as np
import pickle
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
np.random.seed(1337)
nltk.download('stopwords')
stop=set(stopwords.words('english'))

def tokens(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def examples(fn):
    maj=True
    for i, l in enumerate(open(fn)):
        dataset = json.loads(l)
        label = dataset['gold_label']
        test_s1 = ' '.join(tokens(dataset['sentence1_binary_parse']))
        test_s2 = ' '.join(tokens(dataset['sentence2_binary_parse']))
        if maj and label == '-':
            continue
        yield (label, test_s1, test_s2)


B = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
raw_dataset = list(examples("snli_1.0/snli_1.0_train.jsonl"))
raw_dataset1 = list(examples("snli_1.0/snli_1.0_test.jsonl"))
raw_dataset2 = list(examples("snli_1.0/snli_1.0_dev.jsonl"))

left = [test_s1 for _, test_s1, test_s2 in raw_dataset]
right = [test_s2 for _, test_s1, test_s2 in raw_dataset]
Y = np.array([B[l] for l, test_s1, test_s2 in raw_dataset])
training=[left, right, Y]


left = [test_s1 for _, test_s1, test_s2 in raw_dataset1]
right = [test_s2 for _, test_s1, test_s2 in raw_dataset1]
Y = np.array([B[l] for l, test_s1, test_s2 in raw_dataset1])
test=[left, right, Y]

left = [test_s1 for _, test_s1, test_s2 in raw_dataset2]
right = [test_s2 for _, test_s1, test_s2 in raw_dataset2]
Y = np.array([B[l] for l, test_s1, test_s2 in raw_dataset2])
val=[left, right, Y]

train_s1=training[0]
train_s2=training[1]
yy=training[2]
test_s1=test[0]
test_s2=test[1]
yy1=test[2]
val_s1=val[0]
val_s2=val[1]
yy2=val[2]

# y=[np.where(r==1)[0][0] for r in yy]
# y1=[np.where(r==1)[0][0] for r in yy1]



tfidf = TfidfVectorizer(stop_words=stop,use_idf=True, max_df=0.95)

ft1 = tfidf.fit_transform(train_s1)#+train_s2)
f1=tfidf.transform(test_s1)
ff1=tfidf.transform(val_s1)
ft2 = tfidf.fit_transform(train_s2)
f2=tfidf.transform(test_s2)
ff2=tfidf.transform(val_s2)

features_X=hstack([ft1,ft2])
f_t=hstack([f1,f2])
f_t1=hstack([ff1,ff2])
print(np.array(test_s1).shape,np.array(test_s2).shape,np.array(train_s1).shape,np.array(train_s2).shape)

model = LogisticRegression().fit(features_X, yy)

filename = 'tfidf.sav'
pickle.dump(model, open(filename, 'wb'))


prediction=model.predict(f_t1)
print("Model acc is on val set is: ",model.score(f_t1, yy2)*100 )


prediction=model.predict(f_t)
print("Model acc is on test set is: ",model.score(f_t, yy1)*100 )

# prediction=loaded_model.predict(f_t)
result = model.score(f_t, yy1)
# print(result*100)

LABELS = B#{'contradiction': 0, 'neutral': 1, 'entailment': 2}
file=open("tfidf.txt","w+")
fizzbuzz=[]
for f in prediction:
    if f==0:
        fizzbuzz.append("contradiction")

    elif f==1:
        fizzbuzz.append("neutral")

    elif f==2 :

        fizzbuzz.append("entailment")

for i in fizzbuzz:
    file.write(str(i))
    file.write("\n")

#####Tunning code

# clf = LogisticRegression(solver = 'lbfgs',max_iter=100).fit(X_train, y_train)
#penalty=l2; max_iter=250
# # clf = LogisticRegression(max_iter=150).fit(X_train, y_train)
# prediction=model.predict(f_t1)
# print("Model acc is on val set is: ",model.score(f_t1, yy2)*100 )
# prediction=clf.predict(X_test)
# print("Accuracy :: " , round(accuracy_score(prediction, y_test)*100) , "%")

# clf = LogisticRegression(solver = 'lbfgs',max_iter=100).fit(X_train, y_train)
#penalty=l2; max_iter=250
# # clf = LogisticRegression(max_iter=250).fit(X_train, y_train)
# prediction=model.predict(f_t1)
# print("Model acc is on val set is: ",model.score(f_t1, yy2)*100 )
# prediction=clf.predict(X_test)
# print("Accuracy :: " , round(accuracy_score(prediction, y_test)*100) , "%")

# clf = LogisticRegression(solver = 'lbfgs',max_iter=100).fit(X_train, y_train)
# #penalty=l1; max_iter=100
# clf = LogisticRegression(penalty='l1').fit(X_train,y_train)
# prediction=model.predict(f_t1)
# print("Model acc is on val set is: ",model.score(f_t1, yy2)*100 )
# prediction=clf.predict(X_test)
# print("Accuracy :: " , round(accuracy_score(prediction, y_test)*100) , "%")

# clf = LogisticRegression(solver = 'lbfgs',max_iter=100).fit(X_train, y_train)
# #penalty=elasticnet; max_iter=100
# # clf = LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=0.5).fit(X_train,y_train)
# prediction=model.predict(f_t1)
# print("Model acc is on val set is: ",model.score(f_t1, yy2)*100 )
# prediction=clf.predict(X_test)
# print("Accuracy :: " , round(accuracy_score(prediction, y_test)*100) , "%")

# clf = LogisticRegression(solver = 'lbfgs',max_iter=100).fit(X_train, y_train)
#penalty=None; max_iter=100
# clf = LogisticRegression(penalty='none',solver='saga').fit(X_train,y_train)
# prediction=model.predict(f_t1)
# print("Model acc is on val set is: ",model.score(f_t1, yy2)*100 )
# prediction=clf.predict(X_test)
# print("Accuracy :: " , round(accuracy_score(prediction, y_test)*100) , "%")

# clf = LogisticRegression(solver = 'lbfgs',max_iter=100).fit(X_train, y_train)
# #penalty=l2; max_iter=150
# # clf = LogisticRegression(max_iter=150).fit(X_train, y_train)
# prediction=model.predict(f_t1)
# print("Model acc is on val set is: ",model.score(f_t1, yy2)*100 )
# prediction=clf.predict(X_test)
# print("Accuracy :: " , round(accuracy_score(prediction, y_test)*100) , "%")