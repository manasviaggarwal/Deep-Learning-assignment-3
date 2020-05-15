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
filename = 'tfidf.sav'
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
left = [test_s1 for _, test_s1, test_s2 in raw_dataset]
right = [test_s2 for _, test_s1, test_s2 in raw_dataset]

Y = np.array([B[l] for l, test_s1, test_s2 in raw_dataset])
training=[left, right, Y]


left = [test_s1 for _, test_s1, test_s2 in raw_dataset1]
right = [test_s2 for _, test_s1, test_s2 in raw_dataset1]
Y = np.array([B[l] for l, test_s1, test_s2 in raw_dataset1])
test=[left, right, Y]

train_s1=training[0]
train_s2=training[1]
yy=training[2]
test_s1=test[0]
test_s2=test[1]
yy1=test[2]

# y=[np.where(r==1)[0][0] for r in yy]
# y1=[np.where(r==1)[0][0] for r in yy1]



tfidf = TfidfVectorizer(stop_words=stop,use_idf=True, max_df=0.95)

ft1 = tfidf.fit_transform(train_s1)#+train_s2)
f1=tfidf.transform(test_s1)
ft2 = tfidf.fit_transform(train_s2)
f2=tfidf.transform(test_s2)

features_X=hstack([ft1,ft2])
f_t=hstack([f1,f2])
# print(np.array(test_s1).shape,np.array(test_s2).shape,np.array(train_s1).shape,np.array(train_s2).shape)

loaded_model = pickle.load(open(filename, 'rb'))
prediction=loaded_model.predict(f_t)
result = loaded_model.score(f_t, yy1)
print('Test accuracy =', result*100)

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

