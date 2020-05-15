from __future__ import print_function
from functools import reduce
import json
import os
import re
import tarfile
import tempfile

import numpy as np
np.random.seed(1337)

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import concatenate
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from tfidf_test import test_logistic 
from deep_model_test import test_deep

if __name__ == "__main__":
	print("preprocessing Logistic Regression Model")
	test_logistic()

	print("Deep Neural Model")
	test_deep()