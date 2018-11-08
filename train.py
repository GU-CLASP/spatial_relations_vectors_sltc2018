import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', "--vocab", help="numpy file listing all words in vocabulary")
parser.add_argument('-d', "--dataset", help="numpy matrix of all words in sentences pre-processed ready for training the language model")
parser.add_argument('-o', "--output", help="output directory")
parser.add_argument('-g', "--gpu", help="GPU listing number")
args = parser.parse_args()

import os
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding, TimeDistributed, Masking
from keras.layers import Lambda, Reshape, Activation
import keras.backend as K


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


# model definition
def build_model(
        max_len,
        vocab_size,
        embedding_size = 300,
        memory_size = 300,
        mask=None
    ):
    if mask==None:
        mask=-1
    lm = Sequential([
        Masking(mask_value=mask, input_shape=[max_len+1,]),
        Embedding(vocab_size, embedding_size),
        LSTM(memory_size, return_sequences=True, ), 
        TimeDistributed(Dense(vocab_size, activation='softmax')),
    ])
    lm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return lm

# load the data
X_train = np.load(args.dataset)
vocab = list(np.load(args.vocab))
max_len = X_train.shape[1]-2

# training
lmx = build_model(
    max_len = max_len,
    vocab_size = len(vocab),
    embedding_size = 300,
    memory_size = 300,
    mask=vocab.index('<pad>') if '<pad>' in vocab else -1
)
lmx.fit(X_train[:, :-1], np.expand_dims(X_train[:, 1:], 2), epochs=20, batch_size=1024)

lmx.save('{}/language_model.h5'.format(args.output))
