import argparse
import os
import numpy as np

from keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model", help="the language model in h5 format for Keras")
parser.add_argument('-v', "--vocab", help="numpy file listing all words in vocabulary")
parser.add_argument('-i', "--input", help="directory of swapped contexts-words")
parser.add_argument('-o', "--output", help="output directory")
parser.add_argument('-g', "--gpu", help="GPU listing number")
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

root_dir = args.input
vocab = list(np.load(args.vocab))
lmx = load_model(args.model)

def perplexity(lm, X):
    outputs = lm.predict(X[:, :-1])
    results = np.array([
        outputs[i, j, X[i, j+1]] 
        for i in range(outputs.shape[0])
        for j in range(outputs.shape[1]) 
        if vocab[X[i, j+1]] not in {'<pad>', '</s>'}
    ])
    pp = 2**(np.sum(-np.log2(results))/results.shape[0])
    return pp

results = {
    tuple(filename.split('.')[0].split('-')): perplexity(lmx, np.load(os.path.join(root_dir, filename)))
    for filename in os.listdir(root_dir)
}

contexts, words = zip(*list(results.keys()))
contexts = list(set(contexts))
words = list(set(words))
embeddings = [
    [results[(c,w)] for c in contexts]
    for w in words
]


np.save('{dir}/perplexity_vectors.npy'.format(dir=args.output), embeddings)
np.save('{dir}/words_contexts.npy'.format(dir=args.output), (words,contexts))

