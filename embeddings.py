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

spatial_lexicon = list({
    w
    for filename in os.listdir(root_dir)
    for w in filename.split('.')[0].split('-')[1].split()
    if w in vocab
})

embeddings = [
    lmx.layers[1].get_weights()[0][vocab.index(w)]
    for w in spatial_lexicon
]


np.save('{dir}/embeddings.npy'.format(dir=args.output), embeddings)
np.save('{dir}/spatial_lexicon.npy'.format(dir=args.output), spatial_lexicon)

