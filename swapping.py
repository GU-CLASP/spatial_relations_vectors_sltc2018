import argparse
# we need this for making faster bucketing
from multiprocessing import Pool

# import all required codes here
import os
import nltk
import numpy as np
from itertools import product, permutations

parser = argparse.ArgumentParser()
parser.add_argument('-v', "--vocab", help="numpy file listing all words in vocabulary")
parser.add_argument('-b', "--buckets", help="numpy file dictionary of all buckets for context swapping")
parser.add_argument('-o', "--output", help="output directory")
parser.add_argument('-t', "--threshold", type=int, help="frequency threshold for bucket swapping")
parser.add_argument('-l', "--max_len", type=int, help="maximum sentence length")
parser.add_argument('-p', "--processes", type=int, help="number of processing threads")
args = parser.parse_args()

threshold = args.threshold

buckets = np.load(args.buckets)[None][0]
vocab = list(np.load(args.vocab))

max_rel_size = max(len(rel.split()) for rel in buckets)

if args.max_len:
    max_sent_len = args.max_len
else:
    max_sent_len = max(buckets[rel][0][0].shape[0] for rel in buckets)

ready_buckets = buckets
iterate = 2
for _ in range(iterate):
    # long sentences are not valid
    is_valid_instance = {
        rel: np.array([len([w for w in sent if vocab[w] != '<pad>']) for sent,_ in ready_buckets[rel]]) <= max_sent_len
        for rel in ready_buckets
        if len(ready_buckets[rel]) > threshold
    }
    ready_buckets = {
        rel: [instance for is_valid, instance in zip(is_valid_instance[rel],ready_buckets[rel]) if is_valid]
        for rel in ready_buckets
        if len(ready_buckets[rel]) > threshold
    }
    # long sentences are not valid
    is_valid_instance = {
        rel: np.array([len([w for w in sent if vocab[w] != '<pad>']) for sent,_ in ready_buckets[rel]])+max_rel_size-len(rel.split()) <= max_sent_len
        for rel in ready_buckets
        if len(ready_buckets[rel]) > threshold
    }
    # only swap ready instances
    ready_buckets = {
        rel: [instance for is_valid, instance in zip(is_valid_instance[rel],ready_buckets[rel]) if is_valid]
        for rel in ready_buckets
        if len(ready_buckets[rel]) > threshold
    }

buckets = ready_buckets
def swap(old, new, context=None):
    if context is None:
        context = buckets[old]
    def replace(sequence, new, where):
        seq = list(sequence)
        start, length = where
        return (seq[:start] + list(new) +  seq[length+start:] + [vocab.index('<pad>') if '<pad>' in vocab else -1]*(length-len(new)))[:len(seq)]
        
    np.save(
        '{dir}/{context}-{word}.npy'.format(dir=args.output,context=old,word=new), 
        list(map(lambda x: replace(x[0], [vocab.index(w) for w in new.split()], (x[1], len(old.split()))), context))
    )

def temp(x):
    return swap(old=x[0], new=x[1], context=buckets[x[0]])


pool_size = args.processes if args.processes else 16
with Pool(pool_size) as p:
    results = p.map(temp, [(old,new) for old in buckets for new in buckets])


