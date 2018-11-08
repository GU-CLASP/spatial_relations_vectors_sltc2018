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
parser.add_argument('-d', "--dataset", help="numpy matrix of all words in sentences pre-processed")
parser.add_argument('-o', "--output", help="output directory")
parser.add_argument('-s', "--seed", type=int, help="random seed to break the dataset")
parser.add_argument('-p', "--processes", type=int, help="number of processing threads")
args = parser.parse_args()

if args.seed:
    seed = args.seed
else:
    seed = int(np.pi*10002)

#spatial_terms = ['in', 'on', 'at', 'to', 'above', 'below', 'over', 'under']

# Landau English prepositions
en_preps = [
    # simple spatial relations
    'at', 'on', 'in', 'on', 'off',
    'out', 'by', 'from', 'to',
    'up', 'down', 'over', 'under',
    'with', ('within', 'with in'), ('without', 'with out'), 'near',
    'neadby', 'into', ('onto', 'on to'), 'toward',
    'through', 'throughout', 'underneath', 'along',
    'across', ('among', 'amongst'), 'against', 'around',
    'about', 'above', ('amid', 'amidst'), 'before',
    'behind', 'below', 'beneath', 'between',
    'beside', 'outside', 'inside', ('alongside', 'along side'),
    'via', 'after', 'upon', 
    # compounds
    ('top', 'on top of'), ('between', 'in between'), ('right', 'to the right of'), ('parallel', 'parallel to'),
    ('back', 'in back of'), ('left', 'to the left of'), ('side', 'to the side'), ('perpendicular', 'perpendicular to'),
    ('front', 'in front of'),
    # temporal only
    'during', 'since', 'until', 'ago',
    # intransitivies (+ additional variations)
    'here', 'outward', ('backward', 'backwards'), ('south' , 'south of'),
    'there', ('afterward', 'afterwards'), 'away', ('east', 'east of'),
    'upward', 'upstairs', 'apart', ('west', 'west of'),
    'downward', 'downstairs', 'together', 'left',
    'inward', 'sideways', ('north', 'north of'), 'right',
]

# Herskovits projective_terms
en_preps += [(w2, w1+' the '+w2+' of')           for w1 in ['at', 'on', 'to', 'by'] for w2 in ['left', 'right'] ]
en_preps += [(w2, w1+' the '+w2+' side of')      for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['left', 'right']]
en_preps += [(w2, w1+' the '+w2+' hand side of') for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['left', 'right']]
en_preps += [(w2, w1+' the '+w2+' of')           for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['front', 'back', 'side']]
en_preps += [(w1, 'in '+w1+' of')                for w1 in ['front', 'back']]
en_preps += [(w1,)                               for w1 in ['before', 'behind']]
en_preps += [(w1, w1+' of')                      for w1 in ['left', 'right', 'back']]
en_preps += [(w1,)                               for w1 in ['above', 'below']]
en_preps += [(w1,)                               for w1 in ['over', 'under']]
en_preps += [(w2, w1+' the '+w2+' of')           for w1 in ['at', 'on', 'in', 'by'] for w2 in ['top', 'bottom']]
en_preps += [(w2, w1+' '+w2+' of')               for w1 in ['on'] for w2 in ['top']]

# missing items?
en_preps += [('next', 'next to')]

# missing odd variations
en_preps += [('front', 'on the front of', 'on front of')]
en_preps += [('left', 'in the left of', 'in left of'),('right', 'in the right of', 'in right of'),]

# missing 'the'
en_preps += [(w2, w1+' '+w2+' of')           for w1 in ['at', 'on', 'to', 'by'] for w2 in ['left', 'right'] ]
en_preps += [(w2, w1+' '+w2+' side of')      for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['left', 'right']]
en_preps += [(w2, w1+' '+w2+' hand side of') for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['left', 'right']]
en_preps += [(w2, w1+' '+w2+' of')           for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['front', 'back', 'side']]
en_preps += [(w2, w1+' '+w2+' of')           for w1 in ['at', 'on', 'in', 'by'] for w2 in ['top', 'bottom']]

# compositional variation
en_preps += [
    (w2+'_'+w3, w1+_the_+w2+_and_+w3+' of')
    for w1 in ['at', 'on', 'in', 'to', 'by', 'to']
    for _the_ in [' ', ' the ']
    for _and_ in [' ', ' and ']
    for x, y in permutations([
	['upper', 'lower'],
        ['left', 'right',],
        ['front', 'back',],
        ['top', 'bottom'],
        ['before', 'behind'],
        ['above', 'over', 'under', 'below', ],
        #['next', 'close', 'far']
    ], 2)
    for w2, w3 in product(x,y)
]

# fix the tuple types
en_preps = [(w,) if type(w) != tuple else w for w in en_preps]

# This will create a ditionary of preposition variations to a simple tocken
composit2simple = dict()
composit2simple.update({w_alt: w[0] for w in en_preps for w_alt in w})
composit2simple.update({w: w        for w in composit2simple.values()})

vocab = list(np.load(args.vocab))
dataset = np.load(args.dataset)
word2index = lambda w: vocab.index(w) if w in vocab else vocab.index('<unk>')

# shuffle the index and then select from that
X_index = np.arange(dataset.shape[0])

np.random.seed(seed)
np.random.shuffle(X_index)

X = dataset[X_index]

test_split = int(X.shape[0]/10)

X_test = X[:test_split]
X_train = X[test_split:]

# convert all multi word relations into code sequences
rels_codes = {
    rel: [word2index(w) for w in nltk.tokenize.word_tokenize(rel)]
    for rel in composit2simple
}

# we don't want unknown relations
rels_codes = {
    rel: rels_codes[rel]
    for rel in rels_codes
    if word2index('<unk>') not in rels_codes[rel]
}

# this function finds a pattern in a sequence
def find_sub(source, pattern):
    pattern = tuple(pattern)
    partitions = [tuple(source[i:i+len(pattern)]) for i in range(0, len(source)-len(pattern)+1)]
    if pattern in partitions:
        return partitions.index(pattern)
    else:
        return -1

def bucketting(X):
    buckets = {
        rel: []
        for rel in rels_codes
    }
    rels = sorted(rels_codes.keys(), key=lambda x: len(x.split()), reverse=True)
    for index, sent in enumerate(X):
        for rel in rels:
            find = find_sub(sent, rels_codes[rel])
            if find > -1:
                buckets[rel].append((sent, find))
                break
    return buckets

pool_size = args.processes if args.processes else 16
with Pool(pool_size) as p:
    pace = int(X_test.shape[0] / pool_size)
    meta_bucket = p.map(bucketting, [X_test[i:i+pace] for i in range(0, X_test.shape[0], pace)])

final_buckets = dict([])
for buckets in meta_bucket:
    for rel in buckets:
        if len(buckets[rel]) > 0:
            if rel not in final_buckets:
                final_buckets[rel] = buckets[rel]
            else:
                final_buckets[rel] += buckets[rel]

np.save('{}/buckets.npy'.format(args.output), final_buckets)
np.save('{}/dataset_train.npy'.format(args.output), X_train)

