import argparse

import os
import pathlib
import re
import json
import nltk
import numpy as np
from collections import Counter, defaultdict
from itertools import product, permutations

parser = argparse.ArgumentParser()
parser.add_argument('-c', "--corpus", help="name of the corpus")
parser.add_argument('-o', "--output", help="output directory")
args = parser.parse_args()

# english tokenizer:
def preprocess(phrase):
    return [w.lower() for w in nltk.tokenize.word_tokenize(phrase)]

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

if args.corpus.lower() in {'visual_genome', 'vg', 'visual genome', 'visualgenome'}:
    # Read from file:
    regions = json.load(open('visual_genome/region_descriptions.json'))
    phrases = [
        region['phrase']
        for regions_in_image in regions
        for region in regions_in_image['regions']
    ]
    tokenized_phrases = [
        preprocess(phrase)
        for phrase in phrases
    ]
elif args.corpus.lower() in {'mscoco', 'coco', 'ms coco'}:
    coco_captions_2017 = json.load(open('coco/2017/annotations/captions_train2017.json'))
    captions = [
        item['caption']
        for item in coco_captions_2017['annotations']
    ]
    tokenized_phrases = [
        preprocess(phrase)
        for phrase in captions
    ]
elif args.corpus.lower() in {'flickr', 'flickr30k'}:
    def flickr_tokenize(text):
        return re.sub(r'\[\/EN#[0-9]+\/', '', text).replace(']', '').strip()
    root_flickr = 'flickr30k/Flickr30kEntities/Sentences/'
    sentences = [    
        flickr_tokenize(sent)
        for filename in os.listdir(root_flickr)
        for sent in open(os.path.join(root_flickr, filename), encoding='utf8')
    ]
    tokenized_phrases = [
        preprocess(phrase)
        for phrase in sentences
    ]
else:
    print('Visual Genome Relation dataset')
    # Visual Genome Relations dataset
    rels_from_file = json.load(open('visual_genome/relationships.json'))
    def name_extract(x):
        return x['names'][0].lower() if 'names' in x and len(x['names']) else x['name'].lower() if 'name' in x else '' 
    # convert it into a set of (image, subject, predicate, object)
    triplets = {
        (rels_in_image['image_id'],
         name_extract(rel['subject']),
         composit2simple[rel['predicate'].lower()] if rel['predicate'].lower() in composit2simple else rel['predicate'].lower(),
         name_extract(rel['object']))
        for rels_in_image in rels_from_file
        for rel in rels_in_image['relationships']
        if name_extract(rel['subject']) not in composit2simple and name_extract(rel['object']) not in composit2simple
    }
    tokenized_phrases = [
        [sbj, pred, obj]
        for img,sbj,pred,obj in triplets
    ]

vocab_freq = Counter([
    w
    for phrase in tokenized_phrases
    for w in phrase
])

tokenized_phrases_filtered = [
    phrase
    for phrase in tokenized_phrases
    if len([w for w in phrase if vocab_freq[w] < 100]) == 0 # every thing is known about this sentence
]

max_len = max(len(s) for s in tokenized_phrases_filtered)

# recount the words
vocab_freq = Counter([
    w
    for phrase in tokenized_phrases_filtered
    for w in phrase
])


tokenized_phrases_prepared = [
    ['<s>'] + ['<unk>' if vocab_freq[w] < 20 else w for w in phrase] + ['</s>'] + ['<pad>']*(max_len-len(phrase))
    for phrase in tokenized_phrases_filtered
]

# recount the words
vocab_freq = Counter([
    w
    for phrase in tokenized_phrases_prepared
    for w in phrase
])

vocab = list(vocab_freq.keys())
word2index = defaultdict(lambda: vocab.index('<unk>'), ((w, i) for i, w in enumerate(vocab)))

dataset = np.array([
    [word2index[w] for w in phrase]
    for phrase in tokenized_phrases_prepared
])

np.save('{}/vocab.npy'.format(args.output), vocab)
np.save('{}/dataset.npy'.format(args.output), dataset)

composit2simple_clean = {
    rel: composit2simple[rel]
    for rel in composit2simple
    if len([w for w in rel.split() if w not in vocab]) == 0
}
np.save('{}/composit2simple.npy'.format(args.output), list(zip(*composit2simple_clean.items())))

