"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
from nltk.corpus import wordnet
import pickle
import torch

MODEL="stsb-roberta-large"
VOCAB="data/miniImageNet/labels.txt"
SAVE_PATH="description_embeds/miniImageNet_sbert.pickle"

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Retrive labels and definitions.
with open(VOCAB) as f:
    vocab = [line.rstrip() for line in f]
defs = [wordnet.synsets(v.replace(" ", "_"))[0].definition() for v in vocab]

# Load pre-trained Sentence Transformer Model (based on DistilBERT). It will be downloaded automatically
model = SentenceTransformer(MODEL)

# Embed a list of sentences
sentence_embeddings = model.encode(defs)
sentence_embeddings = [torch.FloatTensor(e) for e in sentence_embeddings]

# The result is a list of sentence embeddings as numpy arrays
d = dict(zip(vocab, sentence_embeddings))
with open(SAVE_PATH, 'wb') as f1:
    pickle.dump(d, f1)
