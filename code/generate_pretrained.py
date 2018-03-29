import gensim
import numpy as np
import sys
from gensim.scripts.glove2word2vec import glove2word2vec
import os.path

# Usage: python3 generate_pretrained.py <pretrained_file>

vocab = {}
embedding_size = 300
path_to_bin = sys.argv[1]

with open("../dataset/CoNLL-2003/vocab_dict") as f:
    for line in f:
        pairs = line.strip().split()
        vocab[pairs[0]] = int(pairs[1])     # word: index

if "glove" in path_to_bin:
    word2vec_output_file = path_to_bin + '.word2vec'
    if not os.path.exists(word2vec_output_file):
        glove2word2vec(path_to_bin, word2vec_output_file)
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
else:
    model = gensim.models.KeyedVectors.load_word2vec_format(path_to_bin, binary=True)

word_embedding = np.zeros((len(vocab), embedding_size))


for word, index in vocab.items():
    try:
        word_embed = np.asarray(model[word])
    except KeyError:
        word_embed = 2 * np.random.rand(embedding_size,) - 1.0  # [-1, 1]

    word_embedding[index] = word_embed

print("saving word embedding.")
if "glove" in path_to_bin:
    np.savetxt('../dataset/CoNLL-2003/glove_embed.txt', word_embedding, fmt='%f')
else:
    np.savetxt('../dataset/CoNLL-2003/word2vec_embed.txt', word_embedding, fmt='%f')
