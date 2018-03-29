import gensim
import numpy as np

vocab = {}
embedding_size = 300
path_to_bin = '/Users/susie/Documents/GoogleNews-vectors-negative300.bin'

with open("../dataset/CoNLL-2003/vocab_dict") as f:
    for line in f:
        pairs = line.strip().split()
        vocab[pairs[0]] = int(pairs[1])     # word: index
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_bin, binary=True)
word_embedding = np.zeros((len(vocab), embedding_size))

for word, index in vocab.items():
    try:
        word_embed = np.asarray(model[word])
    except KeyError:
        word_embed = 2 * np.random.rand(embedding_size,) - 1.0  # [-1, 1]

    word_embedding[index] = word_embed

print("saving word embedding for word2vec.")
np.savetxt('../dataset/word2vec_embed.txt', word_embedding, fmt='%f')