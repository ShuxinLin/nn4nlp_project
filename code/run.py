#!/usr/bin/python3

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ner import ner


def main():
	# Temporarily generate data by hand for test purpose
	training_data = [
		("The dog ate the apple".split(" "), ["DET", "NN", "V", "DET", "NN"]),
		("Everybody read that book <p>".split(" "), ["NN", "V", "DET", "NN", "<p>"])
	]
	word_to_ix = {}
	for sent, tags in training_data:
		for word in sent:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)
	print(word_to_ix)

	type_to_ix = {"<s>": 0, "<p>": 1, "DET": 2, "NN": 3, "V": 4}
	print(type_to_ix)

	training_data_in_index = [([word_to_ix[w] for w in sen], [type_to_ix[t] for t in typ]) for sen, typ in training_data]
	
	for idx, (sen, typ) in enumerate(training_data_in_index):
		print("data", idx)
		print(sen)
		print(typ)
	


	######################################
	word_embedding_dim = 6
	hidden_dim = 6
	type_embedding_dim = 6

	machine = ner(word_embedding_dim, hidden_dim, type_embedding_dim, len(word_to_ix), len(type_to_ix), learning_rate = 0.1, minibatch_size = 1, max_epoch = 300, train_data = training_data_in_index, test_data = training_data_in_index)
	#machine = ner(embedding_dim, hidden_dim, len(word_to_ix), len(tag_to_ix), word_to_ix, tag_to_ix, learning_rate = 0.1, minibatch_size = 1, max_epoch = 300, train_data = training_data_in_index)

	machine.train()

	machine.test()







if __name__ == "__main__":
	main()
