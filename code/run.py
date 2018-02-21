#!/usr/bin/python3

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import onmt
import onmt.io
import onmt.modules

from ner import ner


def main():
	# Temporarily generate data by hand for test purpose
	training_data = [
		("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
		("Everybody read that book <p>".split(), ["NN", "V", "DET", "NN", "<p>"])
	]
	word_to_ix = {}
	for sent, tags in training_data:
		for word in sent:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)
	print(word_to_ix)

	tag_to_ix = {"<s>": 0, "<p>": 1, "DET": 2, "NN": 3, "V": 4}
	print(tag_to_ix)

	def prepare_sequence(seq, to_ix):
		idxs = [to_ix[w] for w in seq]
		tensor = torch.LongTensor(idxs)
		return Variable(tensor)

	training_data_in_index = [(prepare_sequence(sen, word_to_ix), prepare_sequence(typ, tag_to_ix)) for sen, typ in training_data]
	"""
	for idx, (sen, typ) in enumerate(training_data_in_index):
		print("data", idx)
		print(sen)
		print(typ)
	"""


	######################################
	embedding_dim = 6
	hidden_dim = 6

	machine = ner(embedding_dim, hidden_dim, len(word_to_ix), len(tag_to_ix), learning_rate = 0.1, minibatch_size = 1, max_epoch = 300, train_data = training_data_in_index)

	machine.train()








if __name__ == "__main__":
	main()
