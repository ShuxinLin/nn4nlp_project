#!/usr/bin/python3

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ner(nn.Module):
	def __init__(self,
		embedding_dim, hidden_dim,
		vocab_size, type_size,
		learning_rate = 0.1, minibatch_size = 1,
		max_epoch = 300,
		train_data = None):
		super(ner, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.type_size = type_size
		self.learning_rate = learning_rate
		self.minibatch_size = minibatch_size
		self.max_epoch = max_epoch
		self.train_data = train_data

		#self.word_embedding = nn.Embedding(self.vocab_size,
		#	self.embedding_dim)
		slef.word_embedding = onmt.modules.Embeddings(self.embedding_dim, self.vocab_size, word_padding_idx=src_padding)

		self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim)
		# Temporarily use same hidden dim for decoder
		self.decoder_cell = nn.LSTMCell(self.type_size, self.hidden_dim)

		# Transform from hidden state to scores of all possible types
		# Is this a good model?
		self.hidden2score = nn.Linear(self.hidden_dim, self.type_size)

		self.enc_hidden, self.enc_cell = self.init_enc_hidden_cell()


	def init_enc_hidden_cell(self):
		# Initialize hidden state h_0 and cell state c_0
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (
			Variable(torch.zeros(1, self.minibatch_size, self.hidden_dim)),
			Variable(torch.zeros(1, self.minibatch_size, self.hidden_dim)))


	def encode(self, sentence):
		sentence_emb = self.word_embedding(sentence)
		enc_hidden_seq, (self.enc_hidden, self.enc_cell) = self.encoder(
			self.word_embedding(sentence).view(
			len(sentence), self.minibatch_size, -1),
			(self.enc_hidden, self.enc_cell))


	# Define a function to output a one-hot vector of size type_size
	# given the index of the type
	# Assume type_size has included <s> as the beginning of a sentence
	# and <p> as padding after the end of a sentence


	def decode_train(self, type_train):
		type_seq_len = type_train.size()
		print(type_seq_len)

		self.dec_hidden_seq = []
		score_seq = []
		# WE ARE HERE...
		#init_type_train = Variable(torch.zeros())
		for i in range(type_seq_len):
			self.dec_hidden, self.dec_cell = self.decoder_cell(type_train[i], (self.enc_hidden, self.enc_cell))
			self.dec_hidden_seq.append(self.dec_hidden)

			# 1 means this is only a word in the sentence
			score = self.hidden2score(self.dec_hidden.view(1, self.hidden_dim))

			score_seq.append(score)

		self.dec_hidden_seq = torch.cat(self.dec_hidden_seq)

		return torch.cat(score_seq)

		# Then for loop to pass each self.dec_hidden into a Linear layer
		# to obtain the scores for all possible types,


		# The following can be in other function
		# then take softmax to get normalized probability
		# Then compute the cross entropy between our prob and true one-hot type label
		# to be our loss function


	def train(self):
		loss_function = nn.CrossEntropyLoss()
		# Note that here we called nn.Module.parameters()
		optimizer = optim.SGD(self.parameters(), lr = self.learning_rate)

		for epoch in range(self.max_epoch):
			loss_sum = 0
			train_size = len(self.train_data)
			for sen, typ in self.train_data:
				# Always clear the gradients before use
				self.zero_grad()

				# Clear the hidden and cell states
				self.hidden, self.cell = self.init_hidden_cell()

				self.encode()

				type_score = self.forward(sen)
				#print(type_score)
				loss = loss_function(type_score, typ)
				loss_sum += loss.data.numpy()[0]
				loss.backward()
				optimizer.step()
			avg_loss = loss_sum / train_size
			print("epoch", epoch, ", loss =", avg_loss)


	def write_log(self):
		pass






