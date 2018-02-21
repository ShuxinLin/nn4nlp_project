#!/usr/bin/python3

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


class ner(nn.Module):
	def __init__(self,
		embedding_dim, hidden_dim,
		vocab_size, type_size,
		#word_to_ix, tag_to_ix,
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

		#self.word_to_ix = word_to_ix
		#self.tag_to_ix = tag_to_ix


		self.word_embedding = nn.Embedding(self.vocab_size,
			self.embedding_dim)

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
			Variable(torch.zeros((1, self.minibatch_size, self.hidden_dim))),
			Variable(torch.zeros((1, self.minibatch_size, self.hidden_dim))))


	def encode(self, sentence):
		sentence_emb = self.word_embedding(sentence)
		enc_hidden_seq, (self.enc_hidden, self.enc_cell) = self.encoder(
			self.word_embedding(sentence).view(
			(len(sentence), self.minibatch_size, -1)),
			(self.enc_hidden, self.enc_cell))


	# Define a function to output a one-hot vector of size type_size
	# given the index of the type
	# Assume type_size has included <s> as the beginning of a sentence
	# and <p> as padding after the end of a sentence
	#
	# Note that: 0 is always for sentence beginning, 1 is always for padding.
	# This is for training. It will truncate the last column of type_train
	# to make room for the sentence begin.
	def make_type_vector_seq(self, type_index_seq):
		type_seq_len = type_index_seq.size()[0]
		#print("here", type_seq_len)

		# For sentence beginning, (Batchsize, 1)
		#type_index_seq = torch.cat(
		#	(Variable(torch.LongTensor(np.zeros((1, 1)))),
		#	type_index_seq[0:-1].view((1, type_seq_len - 1))),
		#	dim = 1)

		# type_seq_len, type_size
		#type_vector_seq = np.zeros((type_seq_len, self.type_size))
		type_vector_seq = Variable(torch.zeros((type_seq_len, self.type_size)))
		type_index_seq_np = type_index_seq.data.numpy().astype(int)
		#print("type_index_seq_np", type_index_seq_np)
		# Mark the sentence beginning
		type_vector_seq[0, 0] = 1
		# Truncate the last column
		for r, col in enumerate(type_index_seq_np[0:-1]):
			type_vector_seq[r + 1, col] = 1
		#type_vector_seq = Variable(torch.DoubleTensor(torch.from_numpy(type_vector_seq)))
		#print("type_vector_seq", type_vector_seq)

		return type_vector_seq


	# type_index_seq is the gold truth to compared with (the one that goes
	# into the loss function)
	# type_vector_seq is the teacher forcing one-hot vector sequence
	# that is fed into the decoder
	def decode_train(self, type_index_seq):
		type_seq_len = type_index_seq.size()[0]
		#print("type_seq_len", type_seq_len)

		self.dec_hidden_seq = []
		score_seq = []
		type_vector_seq = self.make_type_vector_seq(type_index_seq)
		#print("type_vector_seq", type_vector_seq)
		#print("type_vector_seq[0].view(1, 1, self.type_size)", type_vector_seq[0].view(1, 1, self.type_size))
		#print("type_vector_seq[0].view(1, 1, self.type_size).size()", type_vector_seq[0].view(1, 1, self.type_size).size())
		#print("self.enc_hidden", self.enc_hidden)
		#print("self.enc_cell", self.enc_cell)
		for i in range(type_seq_len):
			self.dec_hidden, self.dec_cell = self.decoder_cell(type_vector_seq[i].view((1, self.type_size)), (self.enc_hidden.view(1, self.hidden_dim), self.enc_cell.view(1, self.hidden_dim)))
			self.dec_hidden_seq.append(self.dec_hidden)

			# 1 means this is only a word in the sentence
			score = self.hidden2score(self.dec_hidden.view((1, self.hidden_dim)))

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

		"""
		def prepare_sequence(seq, to_ix):
			idxs = [to_ix[w] for w in seq]
			tensor = torch.LongTensor(idxs)
			return Variable(tensor)
		"""


		for epoch in range(self.max_epoch):
		#for epoch in range(3):
			loss_sum = 0
			train_size = len(self.train_data)
			for sen, typ in self.train_data:
				# Always clear the gradients before use
				self.zero_grad()

				# Clear the hidden and cell states
				self.hidden, self.cell = self.init_enc_hidden_cell()

				#print("epoch", epoch)
				#print("sen", sen)
				#print("typ", typ)

				#training_data_in_index = [(prepare_sequence(sen, word_to_ix), prepare_sequence(typ, tag_to_ix)) for sen, typ in training_data]


				self.encode(sen)
				score_seq = self.decode_train(typ)
				#print(type_score)

				#loss_function = nn.CrossEntropyLoss()


				loss = loss_function(score_seq, typ)
				loss_sum += loss.data.numpy()[0]
				loss.backward(retain_graph = True)
				optimizer.step()
			avg_loss = loss_sum / train_size
			print("epoch", epoch, ", loss =", avg_loss)


	def write_log(self):
		pass






