#!/usr/bin/python3

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time


class ner(nn.Module):
	def __init__(self,
		word_embedding_dim, hidden_dim, type_embedding_dim,
		vocab_size, type_size,
		#word_to_ix, tag_to_ix,
		learning_rate = 0.1, minibatch_size = 1,
		max_epoch = 300,
		train_data = None, test_data = None):
		super(ner, self).__init__()
		self.word_embedding_dim = word_embedding_dim
		self.hidden_dim = hidden_dim
		self.type_embedding_dim = type_embedding_dim
		self.vocab_size = vocab_size
		self.type_size = type_size
		self.learning_rate = learning_rate
		self.minibatch_size = minibatch_size
		self.max_epoch = max_epoch
		self.train_data = train_data
		self.test_data = test_data

		#self.word_to_ix = word_to_ix
		#self.tag_to_ix = tag_to_ix


		self.word_embedding = nn.Embedding(self.vocab_size,
			self.word_embedding_dim)
		self.type_embedding = nn.Embedding(self.type_size,
			self.type_embedding_dim)

		self.encoder = nn.LSTM(self.word_embedding_dim, self.hidden_dim)
		# Temporarily use same hidden dim for decoder
		self.decoder_cell = nn.LSTMCell(self.type_embedding_dim, self.hidden_dim)

		# Transform from hidden state to scores of all possible types
		# Is this a good model?
		self.hidden2score = nn.Linear(self.hidden_dim, self.type_size)

		#self.enc_hidden, self.enc_cell = self.init_enc_hidden_cell()


	"""
	def init_enc_hidden_cell(self):
		# Initialize hidden state h_0 and cell state c_0
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (
			Variable(torch.zeros((1, self.minibatch_size, self.hidden_dim))),
			Variable(torch.zeros((1, self.minibatch_size, self.hidden_dim))))
	"""


	def encode(self, sentence, init_enc_hidden, init_enc_cell):
		sentence_emb = self.word_embedding(sentence)
		enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encoder(
			self.word_embedding(sentence).view(
			(len(sentence), self.minibatch_size, -1)),
			(init_enc_hidden, init_enc_cell))

		return enc_hidden_seq, (enc_hidden_out, enc_cell_out)


	"""
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
		type_vector_seq = Variable(torch.zeros((type_seq_len, self.type_size)), requires_grad=False)
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
	"""



	# type_index_seq is the gold truth to compared with (the one that goes
	# into the loss function)
	# type_vector_seq is the teacher forcing one-hot vector sequence
	# that is fed into the decoder
	def decode_train(self, type_index_seq, init_dec_hidden, init_dec_cell):
		type_seq_len = type_index_seq.size()[0]
		#print("type_seq_len", type_seq_len)

		dec_hidden_seq = []
		score_seq = []
		#type_vector_seq = self.make_type_vector_seq(type_index_seq)
		type_vector_seq = self.type_embedding(type_index_seq)


		#print("type_vector_seq", type_vector_seq)
		#print("type_vector_seq[0].view(1, 1, self.type_size)", type_vector_seq[0].view(1, 1, self.type_size))
		#print("type_vector_seq[0].view(1, 1, self.type_size).size()", type_vector_seq[0].view(1, 1, self.type_size).size())
		#print("self.enc_hidden", self.enc_hidden)
		#print("self.enc_cell", self.enc_cell)

		# Sentence beginning
		#print("initial", self.type_embedding(Variable(torch.LongTensor([[0, 1, 2, 3, 4]]))))
		#print("initial", self.type_embedding(Variable(torch.LongTensor([0]))))
		dec_hidden_out, dec_cell_out = self.decoder_cell(
			self.type_embedding(Variable(torch.LongTensor([0]))).view(1, self.type_embedding_dim),
			(init_dec_hidden.view(1, self.hidden_dim),
			init_dec_cell.view(1, self.hidden_dim))
			)
		dec_hidden_seq.append(dec_hidden_out)
		score = self.hidden2score(dec_hidden_out.view((1, self.hidden_dim)))
		score_seq.append(score)

		# The rest parts of the sentence
		for i in range(type_seq_len - 1):
			dec_hidden_out, dec_cell_out = self.decoder_cell(
				type_vector_seq[i].view((1, self.type_embedding_dim)),
				(dec_hidden_out.view(1, self.hidden_dim),
				dec_cell_out.view(1, self.hidden_dim))
				)
			dec_hidden_seq.append(dec_hidden_out)

			# 1 means this is only a word in the sentence
			score = self.hidden2score(dec_hidden_out.view((1, self.hidden_dim)))
			score_seq.append(score)

		return torch.cat(dec_hidden_seq), torch.cat(score_seq)

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
		def prepare_sequence(seq):
			idxs = [to_ix[w] for w in seq]
			tensor = torch.LongTensor(idxs)
			return Variable(tensor)
		"""

		"""
		train_data_var = [(
			Variable(torch.LongTensor(sen), volatile=False, requires_grad=False), 
			Variable(torch.LongTensor(typ), volatile=False, requires_grad=False))
			for sen, typ in self.train_data]
		print("train_data_var", train_data_var)
		"""


		start_time = time.time()
		for epoch in range(self.max_epoch):
		#for epoch in range(3):
			loss_sum = 0
			train_size = len(self.train_data)
			#train_size = len(train_data_var)
			for sen, typ in self.train_data:
				# Always clear the gradients before use
				self.zero_grad()

				#print("epoch", epoch)
				#print("sen", sen)
				#print("typ", typ)

				#sen_var = Variable(torch.LongTensor(sen), volatile=False, requires_grad=False)
				sen_var = Variable(torch.LongTensor(sen))
				#typ_var = Variable(torch.LongTensor(typ), volatile=False, requires_grad=False)
				typ_var = Variable(torch.LongTensor(typ))

				# Clear the hidden and cell states
				#self.hidden, self.cell = self.init_enc_hidden_cell()
				init_enc_hidden = Variable( torch.zeros((1, self.minibatch_size, self.hidden_dim)) )
				init_enc_cell = Variable( torch.zeros((1, self.minibatch_size, self.hidden_dim)) )


				
				#print("sen_var", sen_var)
				#print("typ_var", typ_var)

				#training_data_in_index = [(prepare_sequence(sen, word_to_ix), prepare_sequence(typ, tag_to_ix)) for sen, typ in training_data]


				enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encode(sen_var, init_enc_hidden, init_enc_cell)
				dec_hidden_seq, score_seq = self.decode_train(typ_var, enc_hidden_out, enc_cell_out)
				#print("score_seq", score_seq)
				#print("typ_var", typ_var)

				loss = loss_function(score_seq, typ_var)
				#print("loss", loss)
				loss_sum += loss.data.numpy()[0]
				loss.backward()
				#loss.backward(retain_graph = True)
				optimizer.step()
			avg_loss = loss_sum / train_size
			print("epoch", epoch, ", loss =", avg_loss, ", time =", time.time() - start_time)
			start_time = time.time()


	def write_log(self):
		pass


	def decode_greedy(self, seq_len, init_dec_hidden, init_dec_cell):
		# Just try to keep beam search in mind
		# Can eventually use torch.max instead
		beam_size = 1

		type_pred_seq = []
		seq_logprob = 0

		# Sentence beginning
		dec_hidden_out, dec_cell_out = self.decoder_cell(
			self.type_embedding(Variable(torch.LongTensor([0]))).view(1, self.type_embedding_dim),
			(init_dec_hidden.view(1, self.hidden_dim),
			init_dec_cell.view(1, self.hidden_dim))
			)
		score = self.hidden2score(dec_hidden_out.view((1, self.hidden_dim)))
		#print("score", score)
		logprob = nn.LogSoftmax(dim = 1)(score) + seq_logprob
		#print("logprob", logprob)
		topk_logprob, topk_type = torch.topk(logprob, beam_size)
		#print("topk_logprob", topk_logprob)
		#print("topk_type", topk_type)
		seq_logprob += topk_logprob[0]
		#print("seq_logprob", seq_logprob)
		type_pred_seq.append(topk_type[0])
		#print("type_pred_seq", type_pred_seq)

		# The rest parts of the sentence
		for i in range(seq_len - 1):
			#print("type_pred_seq[-1]", type_pred_seq[-1])
			dec_hidden_out, dec_cell_out = self.decoder_cell(
				self.type_embedding(type_pred_seq[-1]).view(1, self.type_embedding_dim),
				(dec_hidden_out.view(1, self.hidden_dim),
				dec_cell_out.view(1, self.hidden_dim))
				)
			score = self.hidden2score(dec_hidden_out.view((1, self.hidden_dim)))
			logprob = nn.LogSoftmax(dim = 1)(score) + seq_logprob
			topk_logprob, topk_type = torch.topk(logprob, beam_size)
			seq_logprob += topk_logprob[0]
			type_pred_seq.append(topk_type[0])

		return type_pred_seq


	"""
	def decode_beam(self, seq_len, init_dec_hidden, init_dec_cell, beam_size):
		type_pred_seq = []
		seq_logprob = 0
		#coming_from_beam = []

		# Sentence beginning
		dec_hidden_out, dec_cell_out = self.decoder_cell(
			self.type_embedding(Variable(torch.LongTensor([0]))).view(1, self.type_embedding_dim),
			(init_dec_hidden.view(1, self.hidden_dim),
			init_dec_cell.view(1, self.hidden_dim))
			)
		score = self.hidden2score(dec_hidden_out.view((1, self.hidden_dim)))
		print("score", score)
		logprob = nn.LogSoftmax(dim = 1)(score) + seq_logprob
		print("logprob", logprob)
		topk_logprob, topk_type = torch.topk(logprob, beam_size)
		print("topk_logprob", topk_logprob)
		print("topk_type", topk_type)
		seq_logprob = torch.cat([seq_logprob + topk_logprob[0, i] for i in range(topk_logprob.size()[1])])
		print("seq_logprob", seq_logprob)
		type_pred_seq.append([[0, topk_type[0, i]] for i in range(topk_type.size()[1])])
		print("type_pred_seq", type_pred_seq)

		# The rest parts of the sentence
		for i in range(seq_len - 1):
			print("i", i)
			last_type_pred_list = type_pred_seq[-1]
			print("last_type_pred_list", last_type_pred_list)
			for b, [coming_beam, typ] in enumerate(last_type_pred_list):
				print("b", b)
				print("coming_beam", coming_beam)
				print("typ", typ)
				dec_hidden_out, dec_cell_out = self.decoder_cell(
					self.type_embedding(type_pred_seq[-1]).view(1, self.type_embedding_dim),
					(dec_hidden_out.view(1, self.hidden_dim),
					dec_cell_out.view(1, self.hidden_dim))
					)

			#print("type_pred_seq[-1]", type_pred_seq[-1])
			dec_hidden_out, dec_cell_out = self.decoder_cell(
				self.type_embedding(type_pred_seq[-1]).view(1, self.type_embedding_dim),
				(dec_hidden_out.view(1, self.hidden_dim),
				dec_cell_out.view(1, self.hidden_dim))
				)
			score = self.hidden2score(dec_hidden_out.view((1, self.hidden_dim)))
			logprob = nn.LogSoftmax(dim = 1)(score) + seq_logprob
			topk_logprob, topk_type = torch.topk(logprob, beam_size)
			seq_logprob += topk_logprob[0]
			type_pred_seq.append(topk_type[0])

		return type_pred_seq
	"""


	def test(self):
		for sen, typ in self.test_data:
			# Always clear the gradients before use
			self.zero_grad()
			sen_var = Variable(torch.LongTensor(sen))
			typ_var = Variable(torch.LongTensor(typ))
			init_enc_hidden = Variable( torch.zeros((1, self.minibatch_size, self.hidden_dim)) )
			init_enc_cell = Variable( torch.zeros((1, self.minibatch_size, self.hidden_dim)) )

			enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encode(sen_var, init_enc_hidden, init_enc_cell)

			type_pred_seq = self.decode_greedy(len(sen), enc_hidden_out, enc_cell_out)
			#beam_size = 3
			#type_pred_seq = self.decode_beam(len(sen), enc_hidden_out, enc_cell_out, beam_size)

			print("sen =", sen)
			print("type pred =", type_pred_seq)












