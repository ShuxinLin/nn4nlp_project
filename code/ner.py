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
		word_embedding_dim, hidden_dim, label_embedding_dim,
		vocab_size, label_size,
		#word_to_ix, tag_to_ix,
		learning_rate = 0.1, minibatch_size = 1,
		max_epoch = 300,
		train_X = None, train_Y = None,
		test_X = None, test_Y = None):
		super(ner, self).__init__()
		self.word_embedding_dim = word_embedding_dim
		self.hidden_dim = hidden_dim
		self.label_embedding_dim = label_embedding_dim
		self.vocab_size = vocab_size
		self.label_size = label_size
		self.learning_rate = learning_rate
		self.minibatch_size = minibatch_size
		self.max_epoch = max_epoch
		self.train_X = train_X
		self.train_Y = train_Y
		self.test_X = test_X
		self.test_Y = test_Y

		#self.word_to_ix = word_to_ix
		#self.tag_to_ix = tag_to_ix


		self.word_embedding = nn.Embedding(self.vocab_size,
			self.word_embedding_dim)
		self.label_embedding = nn.Embedding(self.label_size,
			self.label_embedding_dim)

		self.encoder = nn.LSTM(self.word_embedding_dim, self.hidden_dim)
		# Temporarily use same hidden dim for decoder
		self.decoder_cell = nn.LSTMCell(self.label_embedding_dim, self.hidden_dim)

		# Transform from hidden state to scores of all possible labels
		# Is this a good model?
		self.hidden2score = nn.Linear(self.hidden_dim, self.label_size)

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
		# sentence shape is (batch_size, sentence_length)
		sentence_emb = self.word_embedding(sentence)
		sentence_len = sentence.size()[1]
		# enc_hidden_seq shape is (seq_len, batch_size, hidden_dim * num_directions)
		# num_directions = 2 for bi-directional LSTM
		#
		# enc_hidden_out shape is (num_layers * num_directions, batch_size, hidden_dim)
		# We use 1-layer here
		enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encoder(
			sentence_emb.view(
			(sentence_len, self.minibatch_size, self.word_embedding_dim)),
			(init_enc_hidden, init_enc_cell))
		#print("enc_hidden_seq.size()", enc_hidden_seq.size())
		#print("enc_hidden_out.size()", enc_hidden_out.size())

		return enc_hidden_seq, (enc_hidden_out, enc_cell_out)


	"""
	# Define a function to output a one-hot vector of size label_size
	# given the index of the label
	# Assume label_size has included <s> as the beginning of a sentence
	# and <p> as padding after the end of a sentence
	#
	# Note that: 0 is always for sentence beginning, 1 is always for padding.
	# This is for training. It will truncate the last column of label_train
	# to make room for the sentence begin.
	def make_label_vector_seq(self, label_index_seq):
		label_seq_len = label_index_seq.size()[0]
		#print("here", label_seq_len)

		# For sentence beginning, (Batchsize, 1)
		#label_index_seq = torch.cat(
		#	(Variable(torch.LongTensor(np.zeros((1, 1)))),
		#	label_index_seq[0:-1].view((1, label_seq_len - 1))),
		#	dim = 1)

		# label_seq_len, label_size
		#label_vector_seq = np.zeros((label_seq_len, self.label_size))
		label_vector_seq = Variable(torch.zeros((label_seq_len, self.label_size)), requires_grad=False)
		label_index_seq_np = label_index_seq.data.numpy().aslabel(int)
		#print("label_index_seq_np", label_index_seq_np)
		# Mark the sentence beginning
		label_vector_seq[0, 0] = 1
		# Truncate the last column
		for r, col in enumerate(label_index_seq_np[0:-1]):
			label_vector_seq[r + 1, col] = 1
		#label_vector_seq = Variable(torch.DoubleTensor(torch.from_numpy(label_vector_seq)))
		#print("label_vector_seq", label_vector_seq)

		return label_vector_seq
	"""



	# label_index_seq is the gold truth to compared with (the one that goes
	# into the loss function)
	# label_vector_seq is the teacher forcing one-hot vector sequence
	# that is fed into the decoder
	def decode_train(self, label_seq, init_dec_hidden, init_dec_cell):
		label_seq_len = label_seq.size()[1]
		#print("label_seq_len", label_seq_len)

		dec_hidden_seq = []
		score_seq = []
		#label_vector_seq = self.make_label_vector_seq(label_index_seq)
		#print("label_seq", label_seq)
		#label_emb_seq = self.label_embedding(label_seq)
		#print("label_emb_seq", label_emb_seq)
		label_emb_seq = self.label_embedding(label_seq).permute(1, 0, 2)
		#print("label_emb_seq =>", label_emb_seq)


		#print("label_vector_seq", label_vector_seq)
		#print("label_vector_seq[0].view(1, 1, self.label_size)", label_vector_seq[0].view(1, 1, self.label_size))
		#print("label_vector_seq[0].view(1, 1, self.label_size).size()", label_vector_seq[0].view(1, 1, self.label_size).size())
		#print("self.enc_hidden", self.enc_hidden)
		#print("self.enc_cell", self.enc_cell)

		# Sentence beginning
		#print("initial", self.label_embedding(Variable(torch.LongTensor([[0, 1, 2, 3, 4]]))))
		#print("initial", self.label_embedding(Variable(torch.LongTensor([0]))))
		#init_label_emb = self.label_embedding(Variable(torch.LongTensor([[0], [0]])))
		#init_label_emb = self.label_embedding(Variable(torch.LongTensor(torch.zeros((self.minibatch_size, 1)))))

		LABEL_BEGIN_INDEX = 1
		init_label_emb = self.label_embedding(Variable(torch.LongTensor(self.minibatch_size, 1).zero_() + LABEL_BEGIN_INDEX)).view(self.minibatch_size, self.label_embedding_dim)
		#print("init_label_emb", init_label_emb)
		#print("init_label_emb.view =>", init_label_emb.view(self.minibatch_size, self.label_embedding_dim))
		dec_hidden_out, dec_cell_out = self.decoder_cell(
			init_label_emb,	(init_dec_hidden, init_dec_cell))
		#print("dec_hidden_out", dec_hidden_out)
		dec_hidden_seq.append(dec_hidden_out)
		#print("i 0, dec_hidden_seq", dec_hidden_seq)
		score = self.hidden2score(dec_hidden_out)
		#print("score", score)
		score_seq.append(score)

		# The rest parts of the sentence
		for i in range(label_seq_len - 1):
			dec_hidden_out, dec_cell_out = self.decoder_cell(
				label_emb_seq[i], (dec_hidden_out, dec_cell_out))
			dec_hidden_seq.append(dec_hidden_out)
			score = self.hidden2score(dec_hidden_out)
			#print("i", i, " dec_hidden_seq", dec_hidden_seq)
			score_seq.append(score)

			# 1 means this is only a word in the sentence
			#score = self.hidden2score(dec_hidden_out.view((1, self.hidden_dim)))
			#score_seq.append(score)

		# It could make sense to reshape decoder hidden output
		# But currently we don't use this output in later stage
		dec_hidden_seq = torch.cat(dec_hidden_seq, dim = 0).view(label_seq_len, self.minibatch_size, self.hidden_dim)
		# For score_seq, actually don't need to reshape!
		# It happens that directly concatenate along dim = 0 gives you a convenient shape (batch_size * seq_len, label_size) for later cross entropy loss
		score_seq = torch.cat(score_seq, dim = 0)
		#score_seq = torch.cat(score_seq, dim = 0).view(label_seq_len, self.minibatch_size, self.label_size)

		return dec_hidden_seq, score_seq

		# Then for loop to pass each self.dec_hidden into a Linear layer
		# to obtain the scores for all possible labels,


		# The following can be in other function
		# then take softmax to get normalized probability
		# Then compute the cross entropy between our prob and true one-hot label label
		# to be our loss function


	def train(self):
		# Will manually average over (sentence_len * instance_num)
		loss_function = nn.CrossEntropyLoss(size_average = False)
		# Note that here we called nn.Module.parameters()
		optimizer = optim.SGD(self.parameters(), lr = self.learning_rate)

		instance_num = 0
		for batch in self.train_X:
			instance_num += len(batch)

		print("instance_num", instance_num)

		start_time = time.time()
		for epoch in range(self.max_epoch):
			loss_sum = 0
			#train_size = len(self.train_data)
			batch_num = len(self.train_X)

			for batch_idx in range(batch_num):
				sen = self.train_X[batch_idx]
				label = self.train_Y[batch_idx]

				#current_batch_size = len(sen)
				current_sen_len = len(sen[0])

				# Always clear the gradients before use
				self.zero_grad()

				#print("epoch", epoch)
				#print("sen", sen)
				#print("label", label)

				#sen_var = Variable(torch.LongTensor(sen), volatile=False, requires_grad=False)
				sen_var = Variable(torch.LongTensor(sen))
				#print("sen_var", sen_var)
				#label_var = Variable(torch.LongTensor(label), volatile=False, requires_grad=False)
				label_var = Variable(torch.LongTensor(label))

				# Initialize the hidden and cell states
				# The axes semantics are (num_layers, minibatch_size, hidden_dim)
				#self.hidden, self.cell = self.init_enc_hidden_cell()
				init_enc_hidden = Variable( torch.zeros((1, self.minibatch_size, self.hidden_dim)) )
				init_enc_cell = Variable( torch.zeros((1, self.minibatch_size, self.hidden_dim)) )
				#print("init_enc_hidden", init_enc_hidden)

				
				#print("sen_var", sen_var)
				#print("label_var", label_var)

				#training_data_in_index = [(prepare_sequence(sen, word_to_ix), prepare_sequence(label, tag_to_ix)) for sen, label in training_data]


				enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encode(sen_var, init_enc_hidden, init_enc_cell)
				#print("enc_hidden_out", enc_hidden_out)

				#init_dec_hidden = enc_hidden_out.view(self.minibatch_size, self.hidden_dim)
				init_dec_hidden = enc_hidden_out[0]
				init_dec_cell = enc_cell_out[0]
				#print("init_dec_hidden", init_dec_hidden)

				dec_hidden_seq, score_seq = self.decode_train(label_var,
					init_dec_hidden, init_dec_cell)
				#print("in train:")
				#print("dec_hidden_seq", dec_hidden_seq)
				#print("score_seq", score_seq)

				#print("label_var", label_var)

				label_var_for_loss = label_var.permute(1, 0).contiguous().view(-1)
				#print("label_var_for_loss", label_var_for_loss)

				loss = loss_function(score_seq, label_var_for_loss)
				#print("loss", loss)
				loss_sum += loss.data.numpy()[0] / current_sen_len
				loss.backward()
				#loss.backward(retain_graph = True)
				optimizer.step()
			avg_loss = loss_sum / instance_num
			print("epoch", epoch, ", loss =", avg_loss, ", time =", time.time() - start_time)
			start_time = time.time()


	def write_log(self):
		pass


	def decode_greedy(self, seq_len, init_dec_hidden, init_dec_cell):
		# Just try to keep beam search in mind
		# Can eventually use torch.max instead
		beam_size = 1

		label_pred_seq = []
		seq_logprob = Variable(torch.FloatTensor(self.minibatch_size, 1).zero_())

		# Sentence beginning
		LABEL_BEGIN_INDEX = 1
		init_label_emb = self.label_embedding(Variable(torch.LongTensor(self.minibatch_size, 1).zero_() + LABEL_BEGIN_INDEX)).view(self.minibatch_size, self.label_embedding_dim)

		dec_hidden_out, dec_cell_out = self.decoder_cell(
			init_label_emb,	(init_dec_hidden, init_dec_cell))

		score = self.hidden2score(dec_hidden_out)
		#print("score", score)
		#print("nn.log...", nn.LogSoftmax(dim = 1)(score))

		logprob = nn.LogSoftmax(dim = 1)(score) + seq_logprob
		#print("logprob", logprob)

		topk_logprob, topk_label = torch.topk(logprob, beam_size, dim = 1)
		#print("topk_logprob", topk_logprob)
		#print("topk_label", topk_label)

		# We have already added seq_logprob
		# So simply let topk_logprob be the updated accumulated seq_logprob
		seq_logprob = topk_logprob
		#print("seq_logprob", seq_logprob)

		label_pred_seq.append(topk_label)
		#print("label_pred_seq", label_pred_seq)

		#print("here", self.label_embedding(label_pred_seq[-1]))
		#print("here", self.label_embedding(label_pred_seq[-1]).view(self.minibatch_size, self.label_embedding_dim))


		# The rest parts of the sentence
		for i in range(seq_len - 1):
			#print("--------------- i", i)
			prev_pred_label_emb = self.label_embedding(label_pred_seq[-1]) \
				.view(self.minibatch_size, self.label_embedding_dim)
			dec_hidden_out, dec_cell_out = self.decoder_cell(
				prev_pred_label_emb, (init_dec_hidden, init_dec_cell))
			score = self.hidden2score(dec_hidden_out)
			logprob = nn.LogSoftmax(dim = 1)(score) + seq_logprob
			#print("logprob", logprob)
			topk_logprob, topk_label = torch.topk(logprob, beam_size, dim = 1)
			#print("topk_logprob", topk_logprob)
			#print("topk_label", topk_label)
			seq_logprob = topk_logprob
			#print("seq_logprob", seq_logprob)
			label_pred_seq.append(topk_label)
			#print("label_pred_seq", label_pred_seq)

		#print("==========")
		#print("label_pred_seq", label_pred_seq)
		return label_pred_seq


	"""
	def decode_beam(self, seq_len, init_dec_hidden, init_dec_cell, beam_size):
		label_pred_seq = []
		seq_logprob = 0
		#coming_from_beam = []

		# Sentence beginning
		dec_hidden_out, dec_cell_out = self.decoder_cell(
			self.label_embedding(Variable(torch.LongTensor([0]))).view(1, self.label_embedding_dim),
			(init_dec_hidden.view(1, self.hidden_dim),
			init_dec_cell.view(1, self.hidden_dim))
			)
		score = self.hidden2score(dec_hidden_out.view((1, self.hidden_dim)))
		print("score", score)
		logprob = nn.LogSoftmax(dim = 1)(score) + seq_logprob
		print("logprob", logprob)
		topk_logprob, topk_label = torch.topk(logprob, beam_size)
		print("topk_logprob", topk_logprob)
		print("topk_label", topk_label)
		seq_logprob = torch.cat([seq_logprob + topk_logprob[0, i] for i in range(topk_logprob.size()[1])])
		print("seq_logprob", seq_logprob)
		label_pred_seq.append([[0, topk_label[0, i]] for i in range(topk_label.size()[1])])
		print("label_pred_seq", label_pred_seq)

		# The rest parts of the sentence
		for i in range(seq_len - 1):
			print("i", i)
			last_label_pred_list = label_pred_seq[-1]
			print("last_label_pred_list", last_label_pred_list)
			for b, [coming_beam, label] in enumerate(last_label_pred_list):
				print("b", b)
				print("coming_beam", coming_beam)
				print("label", label)
				dec_hidden_out, dec_cell_out = self.decoder_cell(
					self.label_embedding(label_pred_seq[-1]).view(1, self.label_embedding_dim),
					(dec_hidden_out.view(1, self.hidden_dim),
					dec_cell_out.view(1, self.hidden_dim))
					)

			#print("label_pred_seq[-1]", label_pred_seq[-1])
			dec_hidden_out, dec_cell_out = self.decoder_cell(
				self.label_embedding(label_pred_seq[-1]).view(1, self.label_embedding_dim),
				(dec_hidden_out.view(1, self.hidden_dim),
				dec_cell_out.view(1, self.hidden_dim))
				)
			score = self.hidden2score(dec_hidden_out.view((1, self.hidden_dim)))
			logprob = nn.LogSoftmax(dim = 1)(score) + seq_logprob
			topk_logprob, topk_label = torch.topk(logprob, beam_size)
			seq_logprob += topk_logprob[0]
			label_pred_seq.append(topk_label[0])

		return label_pred_seq
	"""


	def test(self):
		batch_num = len(self.test_X)
		for batch_idx in range(batch_num):
			sen = self.test_X[batch_idx]
			label = self.test_Y[batch_idx]
			current_sen_len = len(sen[0])

			# Always clear the gradients before use
			self.zero_grad()
			sen_var = Variable(torch.LongTensor(sen))
			label_var = Variable(torch.LongTensor(label))

			init_enc_hidden = Variable( torch.zeros((1, self.minibatch_size, self.hidden_dim)) )
			init_enc_cell = Variable( torch.zeros((1, self.minibatch_size, self.hidden_dim)) )

			enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encode(sen_var, init_enc_hidden, init_enc_cell)

			init_dec_hidden = enc_hidden_out[0]
			init_dec_cell = enc_cell_out[0]

			label_pred_seq = self.decode_greedy(current_sen_len, init_dec_hidden, init_dec_cell)
			#beam_size = 3
			#label_pred_seq = self.decode_beam(len(sen), enc_hidden_out, enc_cell_out, beam_size)

			print("sen =", sen)
			print("label pred =", label_pred_seq)
			print("label", label)











