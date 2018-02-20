# Reference:
# http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class ner(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, cell_dim, vocab_size,
		tagset_size, minibatch_size = 1):
		super(ner, self).__init__()
		self.hidden_dim = hidden_dim
		self.cell_dim = cell_dim
		slef.minibatch_size = minibatch_size
		self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim)

		# The linear layer that maps from hidden state space to type space
		self.hidden2type = nn.Linear(hidden_dim, type_set_size)
		
		# Initialize hidden state h_0 and cell state c_0
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		self.hidden = Variable(torch.zeros(1, self.minibatch_size,
			self.hidden_dim))
		self.cell = Variable(torch.zeros(1, self.minibatch_size,
			self.hidden_dim))


	def forward(self, sentence):
		sentence_emb = self.word_embedding(sentence)
		hidden_seq, (self.hidden, self.cell) = self.lstm(
			self.word_embedding(sentence).view(
			len(sentence), self.minibatch_size, -1),
			(self.hidden, self.cell))
		type_score = F.log_softmax(self.hidden2type(
			hidden_seq.view(len(sentence), -1)), dim = 1)

		return type_score











