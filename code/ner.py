import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ner(nn.Module):
	def __init__(self,
		embedding_dim, hidden_dim, cell_dim,
		vocab_size, type_size,
		learning_rate = 0.1, minibatch_size = 1,
		max_epoch = 300,
		train_data = None):
		super(ner, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.cell_dim = cell_dim
		self.vocab_size = vocab_size
		self.type_size = type_size
		self.learning_rate = learning_rate
		self.minibatch_size = minibatch_size
		self.max_epoch = max_epoch
		self.train_data = train_data

		self.word_embedding = nn.Embedding(self.vocab_size,
			self.embedding_dim)
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)

		# The linear layer that maps from hidden state space to type space
		self.hidden2type = nn.Linear(self.hidden_dim, self.type_size)

		self.hidden, self.cell = self.init_hidden_cell()


	def init_hidden_cell(self):
		# Initialize hidden state h_0 and cell state c_0
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (
			Variable(torch.zeros(1, self.minibatch_size, self.hidden_dim)),
			Variable(torch.zeros(1, self.minibatch_size, self.hidden_dim)))


	def forward(self, sentence):
		sentence_emb = self.word_embedding(sentence)
		hidden_seq, (self.hidden, self.cell) = self.lstm(
			self.word_embedding(sentence).view(
			len(sentence), self.minibatch_size, -1),
			(self.hidden, self.cell))
		type_score = F.log_softmax(self.hidden2type(
			hidden_seq.view(len(sentence), -1)), dim = 1)

		return type_score


	def train(self):
		loss_function = nn.NLLLoss()
		# Note that here we called nn.Module.parameters()
		optimizer = optim.SGD(self.parameters(), lr = self.learning_rate)

		for epoch in range(self.max_epoch):
			for sen, typ in self.train_data:
				# Always clear the gradients before use
				self.zero_grad()

				# Clear the hidden and cell states
				self.hidden, self.cell = self.init_hidden_cell()

				type_score = self.forward(sen)
				loss = loss_function(type_score, typ)
				loss.backward()
				optimizer.step()


#	def write_log






