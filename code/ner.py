#!/usr/bin/python3

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import itertools

from attention import Attention
from preprocessor import *

class ner(nn.Module):
  def __init__(self,
               word_embedding_dim, hidden_dim, label_embedding_dim,
               vocab_size, label_size,
               learning_rate=0.1, minibatch_size=1,
               max_epoch=300,
               train_X=None, train_Y=None,
               test_X=None, test_Y=None,
               attention="fixed",
               gpu=False,
               pretrained='glove'):

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

    # For now we hard code the index of "<BEG>"
    self.BEG_INDEX = 1

    self.gpu = gpu

    # Attention
    if attention:
      self.attention = Attention(attention, self.hidden_dim, self.gpu)
    # Otherwise no attention
    else:
      self.attention = None

    self.word_embedding = nn.Embedding(self.vocab_size,
                                       self.word_embedding_dim)

    word_embedding_np = np.loadtxt('../dataset/CoNLL-2003/' + pretrained + '_embed.txt', dtype=float)    # load pretrained model: word2vec/glove
    self.word_embedding.weight.data.copy_(torch.from_numpy(word_embedding_np))

    self.label_embedding = nn.Embedding(self.label_size,
                                        self.label_embedding_dim)

    self.encoder = nn.LSTM(input_size=self.word_embedding_dim,
                           hidden_size=self.hidden_dim,
                           bidirectional=True)

    # The semantics of enc_hidden_out is (num_layers * num_directions,
    # batch, hidden_size), and it is "tensor containing the hidden state
    # for t = seq_len".
    #
    # Here we use a linear layer to transform the two-directions of the dec_hidden_out's into a single hid_dim vector, to use as the input of the decoder
    self.enc2dec_hidden = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
    self.enc2dec_cell = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

    # Temporarily use same hidden dim for decoder
    self.decoder_cell = nn.LSTMCell(self.label_embedding_dim,
                                    self.hidden_dim)

    # Transform from hidden state to scores of all possible labels
    self.hidden2score = nn.Linear(self.hidden_dim, self.label_size)

  def encode(self, sentence, init_enc_hidden, init_enc_cell):
    # sentence shape is (batch_size, sentence_length)
    sentence_emb = self.word_embedding(sentence)
    current_batch_size, sentence_len = sentence.size()

    # Input:
    # init_enc_hidden, init_enc_cell shape are both
    # (num_layers * num_directions, batch_size, hidden_size)
    #
    # Output:
    # enc_hidden_seq shape is (seq_len, batch_size, hidden_dim * num_directions)
    # num_directions = 2 for bi-directional LSTM
    # So assume that the 2 hidden vectors coming from the 2 directions
    # are already concatenated.
    #
    # enc_hidden_out shape is (num_layers * num_directions, batch_size, hidden_dim)
    # We use 1-layer here
    # Assume the 0-th dimension is: [forward, backward] final hidden states
    enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encoder(
      sentence_emb.view((sentence_len, current_batch_size, self.word_embedding_dim)),
      (init_enc_hidden, init_enc_cell))

    return enc_hidden_seq, (enc_hidden_out, enc_cell_out)

  # enc_hidden_seq shape is (seq_len, batch_size, hidden_dim * num_directions)
  # num_directions = 2 for bi-directional LSTM
  # So assume that the 2 hidden vectors coming from the 2 directions
  # are already concatenated.
  #
  # init_dec_hidden, init_dec_cell are both (batch_size, hidden_dim)
  def decode_train(self, label_seq, init_dec_hidden, init_dec_cell, enc_hidden_seq):
    # label_seq shape is (batch_size, label_seq_len)
    current_batch_size, label_seq_len = label_seq.size()

    source_seq_len = enc_hidden_seq.size()[0]

    dec_hidden_seq = []
    score_seq = []
    label_emb_seq = self.label_embedding(label_seq).permute(1, 0, 2)

    if self.gpu:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(current_batch_size, 1).zero_() \
        + self.BEG_INDEX).cuda()) \
        .view(current_batch_size, self.label_embedding_dim)
    else:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(current_batch_size, 1).zero_() \
        + self.BEG_INDEX)) \
        .view(current_batch_size, self.label_embedding_dim)

    dec_hidden_out, dec_cell_out = \
      self.decoder_cell(init_label_emb,
      (init_dec_hidden, init_dec_cell))

    # Attention
    if self.attention:
      attention_seq = []
      dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
      # This is the attention between (one time step of) decoder hidden vector
      # and the whole sequence of the encoder hidden vectors.
      # dec_hidden_out has shape (1, batch size, hidden dim)
      # attention has shape (batch size, 1, input sen len)
      # 1 means that here we only treat one hidden vector of decoder
      dec_hidden_out, attention = \
        self.attention(dec_hidden_out, enc_hidden_seq, 0)
      # 0 because we are now at "t=0"

      # remove the added dim
      dec_hidden_out = dec_hidden_out.view(current_batch_size, self.hidden_dim)

      # Remove the single dimension (in dim=1) that was from the fact that
      # here we only treat one hidden vector of decoder,
      # then append the resulting (batch size, input sen len) matrix
      # to attention_seq, expecting that after treating all words in the
      # output sequence, we will have attention_seq that is ready to be
      # transformed into shape (output sen len, batch size, input sen len)
      attention = attention.view(current_batch_size, source_seq_len)
      attention_seq.append(attention)
    # End if self.attention

    dec_hidden_seq.append(dec_hidden_out)
    score = self.hidden2score(dec_hidden_out) \
      .view(current_batch_size, self.label_size)
    score_seq.append(score)

    # The rest parts of the sentence
    for i in range(label_seq_len - 1):
      dec_hidden_out, dec_cell_out = self.decoder_cell(
        label_emb_seq[i], (dec_hidden_out, dec_cell_out))

      # Attention
      if self.attention:
        dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
        dec_hidden_out, attention = \
          self.attention(dec_hidden_out, enc_hidden_seq, i + 1)
        # i + 1 because now i is actually "t-1", and we need to input "t"

        # remove the added dim
        dec_hidden_out = dec_hidden_out.view(current_batch_size, self.hidden_dim)

        attention = attention.view(current_batch_size, source_seq_len)
        attention_seq.append(attention)
      # End if self.attention

      dec_hidden_seq.append(dec_hidden_out)
      score = self.hidden2score(dec_hidden_out) \
        .view(current_batch_size, self.label_size)
      score_seq.append(score)

    # It could make sense to reshape decoder hidden output
    # But currently we don't use this output in later stage
    dec_hidden_seq = torch.cat(dec_hidden_seq, dim=0) \
                     .view(label_seq_len, current_batch_size, self.hidden_dim)

    if self.attention:
      # This would be the attention alpha_{ij} coefficients
      # in the shape of (output seq len, batch size, input seq len)
      attention_seq = torch.cat(attention_seq, dim=0)
    else:
      attention_seq = None

    # For score_seq, actually don't need to reshape!
    # It happens that directly concatenate along dim = 0 gives you
    # a convenient shape (batch_size * seq_len, label_size)
    # for later cross entropy loss
    score_seq = torch.cat(score_seq, dim=0)

    return dec_hidden_seq, score_seq, attention_seq

  def train(self, shuffle):
    # Will manually average over (sentence_len * instance_num)
    loss_function = nn.CrossEntropyLoss(size_average=False)
    # Note that here we called nn.Module.parameters()
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    # self.train_X = [batch_1, batch_2, ...]
    # batch_i = [ [idx_1, idx_2, ...], ...]
    # Note that we don't require all batches have the same size
    instance_num = 0
    for batch in self.train_X:
      instance_num += len(batch)

    train_loss_list = []

    for epoch in range(self.max_epoch):
      time_begin = time.time()
      loss_sum = 0
      batch_num = len(self.train_X)

      batch_idx_list = range(batch_num)
      if shuffle:
        batch_idx_list = np.random.permutation(batch_idx_list)

      for batch_idx in batch_idx_list:
        sen = self.train_X[batch_idx]
        label = self.train_Y[batch_idx]

        current_batch_size = len(sen)
        current_sen_len = len(sen[0])

        # Always clear the gradients before use
        self.zero_grad()

        sen_var = Variable(torch.LongTensor(sen))
        label_var = Variable(torch.LongTensor(label))

        if self.gpu:
          sen_var = sen_var.cuda()
          label_var = label_var.cuda()

        # Initialize the hidden and cell states
        # The axes semantics are
        # (num_layers * num_directions, batch_size, hidden_size)
        # So 1 for single-directional LSTM encoder,
        # 2 for bi-directional LSTM encoder.
        init_enc_hidden = Variable(
          torch.zeros(2, current_batch_size, self.hidden_dim))
        init_enc_cell = Variable(
          torch.zeros(2, current_batch_size, self.hidden_dim))

        if self.gpu:
          init_enc_hidden = init_enc_hidden.cuda()
          init_enc_cell = init_enc_cell.cuda()

        enc_hidden_seq, (enc_hidden_out, enc_cell_out) = \
          self.encode(sen_var, init_enc_hidden, init_enc_cell)

        # The semantics of enc_hidden_out is (num_layers * num_directions,
        # batch, hidden_size), and it is "tensor containing the hidden state
        # for t = seq_len".
        #
        # Here we use a linear layer to transform the two-directions of the dec_hidden_out's into a single hid_dim vector, to use as the input of the decoder
        init_dec_hidden = self.enc2dec_hidden(torch.cat([enc_hidden_out[0], enc_hidden_out[1]], dim=1))
        init_dec_cell = self.enc2dec_cell(torch.cat([enc_cell_out[0], enc_cell_out[1]], dim=1))

        #init_dec_hidden = enc_hidden_out[0]
        #init_dec_cell = enc_cell_out[0]

        # Attention added
        dec_hidden_seq, score_seq, attention_seq = \
          self.decode_train(label_var, init_dec_hidden,
                            init_dec_cell, enc_hidden_seq)

        label_var_for_loss = label_var.permute(1, 0) \
          .contiguous().view(-1)

        loss = loss_function(score_seq, label_var_for_loss)

        if self.gpu:
          loss = loss.cpu()
        loss_sum += loss.data.numpy()[0] / current_sen_len
        
        loss.backward()
        optimizer.step()
      # for batch_idx

      avg_loss = loss_sum / instance_num
      train_loss_list.append(avg_loss)

      time_end = time.time()

      print("epoch", epoch, ", loss =", avg_loss,
            ", time =", time_end - time_begin)

    return train_loss_list

  def write_log(self):
    pass

  def decode_greedy(self, batch_size, seq_len, init_dec_hidden, init_dec_cell, enc_hidden_seq):
    # Current version is as parallel to beam as possible
    # for debugging purpose.

    score_seq = []

    # init_label's shape => (batch size, 1),
    # with all elements self.BEG_INDEX
    if self.gpu:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()).cuda() \
        + self.BEG_INDEX) \
        .view(batch_size, self.label_embedding_dim)
    else:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()) \
        + self.BEG_INDEX) \
        .view(batch_size, self.label_embedding_dim)

    # init_score's shape => (batch size, 1),
    # with all elements 0
    # For greedy, it's actually no need for initial score:
    # see the argument given later
    ##init_score = Variable(torch.FloatTensor(batch_size, 1).zero_())

    # Each y is (batch size, beam size = 1) matrix,
    # and there will be T_y of them in the sequence
    ##y_seq = []

    # dec_hidden_out has shape (batch size, hidden dim)
    dec_hidden_out, dec_cell_out = \
      self.decoder_cell(init_label_emb,
      (init_dec_hidden, init_dec_cell))

    # Attention
    if self.attention:
      dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
      dec_hidden_out, attention = \
        self.attention(dec_hidden_out, enc_hidden_seq, 0)

      # remove the added dim
      dec_hidden_out = dec_hidden_out.view(batch_size, self.hidden_dim)  

      attention = attention.view(batch_size, seq_len)
    # End if self.attention

    # score_out.shape => (batch size, |V^y|)
    ##score_out = self.hidden2score(dec_hidden_out) + init_score
    score_out = self.hidden2score(dec_hidden_out) \
      .view(batch_size, self.label_size)

    # To output the score_seq for calculating loss function value
    # during evaluation
    score_seq.append(score_out)

    # index.shape => (batch size, 1)
    # score => same
    # Here "1" in torch.max is "dim = 1"
    #
    # For greedy, it is actually unnecessary to record score,
    # because at each time step we always pick the highest score among
    # all possible words in vocab for the next step,
    # and this highest score is the same as
    # "this highest score + previous highest score (same constant added
    # to each possible words in vocab)".
    #
    ##score, index = torch.max(score_out, 1, keepdim = True)

    _, index = torch.max(score_out, 1, keepdim = True)
    # index.shape = (batch size, 1)
    label_pred_seq = index

    if self.attention:
      attention_pred_seq = [attention]

    # t = 1, 2, ..., (T_y - 1 == seq_len - 1)
    for t in range(1, seq_len):
      prev_pred_label_emb = \
        self.label_embedding(label_pred_seq[:, t - 1]) \
        .view(batch_size, self.label_embedding_dim)
      dec_hidden_out, dec_cell_out = self.decoder_cell(
        prev_pred_label_emb,
        (dec_hidden_out, dec_cell_out))

      # Attention
      if self.attention:
        dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
        dec_hidden_out, attention = \
          self.attention(dec_hidden_out, enc_hidden_seq, t)
        # Here we use t because it is the correct time step

        # remove the added dim
        dec_hidden_out = dec_hidden_out.view(batch_size, self.hidden_dim)

        attention = attention.view(batch_size, seq_len)
      # End if self.attention

      # For greedy, no need to add (previous) score
      ##score_out = self.hidden2score(dec_hidden_out) + score
      score_out = self.hidden2score(dec_hidden_out) \
        .view(batch_size, self.label_size)

      score_seq.append(score_out)

      _, index = torch.max(score_out, 1, keepdim = True)
      # Note that here, unlike in beam search (backtracking),
      # we simply append next predicted label
      label_pred_seq = torch.cat([label_pred_seq, index], dim = 1)

      if self.attention:
        attention_pred_seq.append(attention)
    # End for t

    if self.attention:
      # This would be the attention alpha_{ij} coefficients
      # in the shape of (output seq len, batch size, input seq len)
      attention_pred_seq = torch.stack(attention_pred_seq, dim=0)
    else:
      attention_pred_seq = None

    # For score_seq, actually don't need to reshape!
    # It happens that directly concatenate along dim = 0 gives you
    # a convenient shape (batch_size * seq_len, label_size)
    # for later cross entropy loss
    score_seq = torch.cat(score_seq, dim=0)

    return label_pred_seq, score_seq, attention_pred_seq

  def decode_beam(self, batch_size, seq_len, init_dec_hidden, init_dec_cell, enc_hidden_seq, beam_size):
    score_seq = []
    # TODO: beam search part would take some effort...

    # init_label's shape => (batch size, 1),
    # with all elements self.BEG_INDEX
    if self.gpu:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()).cuda() \
        + self.BEG_INDEX) \
        .view(batch_size, self.label_embedding_dim)
    else:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()) \
        + self.BEG_INDEX) \
        .view(batch_size, self.label_embedding_dim)

    # init_score's shape => (batch size, 1),
    # with all elements 0
    init_score = Variable(torch.FloatTensor(batch_size, 1).zero_())

    if self.gpu:
      init_score = init_score.cuda()

    # Each beta is (batch size, beam size) matrix,
    # and there will be T_y of them in the sequence
    # y => same
    beta_seq = []
    y_seq = []

    if self.attention:
      # This would be the attention alpha_{ij} coefficients
      # in the shape of (output seq len, batch size, beam size, input seq len)
      attention_seq = []

    # t = 0, only one input beam from init (t = -1)
    # Only one dec_hidden_out, dec_cell_out
    # => dec_hidden_out has shape (batch size, hidden dim)
    dec_hidden_out, dec_cell_out = \
      self.decoder_cell(init_label_emb,
      (init_dec_hidden, init_dec_cell))

    # Attention
    if self.attention:
      dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
      dec_hidden_out, attention = \
        self.attention(dec_hidden_out, enc_hidden_seq, 0)

      # remove the added dim
      dec_hidden_out = dec_hidden_out.view(batch_size, self.hidden_dim)
      attention = attention.view(batch_size, seq_len)

    # dec_hidden_beam shape => (1, batch size, hidden dim),
    # 1 because there is only 1 input beam
    dec_hidden_beam = torch.stack([dec_hidden_out], dim = 0)
    dec_cell_beam = torch.stack([dec_cell_out], dim = 0)

    if self.attention:
      # For better explanation, see in the "for t" loop below
      #
      # Originally attention has shape (batch size, input seq len)
      #
      # At t = 0, there is only 1 beam, so formally attention is actually
      # in shape (1, batch size, input seq len), where 1 is beam size.
      attention_beam = torch.stack([attention], dim = 0)

      # We need to permute (swap) the dimensions into
      # the shape (batch size, 1, input seq len)
      attention_beam = attention_beam.permute(1, 0, 2)

    # score_out.shape => (batch size, |V^y|)
    score_out = self.hidden2score(dec_hidden_out) + init_score
    # score_matrix.shape => (batch size, |V^y| * 1)
    # * 1 because there is only 1 input beam
    score_matrix = torch.cat([score_out], dim = 1)
    # All beta^{t=0, b} are actually 0
    # beta_beam.shape => (batch size, beam size),
    # each row is [y^{t, b=0}, y^{t, b=1}, ..., y^{t, b=B-1}]
    # y_beam, score_beam => same
    score_beam, index_beam = torch.topk(score_matrix, beam_size, dim = 1)
    beta_beam = torch.floor(index_beam.float() / self.label_size).long()
    y_beam = torch.remainder(index_beam, self.label_size)
    beta_seq.append(beta_beam)
    y_seq.append(y_beam)

    if self.attention:
      attention_seq.append(attention_beam)

    # t = 1, 2, ..., (T_y - 1 == seq_len - 1)
    for t in range(1, seq_len):
      # We loop through beam because we expect that
      # usually batch size > beam size
      dec_hidden_out_list = []
      dec_cell_out_list = []
      score_out_list = []

      if self.attention:
        attention_list =[]

      for b in range(beam_size):
        # Extract the b-th column of y_beam
        prev_pred_label_emb = self.label_embedding(
          y_seq[t - 1][:, b].contiguous() \
          .view(batch_size, 1)) \
          .view(batch_size, self.label_embedding_dim)

        # Extract: beta-th beam, batch_index-th row of dec_hidden_beam
        prev_dec_hidden_out = \
          dec_hidden_beam[beta_seq[t - 1][:, b],
          range(batch_size)]
        prev_dec_cell_out = \
          dec_cell_beam[beta_seq[t - 1][:, b],
          range(batch_size)]
        dec_hidden_out, dec_cell_out = self.decoder_cell(
          prev_pred_label_emb,
          (prev_dec_hidden_out, prev_dec_cell_out))

        # Attention
        if self.attention:
          dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
          dec_hidden_out, attention = \
            self.attention(dec_hidden_out, enc_hidden_seq, t)

          # remove the added dim
          dec_hidden_out = dec_hidden_out.view(batch_size, self.hidden_dim)
          attention = attention.view(batch_size, seq_len)
        # End if self.attention

        dec_hidden_out_list.append(dec_hidden_out)
        dec_cell_out_list.append(dec_cell_out)

        if self.attention:
          attention_list.append(attention)

        prev_score = score_beam[:, b].contiguous() \
          .view(batch_size, 1)
        score_out = self.hidden2score(dec_hidden_out) + prev_score
        score_out_list.append(score_out)
      # End for b

      # dec_hidden_beam shape => (beam size, batch size, hidden dim)
      dec_hidden_beam = torch.stack(dec_hidden_out_list, dim = 0)
      dec_cell_beam = torch.stack(dec_cell_out_list, dim = 0)

      if self.attention:
        attention_beam = torch.stack(attention_list, dim = 0)
        # Now attention_beam has shape (beam size, batch size, input seq len)
        # We need to permute (swap) the dimensions into
        # the shape (batch size, beam size, input seq len)
        attention_beam = attention_beam.permute(1, 0, 2)

      # score_matrix.shape => (batch size, |V^y| * beam_size)
      score_matrix = torch.cat(score_out_list, dim = 1)

      score_beam, index_beam = \
        torch.topk(score_matrix, beam_size, dim = 1)
      beta_beam = torch.floor(
        index_beam.float() / self.label_size).long()
      y_beam = torch.remainder(index_beam, self.label_size)
      beta_seq.append(beta_beam)
      y_seq.append(y_beam)

      if self.attention:
        attention_seq.append(attention_beam)
    # End for t

    # Only output the highest-scored beam (for each instance in the batch)
    label_pred_seq = y_seq[seq_len - 1][:, 0].contiguous() \
      .view(batch_size, 1)
    input_beam = beta_seq[seq_len - 1][:, 0]

    if self.attention:
      # Now attention_seq is
      # in the shape of (output seq len, batch size, beam size, input seq len)
      #
      # attention_pred_seq would be the attention alpha_{ij} coefficients
      # in the shape of (output seq len, batch size, input seq len)
      #
      # Here we initialize the first element
      attention_pred_seq = (attention_seq[t][range(batch_size), input_beam, :])[None, :, :]

    for t in range(seq_len - 2, -1, -1):
      label_pred_seq = torch.cat(
        [y_seq[t][range(batch_size), input_beam] \
        .contiguous().view(batch_size, 1),
        label_pred_seq], dim = 1)

      input_beam = beta_seq[t][range(batch_size), input_beam]

      if self.attention:
        attention_pred_seq = torch.cat([(attention_seq[t][range(batch_size), input_beam, :])[None, :, :], attention_pred_seq], dim = 0)
    # End for t

    if self.attention:
      attention_pred_seq = torch.stack(attention_pred_seq, dim = 0)
    else:
      attention_pred_seq = None

    return label_pred_seq, score_seq, attention_pred_seq

  # "beam_size = 0" will use greedy
  # "beam_size = 1" will still use beam search, just with beam size = 1
  def evaluate(self, eval_data_X, eval_data_Y, index2word, index2label, suffix, beam_size = 0):
    batch_num = len(eval_data_X)
    result_path = "../result/"

    f_sen = open(result_path + "sen_" + suffix + ".txt", 'w')
    f_pred = open(result_path + "pred_" + suffix + ".txt", 'w')
    f_label = open(result_path + "label_" + suffix + ".txt", 'w')
    f_result_processed = open(result_path + "result_processed_" + suffix + ".txt", 'w')

    for batch_idx in range(batch_num):
      sen = eval_data_X[batch_idx]
      label = eval_data_Y[batch_idx]
      current_batch_size = len(sen)
      current_sen_len = len(sen[0])

      # Always clear the gradients before use
      self.zero_grad()

      sen_var = Variable(torch.LongTensor(sen))
      label_var = Variable(torch.LongTensor(label))

      if self.gpu:
        sen_var = sen_var.cuda()
        label_var = label_var.cuda()

      # Initialize the hidden and cell states
      # The axes semantics are
      # (num_layers * num_directions, batch_size, hidden_size)
      # So 1 for single-directional LSTM encoder,
      # 2 for bi-directional LSTM encoder.
      init_enc_hidden = Variable(torch.zeros((2, current_batch_size, self.hidden_dim)))
      init_enc_cell = Variable(torch.zeros((2, current_batch_size, self.hidden_dim)))

      if self.gpu:
        init_enc_hidden = init_enc_hidden.cuda()
        init_enc_cell = init_enc_cell.cuda()

      enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encode(sen_var, init_enc_hidden, init_enc_cell)

      # The semantics of enc_hidden_out is (num_layers * num_directions,
      # batch, hidden_size), and it is "tensor containing the hidden state
      # for t = seq_len".
      #
      # Here we use a linear layer to transform the two-directions of the dec_hidden_out's into a single hid_dim vector, to use as the input of the decoder
      init_dec_hidden = self.enc2dec_hidden(torch.cat([enc_hidden_out[0], enc_hidden_out[1]], dim=1))
      init_dec_cell = self.enc2dec_cell(torch.cat([enc_cell_out[0], enc_cell_out[1]], dim=1))

      #init_dec_hidden = enc_hidden_out[0]
      #init_dec_cell = enc_cell_out[0]

      if beam_size > 0:
        label_pred_seq, score_seq, attention_pred_seq = self.decode_beam(current_batch_size, current_sen_len, init_dec_hidden, init_dec_cell, enc_hidden_seq, beam_size)
      else:
        label_pred_seq, score_seq, attention_pred_seq = self.decode_greedy(current_batch_size, current_sen_len, init_dec_hidden, init_dec_cell, enc_hidden_seq)

      # TODO: Need to compute loss function value here...

      # Here label_pred_seq.shape = (batch size, sen len)
      if self.gpu:
        label_pred_seq = label_pred_seq.cpu()

      label_pred_seq = label_pred_seq.data.numpy().tolist()

      # sen, label, label_pred_seq are list of lists,
      # thus I would like to flatten them for iterating easier
      sen = list(itertools.chain.from_iterable(sen))
      label = list(itertools.chain.from_iterable(label))
      label_pred_seq = list(itertools.chain.from_iterable(label_pred_seq))
      assert len(sen) == len(label) and len(label) == len(label_pred_seq)

      for i in range(len(sen)):
        f_sen.write(str(sen[i]) + '\n')
        f_label.write(str(label[i]) + '\n')
        f_pred.write(str(label_pred_seq[i]) + '\n')

        # clean version (does not print <PAD>, print a newline instead of <EOS>)
        #if sen[i] != 0 and sen[i] != 2: # not <PAD> and not <EOS>
        #if sen[i] != 0: # not <PAD>

        result_sen = index2word[sen[i]]
        result_label = index2label[label[i]]
        result_pred = index2label[label_pred_seq[i]]
        f_result_processed.write("%s %s %s\n" % (result_sen, result_label, result_pred))

        #elif sen[i] == 2:   # <EOS>
        #    f_result_processed.write('\n')
    # End for batch_idx

    f_sen.close()
    f_pred.close()
    f_label.close()
    f_result_processed.close()
