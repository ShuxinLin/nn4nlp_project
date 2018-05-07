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

import os


class ner(nn.Module):
  def __init__(self,
               word_embedding_dim, hidden_dim, label_embedding_dim,
               vocab_size, label_size,
               learning_rate=0.1, minibatch_size=1,
               max_epoch=300,
               train_X=None, train_Y=None,
               val_X=None, val_Y=None,
               test_X=None, test_Y=None,
               attention="fixed",
               gpu=False, gpu_no=0,
               pretrained=None,
               load_model_filename=None, load_map_location=None):

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
    self.val_X = val_X
    self.val_Y = val_Y
    self.test_X = test_X
    self.test_Y = test_Y
    self.load_model_filename = load_model_filename

    # For now we hard code the index of "<BEG>"
    self.BEG_INDEX = 1

    self.gpu = gpu
    self.gpu_no = gpu_no
    if self.gpu:
      self.cuda_dev = torch.device("cuda:" + str(self.gpu_no))

    # Attention
    if attention:
      self.attention = Attention(attention, self.hidden_dim, self.gpu)
    # Otherwise no attention
    else:
      self.attention = None

    self.word_embedding = nn.Embedding(self.vocab_size,
                                       self.word_embedding_dim)
    if pretrained:  # not None
      print("Using pretrained word embedding: ", pretrained)
      word_embedding_np = np.loadtxt('../dataset/WordEmbed/' + pretrained + '_embed.txt', dtype=float)    # load pretrained model: word2vec/glove
      assert self.vocab_size == word_embedding_np.shape[0]
      assert self.word_embedding_dim == word_embedding_np.shape[1]

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
    # Here we use a linear layer to transform the two-directions of the dec_hidden_out's into a single hidden_dim vector, to use as the input of the decoder
    self.enc2dec_hidden = nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=False)
    self.enc2dec_cell = nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=False)

    # Temporarily use same hidden dim for decoder
    self.decoder_cell = nn.LSTMCell(self.label_embedding_dim,
                                    self.hidden_dim)

    # Transform from hidden state to scores of all possible labels
    self.hidden2score = nn.Linear(self.hidden_dim, self.label_size)

    # From score to log probability
    self.score2logP = nn.LogSoftmax(dim=1)

    if self.load_model_filename:
      self.checkpoint = torch.load(self.load_model_filename, map_location=load_map_location)
      self.load_state_dict(self.checkpoint["state_dict"])

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

    sentence_emb = sentence_emb.permute(1, 0, 2)

    enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encoder(
      sentence_emb, (init_enc_hidden, init_enc_cell))

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
    logP_seq = []
    label_emb_seq = self.label_embedding(label_seq).permute(1, 0, 2)

    if self.gpu:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(current_batch_size, 1).zero_() \
        + self.BEG_INDEX).cuda(self.cuda_dev)) \
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
        self.attention(dec_hidden_out, enc_hidden_seq, 0, self.enc2dec_hidden)
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
    logP = self.score2logP(score).view(current_batch_size, self.label_size)
    logP_seq.append(logP)

    # The rest parts of the sentence
    for i in range(label_seq_len - 1):
      dec_hidden_out, dec_cell_out = self.decoder_cell(
        label_emb_seq[i], (dec_hidden_out, dec_cell_out))

      # Attention
      if self.attention:
        dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
        dec_hidden_out, attention = \
          self.attention(dec_hidden_out, enc_hidden_seq, i + 1, self.enc2dec_hidden)
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
      logP = self.score2logP(score).view(current_batch_size, self.label_size)
      logP_seq.append(logP)

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
    logP_seq = torch.cat(logP_seq, dim=0)

    #return dec_hidden_seq, score_seq, attention_seq
    return dec_hidden_seq, score_seq, logP_seq, attention_seq


  def train(self, shuffle, result_path, do_evaluation, beam_size, data="ner"):
    # Will manually average over (sentence_len * instance_num)
    #loss_function = nn.CrossEntropyLoss(size_average=False)

    # We now use logSoftmax -> NLLLoss
    loss_function = nn.NLLLoss(size_average=False)

    # Note that here we called nn.Module.parameters()
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    if self.load_model_filename:
      optimizer.load_state_dict(self.checkpoint["optimizer"])

    # self.train_X = [batch_1, batch_2, ...]
    # batch_i = [ [idx_1, idx_2, ...], ...]
    # Note that we don't require all batches have the same size
    instance_num = 0
    for batch in self.train_X:
      instance_num += len(batch)

    batch_num = len(self.train_X)

    train_loss_list = []

    output_file = open(os.path.join(result_path, "log.txt"), "w+")

    initial_epoch = (self.checkpoint["epoch"] + 1) if self.load_model_filename else 0

    for epoch in range(initial_epoch, initial_epoch + self.max_epoch):
      time_begin = time.time()
      loss_sum = 0

      batch_idx_list = range(batch_num)
      if shuffle:
        batch_idx_list = np.random.permutation(batch_idx_list)

      for batch_idx in batch_idx_list:
        sen = self.train_X[batch_idx]
        label = self.train_Y[batch_idx]

        #print("label=",label)

        current_batch_size = len(sen)
        current_sen_len = len(sen[0])

        # Always clear the gradients before use
        self.zero_grad()

        sen_var = Variable(torch.LongTensor(sen))
        label_var = Variable(torch.LongTensor(label))

        if self.gpu:
          sen_var = sen_var.cuda(self.cuda_dev)
          label_var = label_var.cuda(self.cuda_dev)

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
          init_enc_hidden = init_enc_hidden.cuda(self.cuda_dev)
          init_enc_cell = init_enc_cell.cuda(self.cuda_dev)

        enc_hidden_seq, (enc_hidden_out, enc_cell_out) = \
          self.encode(sen_var, init_enc_hidden, init_enc_cell)

        # The semantics of enc_hidden_out is (num_layers * num_directions,
        # batch, hidden_size), and it is "tensor containing the hidden state
        # for t = seq_len".
        #
        # Here we use a linear layer to transform the two-directions of the dec_hidden_out's into a single hidden_dim vector, to use as the input of the decoder
        init_dec_hidden = self.enc2dec_hidden(torch.cat([enc_hidden_out[0], enc_hidden_out[1]], dim=1))
        init_dec_cell = self.enc2dec_cell(torch.cat([enc_cell_out[0], enc_cell_out[1]], dim=1))

        #init_dec_hidden = enc_hidden_out[0]
        #init_dec_cell = enc_cell_out[0]

        # Attention added
        #dec_hidden_seq, score_seq, attention_seq = \
        dec_hidden_seq, score_seq, logP_seq, attention_seq = \
          self.decode_train(label_var, init_dec_hidden,
                            init_dec_cell, enc_hidden_seq)

        label_var_for_loss = label_var.permute(1, 0) \
          .contiguous().view(-1)

        if data == "ner":
          O_INDEX = 4
          gold_not_O_mask = (label_var_for_loss > O_INDEX).float()
          PENALTY = 1.5
          gold_not_O_mask = gold_not_O_mask * PENALTY
          score_seq[:, O_INDEX] = score_seq[:, O_INDEX] + gold_not_O_mask
        elif data == "ccg":
          pass
        else:
          print("Warning: ner.train(): Check special penalty for this dataset")

        logP_seq = self.score2logP(score_seq)

        # Input: (N,C) where C = number of classes
        # Target: (N) where each value is 0 <= targets[i] <= C−1
        #loss = loss_function(score_seq, label_var_for_loss)

        # We now use logSoftmax -> NLLLoss
        loss = loss_function(logP_seq, label_var_for_loss) \
               / (current_sen_len * current_batch_size)

        if self.gpu:
          loss_value = loss.cpu()
        else:
          loss_value = loss
        #print(loss_value.data.numpy())
        loss_sum += loss_value.data.numpy() * current_batch_size

        loss.backward()
        optimizer.step()
      # End for batch_idx

      avg_loss = loss_sum / instance_num
      train_loss_list.append(avg_loss)

      time_end = time.time()

      if do_evaluation:
        # Do evaluation on training set using model at this point
        # using decode_greedy or decode_beam
        #train_loss, train_fscore = self.evaluate(self.train_X, self.train_Y, None, None, "train", None, beam_size)
        # Do evaluation on validation set as well
        val_loss, val_fscore = self.evaluate(self.val_X, self.val_Y, None, None, "val", None, beam_size)
        test_loss, test_fscore = self.evaluate(self.test_X, self.test_Y, None, None, "test", None, beam_size)

        print("epoch", epoch,
              ", accumulated loss during training = %.6f\n" % avg_loss,
              #"\n training loss = %.6f" % train_loss,
              "validation loss = %.6f" % val_loss,
              ", test loss = %.6f\n" % test_loss,
              #"\n training F score = %.6f" % train_fscore,
              "validation F score = %.6f" % val_fscore,
              ", test F score = %.6f" % test_fscore,
              "\n time = %.6f" % (time_end - time_begin))

        #output_file.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epoch, avg_loss, train_loss, val_loss, test_loss, train_fscore, val_fscore, test_fscore, time_end - time_begin))
        output_file.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epoch, avg_loss, val_loss, test_loss, val_fscore, test_fscore, time_end - time_begin))
        output_file.flush()
      else:
        print("epoch", epoch,
              ", accumulated loss during training = %.6f" % avg_loss,
              "\n time = %.6f" % (time_end - time_begin))

        output_file.write("%d\t%f\t%f\n" % (epoch, avg_loss, time_end - time_begin))
        output_file.flush()
      # End if do_evaluation

      # Save model
      # In our current way of doing experiment, we don't keep is_best
      is_best = False
      checkpoint_filename = os.path.join(result_path, "ckpt_" + str(epoch) + ".pth")
      self.save_checkpoint({'epoch': epoch,
                       'state_dict': self.state_dict(),
                       'optimizer' : optimizer.state_dict()},
                      checkpoint_filename,
                      is_best)

    # End for epoch

    output_file.close()

    return train_loss_list


  def save_checkpoint(self, state, filename, is_best):
    torch.save(state, filename)
    if is_best:
      torch.save(state, "best.pth")


  def decode_greedy(self, batch_size, seq_len, init_dec_hidden, init_dec_cell, enc_hidden_seq):
    # Current version is as parallel to beam as possible
    # for debugging purpose.

    logP_seq = []

    # init_label's shape => (batch size, 1),
    # with all elements self.BEG_INDEX
    if self.gpu:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()).cuda(self.cuda_dev) \
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
        self.attention(dec_hidden_out, enc_hidden_seq, 0, self.enc2dec_hidden)

      # remove the added dim
      dec_hidden_out = dec_hidden_out.view(batch_size, self.hidden_dim)  

      attention = attention.view(batch_size, seq_len)
    # End if self.attention

    # score_out.shape => (batch size, |V^y|)
    ##score_out = self.hidden2score(dec_hidden_out) + init_score
    score_out = self.hidden2score(dec_hidden_out) \
      .view(batch_size, self.label_size)

    logP = self.score2logP(score_out).view(batch_size, self.label_size)
    logP_seq.append(logP)

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
          self.attention(dec_hidden_out, enc_hidden_seq, t, self.enc2dec_hidden)
        # Here we use t because it is the correct time step

        # remove the added dim
        dec_hidden_out = dec_hidden_out.view(batch_size, self.hidden_dim)

        attention = attention.view(batch_size, seq_len)
      # End if self.attention

      # For greedy, no need to add (previous) score
      ##score_out = self.hidden2score(dec_hidden_out) + score
      score_out = self.hidden2score(dec_hidden_out) \
        .view(batch_size, self.label_size)

      logP = self.score2logP(score_out).view(batch_size, self.label_size)
      logP_seq.append(logP)

      _, index = torch.max(logP, 1, keepdim = True)
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

    logP_seq = torch.cat(logP_seq, dim=0)
    logP_pred_seq = logP_seq

    return label_pred_seq, logP_pred_seq, attention_pred_seq


  def decode_beam(self, batch_size, seq_len, init_dec_hidden, init_dec_cell, enc_hidden_seq, beam_size):
    # This is for backtracking
    #
    # Each beta is (batch size, beam size) matrix,
    # and there will be T_y of them in the sequence
    # y => same
    beta_seq = []
    y_seq = []
    logP_seq = []
    accum_logP_seq = []
    if self.attention:
      # This would be the attention alpha_{ij} coefficients
      # in the shape of (output seq len, batch size, beam size, input seq len)
      attention_seq = []


    ### Initial step t = 0 ###

    # init_label's shape => (batch size, 1),
    # with all elements self.BEG_INDEX
    if self.gpu:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()).cuda(self.cuda_dev) \
        + self.BEG_INDEX) \
        .view(batch_size, self.label_embedding_dim)
    else:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()) \
        + self.BEG_INDEX) \
        .view(batch_size, self.label_embedding_dim)

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
        self.attention(dec_hidden_out, enc_hidden_seq, 0, self.enc2dec_hidden)

      # remove the added dim
      dec_hidden_out = dec_hidden_out.view(batch_size, self.hidden_dim)
      attention = attention.view(batch_size, seq_len)

    # dec_hidden_beam shape => (1, batch size, hidden dim),
    # 1 because there is only 1 input beam
    dec_hidden_beam = torch.stack([dec_hidden_out], dim = 0)
    dec_cell_beam = torch.stack([dec_cell_out], dim = 0)

    # This one is for backtracking (need permute)
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
    score_out = self.hidden2score(dec_hidden_out) \
      .view(batch_size, self.label_size)
    logP_out = self.score2logP(score_out).view(batch_size, self.label_size)

    # Initial step, accumulated logP is the same as logP
    accum_logP_out = logP_out

    logP_out_list = [logP_out]
    accum_logP_out_list = [accum_logP_out]

    # This one is for backtracking (need permute)
    logP_output_beam = torch.stack(logP_out_list, dim=0).permute(1, 0, 2)
    accum_logP_output_beam = torch.stack(accum_logP_out_list, dim=0).permute(1, 0, 2)

    # This is for topk
    #
    # accum_matrix.shape => (batch size, |V^y| * 1)
    # * 1 because there is only 1 input beam
    logP_matrix = torch.cat(logP_out_list, dim=1)
    accum_logP_matrix = torch.cat(accum_logP_out_list, dim=1)

    # All beta^{t=0, b} are actually 0
    # beta_beam.shape => (batch size, beam size),
    # each row is [y^{t, b=0}, y^{t, b=1}, ..., y^{t, b=B-1}]
    # y_beam, score_beam => same

    accum_logP_beam, index_beam = torch.topk(accum_logP_matrix, beam_size, dim=1)

    beta_beam = torch.floor(index_beam.float() / self.label_size).long()
    y_beam = torch.remainder(index_beam, self.label_size)

    # This one is for backtracking
    beta_seq.append(beta_beam)
    y_seq.append(y_beam)
    if self.attention:
      attention_seq.append(attention_beam)
    logP_seq.append(logP_output_beam)
    accum_logP_seq.append(accum_logP_output_beam)

    # t = 1, 2, ..., (T_y - 1 == seq_len - 1)
    for t in range(1, seq_len):
      # We loop through beam because we expect that
      # usually batch size > beam size
      #
      # DESIGN: This may not be true anymore in adaptive beam search,
      # since we expect batch size = 1 in this case.
      # So is beam operations vectorizable?

      dec_hidden_out_list = []
      dec_cell_out_list = []
      if self.attention:
        attention_list =[]
      ##score_out_list = []
      logP_out_list = []
      accum_logP_out_list = []

      # This is for backtracking
      ##logP_output_beam_list = []
      ##accum_logP_output_beam_list = []

      for b in range(beam_size):
        # Extract the b-th column of y_beam
        prev_pred_label_emb = self.label_embedding(
          y_beam[:, b].contiguous() \
          .view(batch_size, 1)) \
          .view(batch_size, self.label_embedding_dim)

        # Extract: beta-th beam, batch_index-th row of dec_hidden_beam
        prev_dec_hidden_out = \
          dec_hidden_beam[beta_beam[:, b],
          range(batch_size)]
        prev_dec_cell_out = \
          dec_cell_beam[beta_beam[:, b],
          range(batch_size)]
        dec_hidden_out, dec_cell_out = self.decoder_cell(
          prev_pred_label_emb,
          (prev_dec_hidden_out, prev_dec_cell_out))

        # Attention
        if self.attention:
          dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
          dec_hidden_out, attention = \
            self.attention(dec_hidden_out, enc_hidden_seq, t, self.enc2dec_hidden)

          # remove the added dim
          dec_hidden_out = dec_hidden_out.view(batch_size, self.hidden_dim)
          attention = attention.view(batch_size, seq_len)
        # End if self.attention

        dec_hidden_out_list.append(dec_hidden_out)
        dec_cell_out_list.append(dec_cell_out)
        if self.attention:
          attention_list.append(attention)

        score_out = self.hidden2score(dec_hidden_out)
        logP_out = self.score2logP(score_out).view(batch_size, self.label_size)

        accum_logP = accum_logP_beam[:, b].contiguous().view(batch_size, 1)

        accum_logP_out = logP_out + accum_logP

        logP_out_list.append(logP_out)
        accum_logP_out_list.append(accum_logP_out)

        # For backtracking
        ##logP_output_beam_list.append(logP_out)
        ##accum_logP_output_beam_list.append(accum_logP_out)
      # End for b

      # dec_hidden_beam shape => (beam size, batch size, hidden dim)
      dec_hidden_beam = torch.stack(dec_hidden_out_list, dim = 0)
      dec_cell_beam = torch.stack(dec_cell_out_list, dim = 0)

      # This one is for backtracking (need permute)
      if self.attention:
        attention_beam = torch.stack(attention_list, dim = 0)
        # Now attention_beam has shape (beam size, batch size, input seq len)
        # We need to permute (swap) the dimensions into
        # the shape (batch size, beam size, input seq len)
        attention_beam = attention_beam.permute(1, 0, 2)

      # This one is for backtracking (need permute)
      ##logP_output_beam = torch.stack(logP_output_beam_list, dim=0).permute(1, 0, 2)
      ##accum_logP_output_beam = torch.stack(accum_logP_output_beam_list, dim=0).permute(1, 0, 2)
      logP_output_beam = torch.stack(logP_out_list, dim=0).permute(1, 0, 2)
      accum_logP_output_beam = torch.stack(accum_logP_out_list, dim=0).permute(1, 0, 2)

      # score_matrix.shape => (batch size, |V^y| * beam_size)
      logP_matrix = torch.cat(logP_out_list, dim=1)
      accum_logP_matrix = torch.cat(accum_logP_out_list, dim=1)

      accum_logP_beam, index_beam = \
        torch.topk(accum_logP_matrix, beam_size, dim=1)

      beta_beam = torch.floor(
        index_beam.float() / self.label_size).long()
      y_beam = torch.remainder(index_beam, self.label_size)

      # For backtracking
      beta_seq.append(beta_beam)
      y_seq.append(y_beam)
      if self.attention:
        attention_seq.append(attention_beam)
      logP_seq.append(logP_output_beam)
      accum_logP_seq.append(accum_logP_output_beam)
    # End for t

    # Backtracking
    #
    # Only output the highest-scored beam (for each instance in the batch)
    label_pred_seq = y_seq[seq_len - 1][:, 0].contiguous().view(batch_size, 1)
    input_beam = beta_seq[seq_len - 1][:, 0]

    if self.attention:
      # Now attention_seq is
      # in the shape of (output seq len, batch size, beam size, input seq len)
      #
      # attention_pred_seq would be the attention alpha_{ij} coefficients
      # in the shape of (output seq len, batch size, input seq len)
      #
      # Here we initialize the first element
      attention_pred_seq = (attention_seq[seq_len - 1][range(batch_size), input_beam, :])[None, :, :]

    logP_pred_seq = (logP_seq[seq_len - 1][range(batch_size), input_beam, :])[None, :, :]
    accum_logP_pred_seq = (accum_logP_seq[seq_len - 1][range(batch_size), input_beam, :])[None, :, :]

    for t in range(seq_len - 2, -1, -1):
      label_pred_seq = torch.cat(
        [y_seq[t][range(batch_size), input_beam] \
        .contiguous().view(batch_size, 1),
        label_pred_seq], dim = 1)

      input_beam = beta_seq[t][range(batch_size), input_beam]

      if self.attention:
        attention_pred_seq = torch.cat([(attention_seq[t][range(batch_size), input_beam, :])[None, :, :], attention_pred_seq], dim = 0)

      logP_pred_seq = torch.cat([(logP_seq[t][range(batch_size), input_beam, :])[None, :, :], logP_pred_seq], dim = 0)
      accum_logP_pred_seq = torch.cat([(accum_logP_seq[t][range(batch_size), input_beam, :])[None, :, :], accum_logP_pred_seq], dim = 0)
    # End for t

    if self.attention:
      # BUG: fix later
      #attention_pred_seq = torch.stack(attention_pred_seq, dim = 0)
      pass
    else:
      attention_pred_seq = None

    # For score_seq, actually don't need to reshape!
    # It happens that directly concatenate along dim = 0 gives you
    # a convenient shape (batch_size * seq_len, label_size)
    # for later cross entropy loss
    logP_pred_seq = logP_pred_seq.view(batch_size * seq_len, self.label_size)
    accum_logP_pred_seq = accum_logP_pred_seq.view(batch_size * seq_len, self.label_size)

    return label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq


  # For German dataset, f_score_index_begin = 5 (because O_INDEX = 4)
  # For toy dataset, f_score_index_begin = 4 (because {0: '<s>', 1: '<e>', 2: '<p>', 3: '<u>', ...})
  def evaluate(self, eval_data_X, eval_data_Y, index2word, index2label, suffix, result_path, decode_method, beam_size, max_beam_size, agent, reward_coef_fscore, reward_coef_beam_size, f_score_index_begin, generate_episode=True, episode_save_path=None):
    batch_num = len(eval_data_X)

    if result_path:
      f_sen = open(result_path + "sen_" + suffix + ".txt", 'w')
      f_pred = open(result_path + "pred_" + suffix + ".txt", 'w')
      f_label = open(result_path + "label_" + suffix + ".txt", 'w')
      f_result_processed = open(result_path + "result_processed_" + suffix + ".txt", 'w')
      f_beam_size = open(result_path + 'beam_size_' + suffix + ".txt", 'w')

    if generate_episode:
      episode_file = open(episode_save_path, "w+")

    instance_num = 0
    correctness = 0

    beam_size_seqs = []
    action_seqs = []

    for batch in eval_data_X:
      instance_num += len(batch)

    correct_count = 0
    total_count = 0

    # train_memory = [(action, state vector), ...]
    if generate_episode:
      train_memory = []

    for batch_idx in range(batch_num):
      sen = eval_data_X[batch_idx]
      label = eval_data_Y[batch_idx]
      current_batch_size = len(sen)
      current_sen_len = len(sen[0])

      sen_var = Variable(torch.LongTensor(sen))
      label_var = Variable(torch.LongTensor(label))

      if self.gpu:
        sen_var = sen_var.cuda(self.cuda_dev)
        label_var = label_var.cuda(self.cuda_dev)

      # Initialize the hidden and cell states
      # The axes semantics are
      # (num_layers * num_directions, batch_size, hidden_size)
      # So 1 for single-directional LSTM encoder,
      # 2 for bi-directional LSTM encoder.
      init_enc_hidden = Variable(torch.zeros((2, current_batch_size, self.hidden_dim)))
      init_enc_cell = Variable(torch.zeros((2, current_batch_size, self.hidden_dim)))

      if self.gpu:
        init_enc_hidden = init_enc_hidden.cuda(self.cuda_dev)
        init_enc_cell = init_enc_cell.cuda(self.cuda_dev)

      enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encode(sen_var, init_enc_hidden, init_enc_cell)

      # The semantics of enc_hidden_out is (num_layers * num_directions,
      # batch, hidden_size), and it is "tensor containing the hidden state
      # for t = seq_len".
      #
      # Here we use a linear layer to transform the two-directions of the dec_hidden_out's into a single hidden_dim vector, to use as the input of the decoder
      init_dec_hidden = self.enc2dec_hidden(torch.cat([enc_hidden_out[0], enc_hidden_out[1]], dim=1))
      init_dec_cell = self.enc2dec_cell(torch.cat([enc_cell_out[0], enc_cell_out[1]], dim=1))

      if decode_method == "greedy":
        label_pred_seq, logP_pred_seq, attention_pred_seq = self.decode_greedy(current_batch_size, current_sen_len, init_dec_hidden, init_dec_cell, enc_hidden_seq)
        beam_size_seqs.append([1] * (len(label_pred_seq[0]) - 1))
      elif decode_method == "beam":
        label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq = self.decode_beam(current_batch_size, current_sen_len, init_dec_hidden, init_dec_cell, enc_hidden_seq, beam_size)
        beam_size_seqs.append([beam_size] * (len(label_pred_seq[0]) - 1))
      elif decode_method == "adaptive":
        # the input argument "beam_size" serves as initial_beam_size here
        label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq, episode, beam_size_seq = self.decode_beam_adaptive(current_sen_len, init_dec_hidden, init_dec_cell, enc_hidden_seq, beam_size, max_beam_size, agent, reward_coef_fscore, reward_coef_beam_size, label_var, f_score_index_begin, generate_episode=generate_episode)
        beam_size_seqs.append(beam_size_seq)

        if generate_episode:
          for experience_tuple in episode:
            episode_file.write("%d" % experience_tuple[1])
            for state_element in experience_tuple[0]:
              episode_file.write("\t%f" % state_element)
            episode_file.write("\n")

        ### Debugging...
        #print("input sentence =", sen)
        #print("true label =", label)
        #print("predicted label =", label_pred_seq)
        #print("episode =", episode)

      correct_count += (label_pred_seq == label_var).sum()
      total_count += label_var.shape[1]

      # Write result into file
      if result_path:
        if self.gpu:
          label_pred_seq = label_pred_seq.cpu()

        label_pred_seq = label_pred_seq.data.numpy().tolist()

        # Here label_pred_seq.shape = (batch size, sen len)

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

        if decode_method == "adaptive":
          beam_size_seq_str = ' '.join(map(str, beam_size_seq))
          f_beam_size.write(beam_size_seq_str + '\n')

    # End for batch_idx

    if self.gpu:
      correct_count = correct_count.cpu()
      total_count = total_count.cpu()

    """
    true_pos_count = true_pos_count.data.numpy()[0]
    pred_pos_count = pred_pos_count.data.numpy()[0]
    true_pred_pos_count = true_pred_pos_count.data.numpy()[0]
    """
    correct_count = correct_count.data.numpy()
    total_count = total_count.data.numpy()

    accuracy = float(correct_count) / total_count

    if result_path:
      f_sen.close()
      f_pred.close()
      f_label.close()
      f_result_processed.close()
      f_beam_size.close()

    if generate_episode:
      episode_file.close()

    total_beam_number_in_dataset = sum([sum(beam_size_seq) for beam_size_seq in beam_size_seqs])
    avg_beam_sizes = [(sum(beam_size_seq) / len(beam_size_seq) if len(beam_size_seq) else 0) for beam_size_seq in beam_size_seqs]
    avg_beam_sizes = list(filter(lambda xx: xx > 0, avg_beam_sizes))
    avg_beam_size = sum(avg_beam_sizes) / len(avg_beam_sizes)

    return accuracy, total_beam_number_in_dataset, avg_beam_size


  # decode_beam_step - this function basically servers as the "env.step()" function in RL: (citing below)
  # "Take this beam_size picked at t = 1 for sending into t = 2, input them into LSTM at t = 2, and output the new output accum_logP_matrix at t = 2."
  #
  # Args: This function takes a beam of (y, beta) = (label index prediction from the previous step, incoming beam index), and the beam of hidden vectors and cell vectors from the previous step, and the accumulated logP's of these (y, beta)'s in the beam.
  # Returns: Accumulated logP of all the possible labels in all beams [shape is (batch size, incoming beam size * label vocab number)], (non-accumulated, row prediction at this time step) logP of all the possible labels in all beams [shape is (batch size, incoming beam size * label vocab number)], the output hidden vectors and cell vectors from all incoming beams [shape is (incoming beam size, batch size = 1, hidden dim)].
  # Note that: Currently only support single instance, no minibatch.
  #
  # dec_hidden_beam_in: The beam of decoder hidden vectors from the previous step.
  #                     Shape: (beam size, batch size = 1, hidden dim)
  # enc_hidden_seq: For attention. Can be put to None if no attention is used.
  #                 Shape: (seq len, batch size = 1, 2 * hidden dim) => for bi-directional LSTM encoder
  # attend_index: The index (time step, 0-based) in the enc_hidden_seq to attend to (used in fixed attention)
  def decode_beam_step(self, beam_size_in, y_beam_in, beta_beam_in, dec_hidden_beam_in, dec_cell_beam_in, accum_logP_beam_in, enc_hidden_seq, seq_len, attend_index):

    # Currently only support single instance, no minibatch
    batch_size = 1

    # dec_hidden_out_list is the collection of the output hidden vectors from the input beam of y's.
    # If there are beam_size_in incoming beams, then there are also beam_size_in output hidden vectors in dec_hidden_out_list (one-to-one)
    # These hidden vectors will then be chosen according to top-K operation later
    # It is a list of (batch size = 1, hidden dim) matrices
    dec_hidden_out_list = []
    dec_cell_out_list = []
    if self.attention:
      attention_list = []
    accum_logP_out_list = []
    logP_out_list = []

    for b in range(beam_size_in):
      # Extract the b-th column of y_beam
      y_emb_in = self.label_embedding(
        y_beam_in[:, b].contiguous() \
        .view(batch_size, 1)) \
        .view(batch_size, self.label_embedding_dim)

      # Extract: beta-th beam, batch_index-th row of dec_hidden_beam_in
      dec_hidden_in = \
        dec_hidden_beam_in[beta_beam_in[:, b], range(batch_size)] \
        .view(batch_size, self.hidden_dim)
      dec_cell_in = \
        dec_cell_beam_in[beta_beam_in[:, b], range(batch_size)] \
        .view(batch_size, self.hidden_dim)
      dec_hidden_out, dec_cell_out = \
        self.decoder_cell(y_emb_in, (dec_hidden_in, dec_cell_in))

      # Attention
      if self.attention:
        dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
        dec_hidden_out, attention = \
          self.attention(dec_hidden_out, enc_hidden_seq, attend_index, self.enc2dec_hidden)

        # remove the added dim
        dec_hidden_out = dec_hidden_out.view(batch_size, self.hidden_dim)
        attention = attention.view(batch_size, seq_len)
      # End if self.attention

      dec_hidden_out_list.append(dec_hidden_out)
      dec_cell_out_list.append(dec_cell_out)
      if self.attention:
        attention_list.append(attention)

      # score_out and logP_out are shape (batch size = 1, label size)
      # They are the row predictions of the decoder for this step (not accumulated)
      score_out = self.hidden2score(dec_hidden_out)
      logP_out = self.score2logP(score_out).view(batch_size, self.label_size)

      # accum_logP_in is shape (batch size = 1, 1)
      # It is the accumulated logP (sum over the path) of this beam we are now dealing with
      accum_logP_in = accum_logP_beam_in[:, b].contiguous().view(batch_size, 1)

      # The new accumulated logP would be (accum_logP_in + the predicted logP made for each label at this step)
      # Shape (batch size = 1, label size)
      # accum_logP_out is from which top-K will pick the new beam to output
      accum_logP_out = logP_out + accum_logP_in

      accum_logP_out_list.append(accum_logP_out)
      logP_out_list.append(logP_out)
    # End for b

    # Here we should output the "state" we have so far
    # Some external program should take this state, and determine the new beam size. It will then call other function to generate new beams, and then take those beams as new input to this function.

    # This one is for backtracking (need permute)
    logP_output_beam = torch.stack(logP_out_list, dim=0).permute(1, 0, 2)
    accum_logP_output_beam = torch.stack(accum_logP_out_list, dim=0).permute(1, 0, 2)

    accum_logP_matrix = torch.cat(accum_logP_out_list, dim=1) \
                  .view(batch_size, beam_size_in * self.label_size)
    logP_matrix = torch.cat(logP_out_list, dim=1) \
                  .view(batch_size, beam_size_in * self.label_size)
    # dec_hidden_beam_out shape => (beam_size_in, batch size = 1, hidden dim)
    dec_hidden_beam_out = torch.stack(dec_hidden_out_list, dim=0)
    dec_cell_beam_out = torch.stack(dec_cell_out_list, dim=0)

    attention_beam_out = None
    # This one is for backtracking (need permute)
    if self.attention:
      attention_beam_out = torch.stack(attention_list, dim = 0)
      # Now attention_beam has shape (beam size, batch size, input seq len)
      # We need to permute (swap) the dimensions into
      # the shape (batch size, beam size, input seq len)
      attention_beam_out = attention_beam_out.permute(1, 0, 2)
    else:
      attention_beam_out = None

    return accum_logP_matrix, logP_matrix, dec_hidden_beam_out, dec_cell_beam_out, attention_beam_out, accum_logP_output_beam, logP_output_beam


  ###############################
  # Reinforcement learning:
  # Episodes and learning signals only come from sentences with length >= 3.
  #
  # Consider sentence t = 0, 1, ..., L_y - 1, where L_y >= 3.
  # s_0 => accum_logP_matrix (coming from decoding the initial_beam_size incoming beam generated at t = 0 output) and the initial_beam_size (beam size picked to send into t = 1 for LSTM to generate the accum_logP_matrix at the output of t = 1).
  # (We need to take the top-1 of this accum_logP_matrix, do the backtracking to find the best sequence [for t = 0, 1], and compute the F-score. This is for the reward calculation later.)
  # a_0 => Look at this accum_logP_matrix and initial_beam_size, decide whether to increase or decrease the beam size to send into t = 2 LSTM step.
  # "env.step()" => take this beam_size picked at t = 1 for sending into t = 2, input them into LSTM at t = 2, and output the new output accum_logP_matrix at t = 2.
  # r_0 => use this accum_logP_matrix at t = 2 output, simply pick the top-1, and do backtracking, find the best sequence so far (t = 0, 1, 2), and compute the F-score. Take the difference of this F-score and the previous F-score. This is one contribution to the reward. Then, if a_0 is to +1 the beam size, contribute -1 to the reward, and vice versa; this is the second contribution to the reward. (The two contribution is linearly combined with some coefficients.)
  # s_1 => the accum_logP_matrix at t = 2 output, and the beam size picked at t = 1 output (by the agent, after action a_0).
  # Then go on.
  # About terminal state: Easy to see. For example, if sentence length is 3, then t = 2 is the last one. Indeed we don't have to take further action.
  ###############################
  #
  # This function is like generate_episode() for RL
  def decode_beam_adaptive(self, seq_len, init_dec_hidden, init_dec_cell, enc_hidden_seq, initial_beam_size, max_beam_size, agent, reward_coef_fscore, reward_coef_beam_size, label_true_seq, f_score_index_begin, generate_episode=True):
    # Currently, batch size can only be 1
    batch_size = 1

    # Each beta is (batch size, beam size) matrix,
    # and there will be T_y of them in the sequence
    # y => same
    beta_seq = []
    y_seq = []

    logP_seq = []
    accum_logP_seq = []

    if self.attention:
      # This would be the attention alpha_{ij} coefficients
      # in the shape of (output seq len, batch size, beam size, input seq len)
      attention_seq = []
    else:
      attention_seq = None

    # For RL episode
    episode = []

    # init_label's shape => (batch size, 1),
    # with all elements self.BEG_INDEX
    if self.gpu:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()).cuda(self.cuda_dev) \
        + self.BEG_INDEX) \
        .view(batch_size, self.label_embedding_dim)
    else:
      init_label_emb = \
        self.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()) \
        + self.BEG_INDEX) \
        .view(batch_size, self.label_embedding_dim)

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
        self.attention(dec_hidden_out, enc_hidden_seq, 0, self.enc2dec_hidden)

      # remove the added dim
      dec_hidden_out = dec_hidden_out.view(batch_size, self.hidden_dim)
      attention = attention.view(batch_size, seq_len)

    # dec_hidden_beam shape => (1, batch size, hidden dim),
    # 1 because there is only 1 input beam
    dec_hidden_beam = torch.stack([dec_hidden_out], dim = 0)
    dec_cell_beam = torch.stack([dec_cell_out], dim = 0)

    # This one is for backtracking (need permute)
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
    score_out = self.hidden2score(dec_hidden_out) \
      .view(batch_size, self.label_size)
    logP_out = self.score2logP(score_out).view(batch_size, self.label_size)

    # Initial step, accumulated logP is the same as logP
    accum_logP_out = logP_out

    logP_out_list = [logP_out]
    accum_logP_out_list = [accum_logP_out]

    # This one is for backtracking (need permute)
    logP_output_beam = torch.stack(logP_out_list, dim=0).permute(1, 0, 2)
    accum_logP_output_beam = torch.stack(accum_logP_out_list, dim=0).permute(1, 0, 2)

    # score_matrix.shape => (batch size, |V^y| * 1)
    # * 1 because there is only 1 input beam
    logP_matrix = torch.cat(logP_out_list, dim=1)
    accum_logP_matrix = torch.cat(accum_logP_out_list, dim=1)

    # Just for code consistency (about reward calculation)
    cur_beam_size_in = 1

    # Just for code consistency (about experience tuple)
    cur_state = self.make_state(accum_logP_matrix, logP_matrix, 1, max_beam_size)
    action = None

    # All beta^{t=0, b} are actually 0
    # beta_beam.shape => (batch size, beam size),
    # each row is [y^{t, b=0}, y^{t, b=1}, ..., y^{t, b=B-1}]
    # y_beam, score_beam => same

    action_seq = []
    beam_size_seq = []
    beam_size = initial_beam_size
    beam_size_seq.append(beam_size)
    accum_logP_beam, index_beam = torch.topk(accum_logP_matrix, beam_size, dim=1)

    beta_beam = torch.floor(index_beam.float() / self.label_size).long()
    y_beam = torch.remainder(index_beam, self.label_size)

    # This one is for backtracking
    beta_seq.append(beta_beam)
    y_seq.append(y_beam)
    if self.attention:
      attention_seq.append(attention_beam)
    logP_seq.append(logP_output_beam)
    accum_logP_seq.append(accum_logP_output_beam)

    # Just for sentence with length = 1
    label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq = self.backtracking(1, batch_size, y_seq, beta_seq, attention_seq, logP_seq, accum_logP_seq)

    # t = 1, 2, ..., (T_y - 1 == seq_len - 1)
    for t in range(1, seq_len):
      # We loop through beam because we expect that
      # usually batch size > beam size
      #
      # DESIGN: This may not be true anymore in adaptive beam search,
      # since we expect batch size = 1 in this case.
      # So is beam operations vectorizable?

      accum_logP_matrix, logP_matrix, dec_hidden_beam, dec_cell_beam, attention_beam, accum_logP_output_beam, logP_output_beam = \
        self.decode_beam_step(beam_size, y_beam, beta_beam,
                              dec_hidden_beam, dec_cell_beam, accum_logP_beam,
                              enc_hidden_seq, seq_len, t)

      # Actually, at t = T_y - 1 == seq_len - 1,
      # you don't have to take action (you don't have to pick a beam of predictions anymore), because at this last output step, you would pick only the highest result, and do the backtracking from it to determine the best sequence.
      # However, in the current version of this code, we temporarily keep doing one more beam picking, just to be compatible with the backtracking function and the rest of the code.
      # We delay the improvement to the future work.
      #
      # Note that this state is actually the output state at t
      state = self.make_state(accum_logP_matrix, logP_matrix,
                              beam_size, max_beam_size)

      # For experience tuple
      prev_state = cur_state
      cur_state = state
      prev_action = action

      # For reward calculation
      prev_beam_size_in = cur_beam_size_in
      cur_beam_size_in = beam_size

      action = agent.get_action(state)
      action_seq.append(action)
      if action == agent.DECREASE and beam_size > 1:
        beam_size -= 1
      elif action == agent.INCREASE and beam_size < max_beam_size:
        beam_size += 1

      # Fix in the future: We actually don't utilize the beam generated in the last time step---we only use top-1 to do backtracking. So here we don't include the beam size at the last step.
      if t <= seq_len - 2:
        beam_size_seq.append(beam_size)

      accum_logP_beam, index_beam = \
        torch.topk(accum_logP_matrix, beam_size, dim=1)

      beta_beam = torch.floor(
        index_beam.float() / self.label_size).long()
      y_beam = torch.remainder(index_beam, self.label_size)
      beta_seq.append(beta_beam)
      y_seq.append(y_beam)
      if self.attention:
        attention_seq.append(attention_beam)
      logP_seq.append(logP_output_beam)
      accum_logP_seq.append(accum_logP_output_beam)

      if generate_episode:
        # Compute the F-score for the sequence [0, 1, ..., t] (length t+1) using y_seq, betq_seq we got so far. This is the ("partial", so to speak) F-score at this t.
        label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq = self.backtracking(t + 1, batch_size, y_seq, beta_seq, attention_seq, logP_seq, accum_logP_seq)
        cur_fscore = self.get_fscore(label_pred_seq, label_true_seq, f_score_index_begin)

        # If t >= 2, compute the reward,
        # and generate the experience tuple ( s_{t-1}, a_{t-1}, r_{t-1}, s_t )
        if t >= 2:
          reward = self.get_reward(cur_fscore, fscore, cur_beam_size_in, prev_beam_size_in, reward_coef_fscore, reward_coef_beam_size)
          experience_tuple = (prev_state, prev_action, reward, cur_state)
          episode.append(experience_tuple)

        fscore = cur_fscore
      else:
        if t == seq_len - 1:
          label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq = self.backtracking(t + 1, batch_size, y_seq, beta_seq, attention_seq, logP_seq, accum_logP_seq)
    # End for t

    return label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq, episode, beam_size_seq


  # make_state - generate the state for the RL agent
  def make_state(self, accum_logP_matrix, logP_matrix, beam_size, max_beam_size):
    accum_logP_state, _ = \
      torch.topk(accum_logP_matrix, max_beam_size, dim=1)
    logP_state, _ = \
      torch.topk(logP_matrix, max_beam_size, dim=1)

    if self.gpu:
      accum_logP_state = accum_logP_state.cpu().data.numpy()[0]
      logP_state = logP_state.cpu().data.numpy()[0]
    else:
      accum_logP_state = accum_logP_state.data.numpy()[0]
      logP_state = logP_state.data.numpy()[0]

    state = np.concatenate((accum_logP_state, logP_state, np.array([beam_size])), axis=0)

    return state


  def backtracking(self, seq_len, batch_size, y_seq, beta_seq, attention_seq, logP_seq, accum_logP_seq):
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
      attention_pred_seq = (attention_seq[seq_len - 1][range(batch_size), input_beam, :])[None, :, :]

    logP_pred_seq = (logP_seq[seq_len - 1][range(batch_size), input_beam, :])[None, :, :]
    accum_logP_pred_seq = (accum_logP_seq[seq_len - 1][range(batch_size), input_beam, :])[None, :, :]

    for t in range(seq_len - 2, -1, -1):
      label_pred_seq = torch.cat(
        [y_seq[t][range(batch_size), input_beam] \
        .contiguous().view(batch_size, 1),
        label_pred_seq], dim = 1)

      input_beam = beta_seq[t][range(batch_size), input_beam]

      if self.attention:
        attention_pred_seq = torch.cat([(attention_seq[t][range(batch_size), input_beam, :])[None, :, :], attention_pred_seq], dim = 0)

      logP_pred_seq = torch.cat([(logP_seq[t][range(batch_size), input_beam, :])[None, :, :], logP_pred_seq], dim = 0)
      accum_logP_pred_seq = torch.cat([(accum_logP_seq[t][range(batch_size), input_beam, :])[None, :, :], accum_logP_pred_seq], dim = 0)
    # End for t

    if self.attention:
      # BUG: to fix later
      #attention_pred_seq = torch.stack(attention_pred_seq, dim = 0)
      pass
    else:
      attention_pred_seq = None

    # For score_seq, actually don't need to reshape!
    # It happens that directly concatenate along dim = 0 gives you
    # a convenient shape (batch_size * seq_len, label_size)
    # for later cross entropy loss
    #
    # We actually don't calculate loss in evaluation anymore
    logP_pred_seq = logP_pred_seq.view(batch_size * seq_len, self.label_size)
    accum_logP_pred_seq = accum_logP_pred_seq.view(batch_size * seq_len, self.label_size)

    return label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq


  # Observe the beam size to determine the reward, because it is possible that the agent wants to decrease the beam size, but the beam size is already minimum, so the environment does not allow the beam size to decrease.
  def get_reward(self, cur_fscore, prev_fscore, cur_beam_size_in, prev_beam_size_in, reward_coef_fscore, reward_coef_beam_size):
    reward = reward_coef_fscore * (cur_fscore - prev_fscore) * 0.01 + reward_coef_beam_size * (prev_beam_size_in - cur_beam_size_in)
    return reward


  # It computes partial F-score, so expect label_var.size()[1] >= label_pred_seq.size()[1] generally
  # This function should work for batch size > 1 as well
  def get_fscore(self, label_pred_seq_input, label_var_input, f_score_index_begin):
    if self.gpu:
      label_pred_seq = label_pred_seq_input.cpu().data.numpy()
      label_var = label_var_input.cpu().data.numpy()
    else:
      label_pred_seq = label_pred_seq_input.data.numpy()
      label_var = label_var_input.data.numpy()

    #print("label_pred_seq=", label_pred_seq)
    #print("label_pred_seq.shape[1]=", label_pred_seq.shape[1])
    #print("label_var=", label_var)
    #print("label_var.shape[1]=", label_var.shape[1])

    label_pred_seq_padded = np.pad(label_pred_seq, ((0, 0), (0, label_var.shape[1] - label_pred_seq.shape[1])), "constant", constant_values=(-1, -1))

    #print("label_pred_seq_padded=", label_pred_seq_padded)

    correct_count = (label_pred_seq_padded == label_var).sum()
    total_count = label_var.shape[1]
    accuracy = float(correct_count) / total_count

    return accuracy
