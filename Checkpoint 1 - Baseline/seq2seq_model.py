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


class Seq2Seq(nn.Module):
    def __init__(self,
                 word_embedding_dim, hidden_dim, label_embedding_dim,
                 vocab_size, label_size,
                 learning_rate=0.1, minibatch_size=1,
                 max_epoch=300, index2word=None, index2label=None,
                 train_X=None, train_Y=None,
                 test_X=None, test_Y=None):
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.label_embedding_dim = label_embedding_dim
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.max_epoch = max_epoch

        self.index2word = index2word
        self.index2label = index2label

        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

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

        return enc_hidden_seq, (enc_hidden_out, enc_cell_out)

    def decode_train(self, label_seq, init_dec_hidden, init_dec_cell):
        label_seq_len = label_seq.size()[1]

        dec_hidden_seq = []
        score_seq = []
        label_emb_seq = self.label_embedding(label_seq).permute(1, 0, 2)

        LABEL_BEGIN_INDEX = 1
        init_label_emb = self.label_embedding(
            Variable(torch.LongTensor(self.minibatch_size, 1).zero_() + LABEL_BEGIN_INDEX)).view(self.minibatch_size,
                                                                                                 self.label_embedding_dim)
        dec_hidden_out, dec_cell_out = self.decoder_cell(
            init_label_emb, (init_dec_hidden, init_dec_cell))
        dec_hidden_seq.append(dec_hidden_out)
        score = self.hidden2score(dec_hidden_out)
        score_seq.append(score)

        # The rest parts of the sentence
        for i in range(label_seq_len - 1):
            dec_hidden_out, dec_cell_out = self.decoder_cell(
                label_emb_seq[i], (dec_hidden_out, dec_cell_out))
            dec_hidden_seq.append(dec_hidden_out)
            score = self.hidden2score(dec_hidden_out)
            score_seq.append(score)

        # It could make sense to reshape decoder hidden output
        # But currently we don't use this output in later stage
        dec_hidden_seq = torch.cat(dec_hidden_seq, dim=0).view(label_seq_len, self.minibatch_size, self.hidden_dim)

        # For score_seq, actually don't need to reshape!
        # It happens that directly concatenate along dim = 0 gives you a convenient shape (batch_size * seq_len, label_size) for later cross entropy loss
        score_seq = torch.cat(score_seq, dim=0)

        return dec_hidden_seq, score_seq

    def train(self):
        # Will manually average over (sentence_len * instance_num)
        loss_function = nn.CrossEntropyLoss(size_average=False)
        # Note that here we called nn.Module.parameters()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        instance_num = 0
        for batch in self.train_X:
            instance_num += len(batch)

        start_time = time.time()
        for epoch in range(self.max_epoch):
            loss_sum = 0
            batch_num = len(self.train_X)

            for batch_idx in range(batch_num):
                sen = self.train_X[batch_idx]
                label = self.train_Y[batch_idx]

                # current_batch_size = len(sen)
                current_sen_len = len(sen[0])

                # Always clear the gradients before use
                self.zero_grad()

                sen_var = Variable(torch.LongTensor(sen))
                label_var = Variable(torch.LongTensor(label))

                # Initialize the hidden and cell states
                # The axes semantics are (num_layers, minibatch_size, hidden_dim)
                init_enc_hidden = Variable(
                    torch.zeros(1, self.minibatch_size, self.hidden_dim))
                init_enc_cell = Variable(
                    torch.zeros(1, self.minibatch_size, self.hidden_dim))

                enc_hidden_seq, (enc_hidden_out, enc_cell_out) = \
                    self.encode(sen_var, init_enc_hidden, init_enc_cell)

                init_dec_hidden = enc_hidden_out[0]
                init_dec_cell = enc_cell_out[0]

                dec_hidden_seq, score_seq = self.decode_train(label_var,
                                                              init_dec_hidden, init_dec_cell)

                label_var_for_loss = label_var.permute(1, 0) \
                    .contiguous().view(-1)

                loss = loss_function(score_seq, label_var_for_loss)
                loss_sum += loss.data.numpy()[0] / current_sen_len
                loss.backward()
                optimizer.step()
            avg_loss = loss_sum / instance_num
            print("epoch", epoch, ", loss =", avg_loss,
                  ", time =", time.time() - start_time)
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
        init_label_emb = self.label_embedding(
            Variable(torch.LongTensor(self.minibatch_size, 1).zero_() +
                     LABEL_BEGIN_INDEX)) \
            .view(self.minibatch_size, self.label_embedding_dim)
        dec_hidden_out, dec_cell_out = self.decoder_cell(
            init_label_emb, (init_dec_hidden, init_dec_cell))
        score = self.hidden2score(dec_hidden_out)
        logprob = nn.LogSoftmax(dim=1)(score) + seq_logprob
        topk_logprob, topk_label = torch.topk(logprob, beam_size, dim=1)
        seq_logprob = topk_logprob
        label_pred_seq.append(topk_label)

        # The rest parts of the sentence
        for i in range(seq_len - 1):
            prev_pred_label_emb = self.label_embedding(label_pred_seq[-1]) \
                .view(self.minibatch_size, self.label_embedding_dim)
            dec_hidden_out, dec_cell_out = self.decoder_cell(
                prev_pred_label_emb, (init_dec_hidden, init_dec_cell))
            score = self.hidden2score(dec_hidden_out)
            logprob = nn.LogSoftmax(dim=1)(score) + seq_logprob
            topk_logprob, topk_label = torch.topk(logprob, beam_size, dim=1)
            seq_logprob = topk_logprob
            label_pred_seq.append(topk_label)

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
        result_path = "../result/"

        f_sen = open(result_path + "sen.txt", 'w')
        f_pred = open(result_path + "pred.txt", 'w')
        f_label = open(result_path + "label.txt", 'w')
        f_sen_processed = open(result_path + "sen_processed.txt", 'w')
        f_pred_processed = open(result_path + "pred_processed.txt", 'w')
        f_label_processed = open(result_path + "label_processed.txt", 'w')

        for batch_idx in range(batch_num):
            sen = self.test_X[batch_idx]
            label = self.test_Y[batch_idx]
            current_sen_len = len(sen[0])

            # Always clear the gradients before use
            self.zero_grad()
            sen_var = Variable(torch.LongTensor(sen))
            label_var = Variable(torch.LongTensor(label))

            init_enc_hidden = Variable(torch.zeros((1, self.minibatch_size, self.hidden_dim)))
            init_enc_cell = Variable(torch.zeros((1, self.minibatch_size, self.hidden_dim)))

            enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encode(sen_var, init_enc_hidden, init_enc_cell)

            init_dec_hidden = enc_hidden_out[0]
            init_dec_cell = enc_cell_out[0]

            label_pred_seq = self.decode_greedy(current_sen_len, init_dec_hidden, init_dec_cell)

            label_pred_seq = [seq.data.numpy().squeeze() for seq in label_pred_seq]
            label_pred_seq = np.asarray(label_pred_seq).transpose().tolist()

            sen = list(itertools.chain.from_iterable(sen))
            label = list(itertools.chain.from_iterable(label))
            label_pred_seq = list(itertools.chain.from_iterable(label_pred_seq))
            assert len(sen) == len(label) and len(label) == len(label_pred_seq)

            for i in range(len(sen)):
                f_sen.write(str(sen[i]) + '\n')
                f_label.write(str(label[i]) + '\n')
                f_pred.write(str(label_pred_seq[i]) + '\n')

                # clean version
                if sen[i] != 0 and sen[i] != 2: # not <PAD> and not <EOS>
                    f_sen_processed.write(self.index2word[sen[i]] + '\n')
                    f_label_processed.write(self.index2label[label[i]] + '\n')
                    f_pred_processed.write(self.index2label[label_pred_seq[i]] + '\n')
                elif sen[i] == 2:   # <EOS>
                    f_sen_processed.write('\n')
                    f_label_processed.write('\n')
                    f_pred_processed.write('\n')

                def test(self):
                  batch_num = len(self.test_X)
                  result_path = "../result/"

                  f_sen = open(result_path + "sen.txt", 'w')
                  f_pred = open(result_path + "pred.txt", 'w')
                  f_label = open(result_path + "label.txt", 'w')
                  f_result_processed = open(
                    result_path + "result_processed.txt", 'w')

                  for batch_idx in range(batch_num):
                    sen = self.test_X[batch_idx]
                    label = self.test_Y[batch_idx]
                    current_sen_len = len(sen[0])

                    # Always clear the gradients before use
                    self.zero_grad()
                    sen_var = Variable(torch.LongTensor(sen))
                    label_var = Variable(torch.LongTensor(label))

                    init_enc_hidden = Variable(
                      torch.zeros((1, self.minibatch_size, self.hidden_dim)))
                    init_enc_cell = Variable(
                      torch.zeros((1, self.minibatch_size, self.hidden_dim)))

                    enc_hidden_seq, (
                    enc_hidden_out, enc_cell_out) = self.encode(sen_var,
                                                                init_enc_hidden,
                                                                init_enc_cell)

                    init_dec_hidden = enc_hidden_out[0]
                    init_dec_cell = enc_cell_out[0]

                    label_pred_seq = self.decode_greedy(current_sen_len,
                                                        init_dec_hidden,
                                                        init_dec_cell)

                    # write results to file
                    # each element in label_pred_seq is pytorch.Variable, thus convert to list first
                    label_pred_seq = [seq.data.numpy().squeeze() for seq in
                                      label_pred_seq]
                    label_pred_seq = np.asarray(
                      label_pred_seq).transpose().tolist()

                    # sen, label, label_pred_seq are list of lists,
                    # thus I would like to flatten them for iterating easier
                    sen = list(itertools.chain.from_iterable(sen))
                    label = list(itertools.chain.from_iterable(label))
                    label_pred_seq = list(
                      itertools.chain.from_iterable(label_pred_seq))
                    assert len(sen) == len(label) and len(label) == len(
                      label_pred_seq)

                    for i in range(len(sen)):
                      f_sen.write(str(sen[i]) + '\n')
                      f_label.write(str(label[i]) + '\n')
                      f_pred.write(str(label_pred_seq[i]) + '\n')

                      # clean version (does not print <PAD>, print a newline instead of <EOS>)
                      if sen[i] != 0 and sen[i] != 2:  # not <PAD> and not <EOS>
                        result_sen = self.index2word[sen[i]]
                        result_label = self.index2label[label[i]]
                        result_pred = self.index2label[label_pred_seq[i]]
                        f_result_processed.write("%s %s %s\n" % (
                        result_sen, result_label, result_pred))
                      elif sen[i] == 2:  # <EOS>
                        f_result_processed.write('\n')

    # just a copy of test() but use train data
    def eval_on_train(self):
        batch_num = len(self.train_X)
        result_path = "../result/"

        f_sen_train = open(result_path + "sen_train.txt", 'w')
        f_pred_train = open(result_path + "pred_train.txt", 'w')
        f_label_train = open(result_path + "label_train.txt", 'w')
        f_result_processed_train = open(result_path + "result_processed_train.txt", 'w')

        for batch_idx in range(batch_num):
            sen = self.train_X[batch_idx]
            label = self.train_Y[batch_idx]
            current_sen_len = len(sen[0])

            # Always clear the gradients before use
            self.zero_grad()
            sen_var = Variable(torch.LongTensor(sen))
            label_var = Variable(torch.LongTensor(label))

            init_enc_hidden = Variable(torch.zeros((1, self.minibatch_size, self.hidden_dim)))
            init_enc_cell = Variable(torch.zeros((1, self.minibatch_size, self.hidden_dim)))

            enc_hidden_seq, (enc_hidden_out, enc_cell_out) = self.encode(sen_var, init_enc_hidden, init_enc_cell)

            init_dec_hidden = enc_hidden_out[0]
            init_dec_cell = enc_cell_out[0]

            label_pred_seq = self.decode_greedy(current_sen_len, init_dec_hidden, init_dec_cell)

            label_pred_seq = [seq.data.numpy().squeeze() for seq in label_pred_seq]
            label_pred_seq = np.asarray(label_pred_seq).transpose().tolist()

            sen = list(itertools.chain.from_iterable(sen))
            label = list(itertools.chain.from_iterable(label))
            label_pred_seq = list(itertools.chain.from_iterable(label_pred_seq))
            assert len(sen) == len(label) and len(label) == len(label_pred_seq)

            for i in range(len(sen)):
                f_sen_train.write(str(sen[i]) + '\n')
                f_label_train.write(str(label[i]) + '\n')
                f_pred_train.write(str(label_pred_seq[i]) + '\n')

                # clean version
                if sen[i] != 0 and sen[i] != 2: # not <PAD> and not <EOS>
                    result_sen = self.index2word[sen[i]]
                    result_label = self.index2label[label[i]]
                    result_pred = self.index2label[label_pred_seq[i]]
                    f_result_processed_train.write("%s %s %s\n" % (result_sen, result_label, result_pred))

                elif sen[i] == 2:   # <EOS>
                    f_result_processed_train.write('\n')






