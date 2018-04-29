#!/usr/bin/python3

import numpy as np
import torch

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from operator import itemgetter
import collections

from ner import ner
from det_agent import det_agent

from data.toy_reverse_data_feeder import ToyDataFeeder
from data.toy_reverse_data import parse_data


def main():
  result_path = "../result_toy/"

  train = "../dataset/toy_reverse/train/"
  X_train, y_train = parse_data(train)
  train_ner_data = ToyDataFeeder(X_train, y_train)

  batch_size = 1
  val = "../dataset/toy_reverse/valid/"
  test = "../dataset/toy_reverse/test/"
  X_val, y_val = parse_data(val)
  X_test, y_test = parse_data(test)

  val_ner_data = ToyDataFeeder(X_val, y_val, word_to_idx=train_ner_data._word_to_idx, idx_to_word=train_ner_data._idx_to_word, label_to_idx=train_ner_data._label_to_idx, idx_to_label=train_ner_data._idx_to_label)
  val_X, val_Y, _ \
    = val_ner_data.naive_batch_buckets(batch_size)

  test_ner_data = ToyDataFeeder(X_test, y_test, word_to_idx=train_ner_data._word_to_idx, idx_to_word=train_ner_data._idx_to_word, label_to_idx=train_ner_data._label_to_idx, idx_to_label=train_ner_data._idx_to_label)
  test_X, test_Y, _ \
    = test_ner_data.naive_batch_buckets(batch_size)

  index2word = train_ner_data._idx_to_word
  index2label = train_ner_data._idx_to_label
  vocab_size = len(index2word)
  label_size = len(index2label)

  # Using word2vec pre-trained embedding
  # word_embedding_dim = 300
  word_embedding_dim = 8

  hidden_dim = 64
  label_embedding_dim = 8

  max_epoch = 50

  # 0.001 is a good value
  learning_rate = 0.001

  #attention = "fixed"
  attention = None

  #pretrained = 'de64'
  pretrained = None

  if pretrained == 'de64':
    word_embedding_dim = 64

  gpu = False


  ##################

  eval_output_file = open(os.path.join(result_path, "eval_beam_5_adapt_ckpt_24.txt"), "w+")

  epoch = 24

  load_model_filename = os.path.join(result_path, "ckpt_" + str(epoch) + ".pth")

  machine = ner(word_embedding_dim, hidden_dim, label_embedding_dim, vocab_size, label_size, learning_rate=learning_rate, minibatch_size=batch_size, max_epoch=max_epoch, train_X=None, train_Y=None, val_X=val_X, val_Y=val_Y, test_X=test_X, test_Y=test_Y, attention=attention, gpu=gpu, pretrained=pretrained, load_model_filename=load_model_filename, load_map_location=lambda storage, loc: storage)
  if gpu:
    machine = machine.cuda()

  decode_method = "adaptive"

  beam_size = 5
  max_beam_size = label_size

  accum_logP_ratio_low = 0.1
  logP_ratio_low = 0.1

  agent = det_agent(max_beam_size, accum_logP_ratio_low, logP_ratio_low)
  #agent = None

  # For German dataset, f_score_index_begin = 5 (because O_INDEX = 4)
  # For toy dataset, f_score_index_begin = 4 (because {0: '<s>', 1: '<e>', 2: '<p>', 3: '<u>', ...})
  f_score_index_begin = 4

  # We don't evaluate on training set simply because it is too slow since we can't use mini-batch in adaptive beam search
  val_fscore = machine.evaluate(val_X, val_Y, index2word, index2label, "val", None, decode_method, beam_size, max_beam_size, agent, f_score_index_begin)

  time_begin = time.time()
  test_fscore = machine.evaluate(test_X, test_Y, index2word, index2label, "test", None, decode_method, beam_size, max_beam_size, agent, f_score_index_begin)
  time_end = time.time()

  print_msg = "epoch %d, val F = %.6f, test F = %.6f, test time = %.6f" % (epoch, val_fscore, test_fscore, time_end - time_begin)
  log_msg = "%d\t%f\t%f\t%f" % (epoch, val_fscore, test_fscore, time_end - time_begin)
  print(print_msg)
  print(log_msg, file=eval_output_file, flush=True)

  eval_output_file.close()


  # Write out files
  #train_eval_loss, train_eval_fscore = machine.evaluate(train_X, train_Y, index2word, index2label, "train", result_path, beam_size)
  #val_eval_loss, val_eval_fscore = machine.evaluate(val_X, val_Y, index2word, index2label, "val", result_path, beam_size)
  #test_eval_loss, test_eval_fscore = machine.evaluate(test_X, test_Y, index2word, index2label, "test", result_path, beam_size)


if __name__ == "__main__":
  main()
