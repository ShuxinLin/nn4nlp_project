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

from data.toy_reverse_data_feeder import ToyDataFeeder
from data.toy_reverse_data import parse_data


def main():
  #data_path = "../dataset/German/"

  result_path = "../result/"
  if not os.path.exists(result_path):
    os.makedirs(result_path)

  batch_size = 32

  test = '../dataset/toy_reverse/test/'
  valid = '../dataset/toy_reverse/valid/'
  train = '../dataset/toy_reverse/train/'

  X_test, y_test = parse_data(test)
  X_valid, y_valid = parse_data(valid)
  X_train, y_train = parse_data(train)

  train_ner_data = ToyDataFeeder(X_train, y_train)
  train_X, train_Y, _ \
    = train_ner_data.naive_batch_buckets(batch_size)

  val_ner_data = ToyDataFeeder(X_valid, y_valid, word_to_idx=train_ner_data._word_to_idx, idx_to_word=train_ner_data._idx_to_word,
               label_to_idx=train_ner_data._label_to_idx,
               idx_to_label=train_ner_data._idx_to_label)
  val_X, val_Y, _ \
    = val_ner_data.naive_batch_buckets(batch_size)

  test_ner_data = ToyDataFeeder(X_test, y_test, word_to_idx=train_ner_data._word_to_idx, idx_to_word=train_ner_data._idx_to_word,
               label_to_idx=train_ner_data._label_to_idx,
               idx_to_label=train_ner_data._idx_to_label)
  test_X, test_Y, _ \
    = test_ner_data.naive_batch_buckets(batch_size)

  #print("train_ner_data._idx_to_word=",train_ner_data._idx_to_word)
  #print("val_ner_data._idx_to_word=",val_ner_data._idx_to_word)

  """
  print("len(test_X)=",len(test_X))
  print("test_X[0]=",test_X[0])
  print("test_Y[0]=",test_Y[0])
  """

  #dict_file = "../dataset/toy_reverse/vocab.src"
  #entity_file = "../dataset/toy_reverse/vocab.tgt"

  index2word = train_ner_data._idx_to_word
  index2label = train_ner_data._idx_to_label

  #index2word = get_index2word(dict_file)
  #index2label = get_index2label(entity_file)
  vocab_size = len(index2word)
  label_size = len(index2label)
  #print("label_size=",label_size)

  #train_X, train_Y = minibatch_de('train', batch_size)
  #val_X, val_Y = minibatch_de('valid', batch_size)
  #test_X, test_Y = minibatch_de('test', batch_size)

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

  machine = ner(word_embedding_dim, hidden_dim, label_embedding_dim, vocab_size, label_size, learning_rate=learning_rate, minibatch_size=batch_size, max_epoch=max_epoch, train_X=train_X, train_Y=train_Y, val_X=val_X, val_Y=val_Y, test_X=test_X, test_Y=test_Y, attention=attention, gpu=gpu, pretrained=pretrained)
  if gpu:
    machine = machine.cuda()

  # "beam_size = 0" will use greedy
  # "beam_size = 1" will still use beam search, just with beam size = 1
  beam_size = 3

  shuffle = True

  train_loss_list = machine.train(shuffle, beam_size, result_path)
  # Write out files
  train_eval_loss, train_eval_fscore = machine.evaluate(train_X, train_Y, index2word, index2label, "train", result_path, beam_size)
  val_eval_loss, val_eval_fscore = machine.evaluate(val_X, val_Y, index2word, index2label, "val", result_path, beam_size)
  test_eval_loss, test_eval_fscore = machine.evaluate(test_X, test_Y, index2word, index2label, "test", result_path, beam_size)

  #print("train_eval_loss =", train_eval_loss)
  #print("val_eval_loss =", val_eval_loss)

  #print(train_loss_list)

  """
  plt.figure(1)
  plt.plot(list(range(len(train_loss_list))) , train_loss_list, "k-")
  #plt.xlim([0, 11])
  #plt.ylim([0, 0.5])
  plt.xlabel("Epoch")
  plt.ylabel("Cross-entropy loss")
  plt.savefig(result_path + "fig_exp1.pdf")
  """

if __name__ == "__main__":
  main()
