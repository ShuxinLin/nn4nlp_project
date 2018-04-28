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
  result_path = "../result_toy/"
  if not os.path.exists(result_path):
    os.makedirs(result_path)

  batch_size = 32

  train = '../dataset/toy_reverse/train/'

  X_train, y_train = parse_data(train)

  train_ner_data = ToyDataFeeder(X_train, y_train)
  train_X, train_Y, _ \
    = train_ner_data.naive_batch_buckets(batch_size)

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

  gpu = True

  load_model_filename = None

  machine = ner(word_embedding_dim, hidden_dim, label_embedding_dim, vocab_size, label_size, learning_rate=learning_rate, minibatch_size=batch_size, max_epoch=max_epoch, train_X=train_X, train_Y=train_Y, val_X=None, val_Y=None, test_X=None, test_Y=None, attention=attention, gpu=gpu, pretrained=pretrained, load_model_filename=load_model_filename)
  if gpu:
    machine = machine.cuda()

  shuffle = True

  # Pure training, no evaluation
  train_loss_list = machine.train(shuffle, result_path, False, None)


if __name__ == "__main__":
  main()
