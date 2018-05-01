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

from ner_multigpu import ner


def get_index2word(dict_file):
  index2word = dict()
  with open(dict_file) as f:
    for line in f:
      (word, index) = line.split()
      index2word[int(index)] = word

  return index2word

def get_index2label(entity_file):
  index2label = dict()
  with open(entity_file) as f:
    for line in f:
      (entity, index) = line.split()
      index2label[int(index)] = entity
  return index2label

def construct_df(data):
  dataset_path = '../dataset/CCGbank/'

  indexed_sentence_file = dataset_path + data + '_x'
  indexed_entity_file = dataset_path + data + '_y'
  with open(indexed_sentence_file) as f:
    indexed_sentences = f.readlines()
  with open(indexed_entity_file) as f:
    indexed_entities = f.readlines()
  df = pd.DataFrame({'SENTENCE': indexed_sentences, 'ENTITY': indexed_entities})

  return df

def minibatch_de(data, batch_size):
  print("Generate mini batches.")
  X_batch = []
  Y_batch = []
  all_data = []
  indexed_data = construct_df(data)
  for index, row in indexed_data.iterrows():
    splitted_sentence = list(map(int, row['SENTENCE'].split()))
    splitted_entities = list(map(int, row['ENTITY'].split()))
    assert len(splitted_entities) == len(splitted_sentence)
    all_data.append((len(splitted_sentence), splitted_sentence, splitted_entities))

  sorted_all_data = sorted(all_data, key=itemgetter(0))
  prev_len = 1
  X_minibatch = []
  Y_minibatch = []
  for data in sorted_all_data:
    if prev_len == data[0]:
      X_minibatch.append(data[1])
      Y_minibatch.append(data[2])
    else:
      X_minibatch = [X_minibatch[x:x + batch_size] for x in range(0, len(X_minibatch), batch_size)]
      Y_minibatch = [Y_minibatch[x:x + batch_size] for x in range(0, len(Y_minibatch), batch_size)]
      X_batch.extend(X_minibatch)
      Y_batch.extend(Y_minibatch)
      X_minibatch = []
      Y_minibatch = []
      X_minibatch.append(data[1])
      Y_minibatch.append(data[2])
      prev_len = data[0]
  X_minibatch = [X_minibatch[x:x + batch_size] for x in range(0, len(X_minibatch), batch_size)]
  Y_minibatch = [Y_minibatch[x:x + batch_size] for x in range(0, len(Y_minibatch), batch_size)]
  X_batch.extend(X_minibatch)
  Y_batch.extend(Y_minibatch)
  assert len(X_batch) == len(Y_batch)

  return list(X_batch), list(Y_batch)


def main():
  rnd_seed = None
  if rnd_seed:
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)

  #data_path = "../dataset/German/"

  result_path = "../result_ccg_lrn_0p001_atten/"
  if not os.path.exists(result_path):
    os.makedirs(result_path)

  batch_size = 32

  dict_file = "../dataset/CCGbank/dict_word"
  entity_file = "../dataset/CCGbank/dict_tag"
  index2word = get_index2word(dict_file)
  index2label = get_index2label(entity_file)
  vocab_size = len(index2word)
  label_size = len(index2label)
  #print("label_size=",label_size)

  train_X, train_Y = minibatch_de('train', batch_size)

  # Using word2vec pre-trained embedding
  word_embedding_dim = 300

  hidden_dim = 512
  label_embedding_dim = 512

  max_epoch = 100

  # 0.001 is a good value
  learning_rate = 0.001

  attention = "fixed"
  #attention = None

  pretrained = None

  if pretrained == 'de64':
    word_embedding_dim = 64

  gpu = True
  if gpu and rnd_seed:
    torch.cuda.manual_seed(rnd_seed)
  gpu_no = 2
  cuda = torch.device("cuda:" + str(gpu_no))

  load_model_filename = None

  machine = ner(word_embedding_dim, hidden_dim, label_embedding_dim, vocab_size, label_size, learning_rate=learning_rate, minibatch_size=batch_size, max_epoch=max_epoch, train_X=train_X, train_Y=train_Y, val_X=None, val_Y=None, test_X=None, test_Y=None, attention=attention, gpu=gpu, gpu_no=gpu_no, pretrained=pretrained, load_model_filename=load_model_filename)
  if gpu:
    machine = machine.cuda(cuda)

  shuffle = True

  # Pure training, no evaluation
  train_loss_list = machine.train(shuffle, result_path, False, None)


if __name__ == "__main__":
  main()
