#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from operator import itemgetter
import collections
from ner import ner
import os
import time


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

"""
def construct_df(set_name):
  with open(os.path.join("../dataset/toy_reverse", set_name, "data.txt")) as f:
    for line in f:
      res = line.split("\t")
      x_seq = res[0].split(" ")
      y_seq = res[1].split(" ")
      print("x_seq=",x_seq)
      print("y_seq=",y_seq)
      time.sleep(1)

  indexed_sentence_file = dataset_path + data + '.de-en.ids1.de'
  indexed_entity_file = dataset_path + data + '.de-en.ids1.en'
  with open(indexed_sentence_file) as f:
    indexed_sentences = f.readlines()
  with open(indexed_entity_file) as f:
    indexed_entities = f.readlines()
  df = pd.DataFrame({'SENTENCE': indexed_sentences, 'ENTITY': indexed_entities})

  return df
"""

def minibatch_de(set_name, batch_size):
  print("Generate mini batches.")

  X_batch = []
  Y_batch = []
  all_data = []

  with open(os.path.join("../dataset/toy_reverse", set_name, "data.txt")) as f:
    for line in f:
      res = line.strip().split("\t")
      x_seq = list(map(int, res[0].split(" ")))
      y_seq = list(map(int, res[1].split(" ")))
      all_data.append((len(x_seq), x_seq, y_seq))

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
  #data_path = "../dataset/German/"

  result_path = "../result_3/"
  if not os.path.exists(result_path):
    os.makedirs(result_path)

  batch_size = 32

  dict_file = "../dataset/toy_reverse/vocab.src"
  entity_file = "../dataset/toy_reverse/vocab.tgt"
  index2word = get_index2word(dict_file)
  index2label = get_index2label(entity_file)
  vocab_size = len(index2word)
  label_size = len(index2label)
  #print("label_size=",label_size)

  train_X, train_Y = minibatch_de('train', batch_size)
  val_X, val_Y = minibatch_de('valid', batch_size)
  test_X, test_Y = minibatch_de('test', batch_size)

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
