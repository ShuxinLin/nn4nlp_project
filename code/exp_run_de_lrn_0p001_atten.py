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

import torch


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
  dataset_path = '../dataset/German/'

  indexed_sentence_file = dataset_path + data + '.de-en.ids1.de'
  indexed_entity_file = dataset_path + data + '.de-en.ids1.en'
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

  data_path = "../dataset/German/"

  result_path = "../result_lrn_0p001_atten/"
  if not os.path.exists(result_path):
    os.makedirs(result_path)

  batch_size = 32

  dict_file = "../dataset/German/vocab1.de"
  entity_file = "../dataset/German/vocab1.en"
  index2word = get_index2word(dict_file)
  index2label = get_index2label(entity_file)
  vocab_size = len(index2word)
  label_size = len(index2label)
  #print("label_size=",label_size)

  train_X, train_Y = minibatch_de('train', batch_size)
  val_X, val_Y = minibatch_de('valid', batch_size)
  test_X, test_Y = minibatch_de('test', batch_size)

  # Using word2vec pre-trained embedding
  word_embedding_dim = 300

  hidden_dim = 64
  label_embedding_dim = 8

  max_epoch = 100

  # 0.001 is a good value
  learning_rate = 0.001

  attention = "fixed"

  pretrained = 'de64'

  if pretrained == 'de64':
    word_embedding_dim = 64

  gpu = True
  if gpu and rnd_seed:
    torch.cuda.manual_seed(rnd_seed)

  load_model_filename = None

  machine = ner(word_embedding_dim, hidden_dim, label_embedding_dim, vocab_size, label_size, learning_rate=learning_rate, minibatch_size=batch_size, max_epoch=max_epoch, train_X=train_X, train_Y=train_Y, val_X=val_X, val_Y=val_Y, test_X=test_X, test_Y=test_Y, attention=attention, gpu=gpu, pretrained=pretrained, load_model_filename=load_model_filename)
  if gpu:
    machine = machine.cuda()

  shuffle = True

  # Pure training, no evaluation
  train_loss_list = machine.train(shuffle, result_path, False, None)


  ##################

  eval_output_file_greedy = open(os.path.join(result_path, "eval_greedy.txt"), "w+")
  eval_output_file_beam_1 = open(os.path.join(result_path, "eval_beam_1.txt"), "w+")
  eval_output_file_beam_3 = open(os.path.join(result_path, "eval_beam_3.txt"), "w+")

  for epoch in range(0, max_epoch):
    load_model_filename = os.path.join(result_path, "ckpt_" + str(epoch) + ".pth")

    machine = ner(word_embedding_dim, hidden_dim, label_embedding_dim, vocab_size, label_size, learning_rate=learning_rate, minibatch_size=batch_size, max_epoch=max_epoch, train_X=train_X, train_Y=train_Y, val_X=val_X, val_Y=val_Y, test_X=test_X, test_Y=test_Y, attention=attention, gpu=gpu, pretrained=pretrained, load_model_filename=load_model_filename)
    if gpu:
      machine = machine.cuda()

    # "beam_size = 0" will use greedy
    # "beam_size = 1" will still use beam search, just with beam size = 1
    beam_size = 0

    train_loss_greedy, train_fscore_greedy = machine.evaluate(train_X, train_Y, index2word, index2label, "train", None, beam_size)
    val_loss_greedy, val_fscore_greedy = machine.evaluate(val_X, val_Y, index2word, index2label, "val", None, beam_size)

    time_begin_greedy = time.time()
    test_loss_greedy, test_fscore_greedy = machine.evaluate(test_X, test_Y, index2word, index2label, "test", None, beam_size)
    time_end_greedy = time.time()

    ###
    beam_size = 1

    train_loss_beam_1, train_fscore_beam_1 = machine.evaluate(train_X, train_Y, index2word, index2label, "train", None, beam_size)
    val_loss_beam_1, val_fscore_beam_1 = machine.evaluate(val_X, val_Y, index2word, index2label, "val", None, beam_size)

    time_begin_beam_1 = time.time()
    test_loss_beam_1, test_fscore_beam_1 = machine.evaluate(test_X, test_Y, index2word, index2label, "test", None, beam_size)
    time_end_beam_1 = time.time()


    ###
    beam_size = 3

    train_loss_beam_3, train_fscore_beam_3 = machine.evaluate(train_X, train_Y, index2word, index2label, "train", None, beam_size)
    val_loss_beam_3, val_fscore_beam_3 = machine.evaluate(val_X, val_Y, index2word, index2label, "val", None, beam_size)

    time_begin_beam_3 = time.time()
    test_loss_beam_3, test_fscore_beam_3 = machine.evaluate(test_X, test_Y, index2word, index2label, "test", None, beam_size)
    time_end_beam_3 = time.time()


    print("epoch %d" % epoch)
    print("Greedy\n"
          "training loss = %.6f" % train_loss_greedy,
          ", validation loss = %.6f" % val_loss_greedy,
          ", test loss = %.6f\n" % test_loss_greedy,
          "training F score = %.6f" % train_fscore_greedy,
          ", validation F score = %.6f" % val_fscore_greedy,
          ", test F score = %.6f\n" % test_fscore_greedy,
          "test time = %.6f" % (time_end_greedy - time_begin_greedy))
    print("Beam size 1\n"
          "training loss = %.6f" % train_loss_beam_1,
          ", validation loss = %.6f" % val_loss_beam_1,
          ", test loss = %.6f\n" % test_loss_beam_1,
          "training F score = %.6f" % train_fscore_beam_1,
          ", validation F score = %.6f" % val_fscore_beam_1,
          ", test F score = %.6f\n" % test_fscore_beam_1,
          "test time = %.6f" % (time_end_beam_1 - time_begin_beam_1))
    print("Beam size 3\n"
          "training loss = %.6f" % train_loss_beam_3,
          ", validation loss = %.6f" % val_loss_beam_3,
          ", test loss = %.6f\n" % test_loss_beam_3,
          "training F score = %.6f" % train_fscore_beam_3,
          ", validation F score = %.6f" % val_fscore_beam_3,
          ", test F score = %.6f\n" % test_fscore_beam_3,
          "test time = %.6f" % (time_end_beam_3 - time_begin_beam_3))

    eval_output_file_greedy.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epoch, train_loss_greedy, val_loss_greedy, test_loss_greedy, train_fscore_greedy, val_fscore_greedy, test_fscore_greedy, time_end_greedy - time_begin_greedy))
    eval_output_file_greedy.flush()

    eval_output_file_beam_1.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epoch, train_loss_beam_1, val_loss_beam_1, test_loss_beam_1, train_fscore_beam_1, val_fscore_beam_1, test_fscore_beam_1, time_end_beam_1 - time_begin_beam_1))
    eval_output_file_beam_1.flush()

    eval_output_file_beam_3.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epoch, train_loss_beam_3, val_loss_beam_3, test_loss_beam_3, train_fscore_beam_3, val_fscore_beam_3, test_fscore_beam_3, time_end_beam_3 - time_begin_beam_3))
    eval_output_file_beam_3.flush()
  # End for epoch

  eval_output_file_greedy.close()
  eval_output_file_beam_1.close()
  eval_output_file_beam_3.close()








  # Write out files
  #train_eval_loss, train_eval_fscore = machine.evaluate(train_X, train_Y, index2word, index2label, "train", result_path, beam_size)
  #val_eval_loss, val_eval_fscore = machine.evaluate(val_X, val_Y, index2word, index2label, "val", result_path, beam_size)
  #test_eval_loss, test_eval_fscore = machine.evaluate(test_X, test_Y, index2word, index2label, "test", result_path, beam_size)


if __name__ == "__main__":
  main()
