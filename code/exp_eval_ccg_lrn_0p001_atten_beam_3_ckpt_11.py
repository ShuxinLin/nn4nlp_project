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
from det_agent import det_agent


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


def minibatch_of_one_de(data):
  print("Generate mini batches, each with only 1 instance.")
  X_batch = []
  Y_batch = []
  all_data = []
  indexed_data = construct_df(data)
  for index, row in indexed_data.iterrows():
    splitted_sentence = list(map(int, row['SENTENCE'].split()))
    splitted_entities = list(map(int, row['ENTITY'].split()))
    assert len(splitted_entities) == len(splitted_sentence)
    all_data.append((len(splitted_sentence), splitted_sentence, splitted_entities))

  # Does not have to sort if each minibatch has only 1 instance

  X_batch = [[data[1]] for data in all_data]
  Y_batch = [[data[2]] for data in all_data]
  assert len(X_batch) == len(Y_batch)

  return list(X_batch), list(Y_batch)


def main():
  rnd_seed = None
  if rnd_seed:
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)

  result_path = "../result_ccg_lrn_0p001_atten/"

  dict_file = "../dataset/CCGbank/dict_word"
  entity_file = "../dataset/CCGbank/dict_tag"
  index2word = get_index2word(dict_file)
  index2label = get_index2label(entity_file)
  vocab_size = len(index2word)
  label_size = len(index2label)

  val_X, val_Y = minibatch_of_one_de('val')
  test_X, test_Y = minibatch_of_one_de('test')

  # Using word2vec pre-trained embedding
  word_embedding_dim = 300

  hidden_dim = 512
  label_embedding_dim = 512

  max_epoch = 30

  # 0.001 is a good value
  learning_rate = 0.001

  attention = "fixed"
  #attention = None

  pretrained = None

  gpu = True
  if gpu and rnd_seed:
    torch.cuda.manual_seed(rnd_seed)
  gpu_no = 2
  cuda_dev = torch.device("cuda:" + str(gpu_no))

  ##################

  os.environ['OMP_NUM_THREADS'] = '1'

  eval_output_file = open(os.path.join(result_path, "eval_beam_3_ckpt_11.txt"), "w+")

  for epoch in range(11, 12):
    load_model_filename = os.path.join(result_path, "ckpt_" + str(epoch) + ".pth")
    batch_size = 1

    machine = ner(word_embedding_dim, hidden_dim, label_embedding_dim, vocab_size, label_size, learning_rate=learning_rate, minibatch_size=batch_size, max_epoch=max_epoch, train_X=None, train_Y=None, val_X=val_X, val_Y=val_Y, test_X=test_X, test_Y=test_Y, attention=attention, gpu=gpu, gpu_no=gpu_no, pretrained=pretrained, load_model_filename=load_model_filename)
    if gpu:
      machine = machine.cuda(cuda_dev)

    decode_method = "beam"

    beam_size = 3
    max_beam_size = label_size

    accum_logP_ratio_low = 0.1
    logP_ratio_low = 0.1

    #agent = det_agent(max_beam_size, accum_logP_ratio_low, logP_ratio_low)
    agent = None

    # For German dataset, f_score_index_begin = 5 (because O_INDEX = 4)
    # For toy dataset, f_score_index_begin = 4 (because {0: '<s>', 1: '<e>', 2: '<p>', 3: '<u>', ...})
    # For CCG dataset, f_score_index_begin = 2 (because {0: _PAD, 1: _SOS, ...})
    f_score_index_begin = 2

    reward_coef_fscore = 1
    reward_coef_beam_size = 0.02

    # We don't evaluate on training set simply because it is too slow since we can't use mini-batch in adaptive beam search
    val_fscore, val_beam_number, val_avg_beam_size = machine.evaluate(val_X, val_Y, index2word, index2label, "val", None, decode_method, beam_size, max_beam_size, agent, reward_coef_fscore, reward_coef_beam_size, f_score_index_begin, generate_episode=False, episode_save_path=None)

    time_begin = time.time()
    test_fscore, test_beam_number, test_avg_beam_size = machine.evaluate(test_X, test_Y, index2word, index2label, "test", None, decode_method, beam_size, max_beam_size, agent, reward_coef_fscore, reward_coef_beam_size, f_score_index_begin, generate_episode=False, episode_save_path=None)
    time_end = time.time()

    print_msg = "epoch %d, val F = %.6f, test F = %.6f, test time = %.6f" % (epoch, val_fscore, test_fscore, time_end - time_begin)
    log_msg = "%d\t%f\t%f\t%d\t%d\t%f\t%f\t%f" % (epoch, val_fscore, test_fscore, val_beam_number, test_beam_number, val_avg_beam_size, test_avg_beam_size, time_end - time_begin)
    print(print_msg)
    print(log_msg, file=eval_output_file, flush=True)
  # End for epoch

  eval_output_file.close()


if __name__ == "__main__":
  main()
