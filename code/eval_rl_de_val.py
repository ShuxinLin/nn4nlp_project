#!/usr/bin/python3

import argparse
import os
from operator import itemgetter

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from model import AdaptiveActorCritic
from ner import ner
from optim import SharedAdam
from rl_trainer import train_adaptive, eval_adaptive

matplotlib.use("Agg")


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
    all_data.append(
      (len(splitted_sentence), splitted_sentence, splitted_entities))

  sorted_all_data = sorted(all_data, key=itemgetter(0))
  prev_len = 1
  X_minibatch = []
  Y_minibatch = []
  for data in sorted_all_data:
    if prev_len == data[0]:
      X_minibatch.append(data[1])
      Y_minibatch.append(data[2])
    else:
      X_minibatch = [X_minibatch[x:x + batch_size] for x in
                     range(0, len(X_minibatch), batch_size)]
      Y_minibatch = [Y_minibatch[x:x + batch_size] for x in
                     range(0, len(Y_minibatch), batch_size)]
      X_batch.extend(X_minibatch)
      Y_batch.extend(Y_minibatch)
      X_minibatch = []
      Y_minibatch = []
      X_minibatch.append(data[1])
      Y_minibatch.append(data[2])
      prev_len = data[0]
  X_minibatch = [X_minibatch[x:x + batch_size] for x in
                 range(0, len(X_minibatch), batch_size)]
  Y_minibatch = [Y_minibatch[x:x + batch_size] for x in
                 range(0, len(Y_minibatch), batch_size)]
  X_batch.extend(X_minibatch)
  Y_batch.extend(Y_minibatch)
  assert len(X_batch) == len(Y_batch)

  return list(X_batch), list(Y_batch)


def minibatch_of_one_de(data):
  print("Generate mini batches, each with only 1 instance.")
  all_data = []
  indexed_data = construct_df(data)
  for index, row in indexed_data.iterrows():
    splitted_sentence = list(map(int, row['SENTENCE'].split()))
    splitted_entities = list(map(int, row['ENTITY'].split()))
    assert len(splitted_entities) == len(splitted_sentence)
    all_data.append(
      (len(splitted_sentence), splitted_sentence, splitted_entities))

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


  # ---------------------------------------
  #           DATA LOADING
  # ---------------------------------------
  #result_path = "../result_lrn_0p001_rl/"

  dict_file = "../dataset/German/vocab1.de"
  entity_file = "../dataset/German/vocab1.en"
  index2word = get_index2word(dict_file)
  index2label = get_index2label(entity_file)
  vocab_size = len(index2word)
  label_size = len(index2label)

  train_X, train_Y = minibatch_of_one_de('train')
  val_X, val_Y = minibatch_of_one_de('valid')
  test_X, test_Y = minibatch_of_one_de('test')

  # ---------------------------------------
  #           HYPER PARAMETERS
  # ---------------------------------------
  # Using word2vec pre-trained embedding
  #word_embedding_dim = 300

  hidden_dim = 64
  label_embedding_dim = 8
  max_epoch = 100
  # 0.001 is a good value
  ner_learning_rate = 0.001

  pretrained = 'de64'
  word_embedding_dim = 64

  # ---------------------------------------
  #           GPU OR NOT?
  # ---------------------------------------
  gpu = False
  if gpu and rnd_seed:
    torch.cuda.manual_seed(rnd_seed)

  # ---------------------------------------
  #        MODEL INSTANTIATION
  # ---------------------------------------
  #attention = None
  attention = "fixed"

  load_model_dir = "../result_lrn_0p001_atten/"
  load_model_filename = os.path.join(load_model_dir, "ckpt_46.pth")

  batch_size = 1
  machine = ner(word_embedding_dim, hidden_dim, label_embedding_dim, vocab_size,
                label_size, learning_rate=ner_learning_rate,
                minibatch_size=batch_size, max_epoch=max_epoch, train_X=None,
                train_Y=None, val_X=val_X, val_Y=val_Y, test_X=test_X,
                test_Y=test_Y, attention=attention, gpu=gpu,
                pretrained=pretrained, load_model_filename=load_model_filename,
                load_map_location="cpu")
  if gpu:
    machine = machine.cuda()

  initial_beam_size = 3
  # When you have only one beam, it does not make sense to consider
  # max_beam_size larger than the size of your label vocabulary
  max_beam_size = label_size

  # ============   INIT RL =====================
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['CUDA_VISIBLE_DEVICES'] = ""


  parser = argparse.ArgumentParser(description='A3C')

  parser.add_argument('--logdir', default='../result_lrn_0p001_atten_rl',
                      help='name of logging directory')
  parser.add_argument('--lr', type=float, default=0.0001,
                      help='learning rate (default: 0.0001)')
  parser.add_argument('--gamma', type=float, default=0.99,
                      help='discount factor for rewards (default: 0.99)')
  parser.add_argument('--n_epochs', type=int, default=2,
                      help='number of epochs for training agent(default: 30)')
  parser.add_argument('--entropy-coef', type=float, default=0.01,
                      help='entropy term coefficient (default: 0.01)')
  parser.add_argument('--num-processes', type=int, default=2,
                      help='how many training processes to use (default: 4)')
  parser.add_argument('--num-steps', type=int, default=20,
                      help='number of forward steps in A3C (default: 20)')

  parser.add_argument('--tau', type=float, default=1.00,
                      help='parameter for GAE (default: 1.00)')
  parser.add_argument('--value-loss-coef', type=float, default=0.5,
                      help='value loss coefficient (default: 0.5)')
  parser.add_argument('--max-grad-norm', type=float, default=5,
                      help='value loss coefficient (default: 5)')
  parser.add_argument('--seed', type=int, default=1,
                      help='random seed (default: 1)')
  parser.add_argument('--max-episode-length', type=int, default=1000000,
                      help='maximum length of an episode (default: 1000000)')
  parser.add_argument('--name', default='train',
                      help='name of the process')
  parser.add_argument('--no-shared', default=False,
                      help='use an optimizer without shared momentum.')
  args = parser.parse_args()

  if not os.path.exists(args.logdir):
    os.mkdir(args.logdir)

  # For German dataset, f_score_index_begin = 5 (because O_INDEX = 4)
  # For toy dataset, f_score_index_begin = 4 (because {0: '<s>', 1: '<e>', 2: '<p>', 3: '<u>', ...})
  f_score_index_begin = 5
  # RL reward coefficient
  reward_coef_fscore = 1
  reward_coef_beam_size = 0.1

  logfile = open(os.path.join(args.logdir, "eval_val.txt"), "w+")

  model = AdaptiveActorCritic(max_beam_size=max_beam_size, action_space=3)
  # Marking as for evaluation
  model.eval()

  load_map_location = "cpu"

  for epoch in range(0, args.n_epochs):
    load_model_filename = os.path.join(args.logdir, "ckpt_" + str(epoch) + ".pth")
    checkpoint = torch.load(load_model_filename, map_location=load_map_location)
    model.load_state_dict(checkpoint["state_dict"])

    fscore, total_beam_number_in_dataset, avg_beam_size, time_used = \
      eval_adaptive(machine,
                    max_beam_size,
                    model,
                    val_X, val_Y, index2word, index2label,
                    "val", False, "adaptive", initial_beam_size,
                    reward_coef_fscore, reward_coef_beam_size,
                    f_score_index_begin,
                    args)

    log_msg = "%d\t%f\t%d\t%f\t%f" % (epoch, fscore, total_beam_number_in_dataset, avg_beam_size, time_used)
    print(log_msg)
    print(log_msg, file=logfile, flush=True)
  # End for epoch

  logfile.close()


if __name__ == "__main__":
  main()
