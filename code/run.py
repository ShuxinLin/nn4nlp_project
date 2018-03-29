#!/usr/bin/python3

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from operator import itemgetter
import collections
from ner import ner
from preprocessor import Preprocessor

def prepocess(data_path, train, val, batch_size):
  train_preprocessor = Preprocessor(data_path, train)
  train_preprocessor.read_file()
  train_preprocessor.preprocess()

  val_preprocessor = Preprocessor(data_path, val)
  val_preprocessor.read_file()
  val_preprocessor.preprocess()

  load_vocab(data_path + "vocab_dict", train_preprocessor, val_preprocessor)
  # build_vocab(data_path, train_preprocessor, val_preprocessor)

  train_preprocessor.index_preprocess()
  train_X, train_Y = train_preprocessor.minibatch(batch_size)
  val_preprocessor.index_preprocess()
  val_X, val_Y = val_preprocessor.minibatch(batch_size)
  vocab_size = train_preprocessor.vocabulary_size
  label_size = train_preprocessor.entity_dict_size

  entity_dict = train_preprocessor.entity_dict

  return train_X, train_Y, val_X, val_Y, vocab_size, label_size, entity_dict


def load_vocab(vocab_path, train_preprocessor, val_preprocessor):
    vocab_dict = dict()
    with open(vocab_path) as f:
        for line in f:
            splitted = line.split('\t')
            vocab_dict[splitted[0]] = int(splitted[1])

    train_preprocessor.vocab_dict = val_preprocessor.vocab_dict = vocab_dict
    train_preprocessor.vocabulary_size = val_preprocessor.vocabulary_size = len(vocab_dict)
    print("Loaded the existing vocabulary dictionary. vocab_size: ", len(vocab_dict))


def build_vocab(data_path, train_preprocessor, val_preprocessor):
  all_text = []
  for sentence in train_preprocessor.new_data['SENTENCE']:
    all_text.extend(sentence.split())
  for sentence in val_preprocessor.new_data['SENTENCE']:
    all_text.extend(sentence.split())
  #all_text = filter(lambda a: a not in [train_preprocessor.EOS_TOKEN, train_preprocessor.PAD_TOKEN], all_text)
  all_text = filter(lambda a: a != train_preprocessor.PAD_TOKEN, all_text)
  all_words = collections.Counter(all_text).most_common()

  sorted_by_name = sorted(all_words, key=lambda x: x[0])
  all_words = sorted(sorted_by_name, key=lambda x: x[1], reverse=True)
  #tokens = [(train_preprocessor.PAD_TOKEN, -1), (train_preprocessor.UNK_TOKEN, -1), (train_preprocessor.EOS_TOKEN, -1)]
  tokens = [(train_preprocessor.PAD_TOKEN, -1), (train_preprocessor.UNK_TOKEN, -1)]
  all_words = tokens + all_words

  vocab_dict = dict()
  for word in all_words:
    if word[0] not in vocab_dict:
      vocab_dict[word[0]] = len(vocab_dict)
  vocabulary_size = len(vocab_dict)
  train_preprocessor.vocab_dict = val_preprocessor.vocab_dict = vocab_dict
  train_preprocessor.vocabulary_size = val_preprocessor.vocabulary_size = vocabulary_size

  vocab_file = data_path + "vocab_dict"
  with open(vocab_file, 'w') as f:
    for word in all_words:
      f.write("%s\t%d\n" % (word[0], vocab_dict[word[0]]))
  print('Saved vocabulary to vocabulary file. vocab_size: ', vocabulary_size)

def get_index2word(dict_file):
  index2word = dict()
  with open(dict_file) as f:
    for line in f:
      (word, index) = line.split()
      index2word[int(index)] = word

  return index2word

def get_index2label(entity_dict):
  index2label = dict()
  for entity, index in entity_dict.items():
    index2label[int(index)] = entity

  return index2label

def main():
  data_path = "../dataset/CoNLL-2003/"
  train_file = "eng.train"
  val_file = "eng.testa"

  #train_file = "eng.testa.nano.txt"
  #val_file = "eng.testa.nano.txt"

  #test_file = "eng.testb"

  result_path = "../result/"

  batch_size = 32

  train_X, train_Y, val_X, val_Y, vocab_size, label_size, entity_dict = prepocess(data_path, train_file, val_file, batch_size)

  dict_file = "../dataset/CoNLL-2003/vocab_dict"
  index2word = get_index2word(dict_file)
  index2label = get_index2label(entity_dict)

  # Using word2vec pre-trained embedding
  word_embedding_dim = 300

  #hidden_dim = 64
  hidden_dim = 6
  label_embedding_dim = 8

  max_epoch = 2

  # 0.001 is a good value
  learning_rate = 0.001

  attention = "fixed"
  #attention = None

  pretrained = 'glove'

  gpu = True

  machine = ner(word_embedding_dim, hidden_dim, label_embedding_dim, vocab_size, label_size, learning_rate=learning_rate, minibatch_size=32, max_epoch=max_epoch, train_X=train_X, train_Y=train_Y, test_X=val_X, test_Y=val_Y, attention=attention, gpu=gpu, pretrained=pretrained)
  if gpu:
    machine = machine.cuda()

  # "beam_size = 0" will use greedy
  # "beam_size = 1" will still use beam search, just with beam size = 1
  beam_size = 3

  shuffle = True

  train_loss_list = machine.train(shuffle)
  machine.evaluate(train_X, train_Y, index2word, index2label, "train", beam_size)
  machine.evaluate(val_X, val_Y, index2word, index2label, "val", beam_size)

  #print(train_loss_list)

  plt.figure(1)
  plt.plot(list(range(len(train_loss_list))) , train_loss_list, "k-")
  #plt.xlim([0, 11])
  #plt.ylim([0, 0.5])
  plt.xlabel("Epoch")
  plt.ylabel("Cross-entropy loss")
  plt.savefig(result_path + "fig_exp1.pdf")

if __name__ == "__main__":
  main()
