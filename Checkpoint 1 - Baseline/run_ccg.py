#!/usr/bin/python3

from seq2seq_model import Seq2Seq
from data.ccg_data import parse_data
from data.ccg_data_feeder import CCGData

batch_size = 3


def main():
  test = "../dataset/supertag_data/test.dat"
  dev = "../dataset/supertag_data/dev.dat"
  train = "../dataset/supertag_data/train.dat"
  X_test, y_test, _ = parse_data(test)
  X_dev, y_dev, _ = parse_data(dev)
  X_train, y_train, _ = parse_data(train)

  train_ccg_data = CCGData(X_train, y_train)

  # get word to idx and apply to dev and test
  word_to_idx = train_ccg_data.word_to_idx
  idx_to_word = train_ccg_data.idx_to_word
  # print(word_to_idx)
  print(idx_to_word)

  label_to_idx = train_ccg_data.label_to_idx
  idx_to_label = train_ccg_data.idx_to_label
  # print(label_to_idx)
  print(idx_to_label)

  dev_ner_data = CCGData(X_dev, y_dev, word_to_idx, label_to_idx,
                         idx_to_word, idx_to_label)
  testa_ner_data = CCGData(X_test, y_test, word_to_idx, label_to_idx,
                           idx_to_word, idx_to_label)
  # print(testa_ner_data.word_to_idx)
  print(testa_ner_data.idx_to_word)

  batch_size = 3
  X_train_batch, y_train_batch = train_ccg_data.naive_batch(batch_size)
  X_dev_batch, y_dev_batch = dev_ner_data.naive_batch(batch_size)
  X_test_batch, y_test_batch = testa_ner_data.naive_batch(batch_size)

  # import pdb; pdb.set_trace()

  word_embedding_dim = 50
  hidden_dim = 10
  label_embedding_dim = 10

  machine = Seq2Seq(word_embedding_dim, hidden_dim, label_embedding_dim,
                    len(word_to_idx), len(label_to_idx), learning_rate=0.1,
                    minibatch_size=batch_size, max_epoch=3,
                    index2word=idx_to_word, index2label=idx_to_label,
                    train_X=X_train_batch, train_Y=y_train_batch,
                    test_X=X_dev_batch, test_Y=y_dev_batch)

  machine.train()
  machine.eval_on_train()
  machine.test()


if __name__ == "__main__":
  main()
