#!/usr/bin/python3

from ner import NER
from data.ner_data import parse_data
from data.ner_data_feeder import get_word2idx, naive_batch, label_to_idx

batch_size = 3


def main():
  testa = "../../dataset/CoNLL-2003/eng.testa"
  testb = "../../dataset/CoNLL-2003/eng.testb"
  train = "../../dataset/CoNLL-2003/eng.train"
  X_testa, y_testa = parse_data(testa)
  X_testb, y_testb = parse_data(testb)
  X_train, y_train = parse_data(train)

  word_to_idx = get_word2idx([X_train, X_testa, X_testb])
  print word_to_idx
  print(len(word_to_idx))

  X_train_batch, y_train_batch = naive_batch(X_train, y_train,
                                             batch_size, word_to_idx)
  X_testa_batch, y_testa_batch = naive_batch(X_testa, y_testa,
                                             batch_size, word_to_idx)
  # import pdb; pdb.set_trace()

  word_embedding_dim = 50
  hidden_dim = 10
  label_embedding_dim = 10

  machine = NER(word_embedding_dim, hidden_dim, label_embedding_dim,
                len(word_to_idx), len(label_to_idx), learning_rate=0.1,
                minibatch_size=3, max_epoch=3,
                train_X=X_train_batch, train_Y=y_train_batch,
                test_X=X_testa_batch, test_Y=y_testa_batch)

  machine.train()
  machine.test()


if __name__ == "__main__":
  main()
