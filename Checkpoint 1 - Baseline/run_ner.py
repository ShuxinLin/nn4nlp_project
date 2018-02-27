#!/usr/bin/python3

from seq2seq_model import Seq2Seq
from data.ner_data import parse_data
from data.ner_data_feeder import NERData

batch_size = 3


def main():
  testa = "../dataset/CoNLL-2003/eng.testa"
  testb = "../dataset/CoNLL-2003/eng.testb"
  train = "../dataset/CoNLL-2003/eng.train"
  X_testa, y_testa = parse_data(testa)
  X_testb, y_testb = parse_data(testb)
  X_train, y_train = parse_data(train)

  train_ner_data = NERData(X_train, y_train)

  # get word to idx and apply to dev and test
  word_to_idx = train_ner_data.word_to_idx
  idx_to_word = train_ner_data.idx_to_word
  # print(word_to_idx)
  label_to_idx = train_ner_data.label_to_idx
  idx_to_label = train_ner_data.idx_to_label

  # print(label_to_idx)
  testa_ner_data = NERData(X_testa, y_testa, word_to_idx, idx_to_word)
  testb_ner_data = NERData(X_testb, y_testb, word_to_idx, idx_to_word)
  # print(testa_ner_data.word_to_idx)

  batch_size = 3
  X_train_batch, y_train_batch = train_ner_data.naive_batch(batch_size)
  X_testa_batch, y_testa_batch = testa_ner_data.naive_batch(batch_size)
  X_testb_batch, y_testb_batch = testb_ner_data.naive_batch(batch_size)

  # # import pdb; pdb.set_trace()

  word_embedding_dim = 50
  hidden_dim = 10
  label_embedding_dim = 10

  machine = Seq2Seq(word_embedding_dim, hidden_dim, label_embedding_dim,
                    len(word_to_idx), len(label_to_idx), learning_rate=0.1,
                    minibatch_size=batch_size, max_epoch=1,
                    index2word=idx_to_word, index2label=idx_to_label,
                    train_X=X_train_batch, train_Y=y_train_batch,
                    test_X=X_testa_batch, test_Y=y_testa_batch)

  machine.train()
  machine.test()


if __name__ == "__main__":
  main()
