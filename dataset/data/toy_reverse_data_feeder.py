"""
Given X, y, generate batches of sentences
"""
from data_feeder import DataFeeder


class ToyDataFeeder(DataFeeder):
  def __init__(self, X, y, word_to_idx=None, idx_to_word=None,
               label_to_idx=None,
               idx_to_label=None):
    super(ToyDataFeeder, self).__init__(X, y)

    if word_to_idx is None:
      self._populate_word2idx()
    else:
      self._word_to_idx = word_to_idx
      self._idx_to_word = idx_to_word

    if label_to_idx is None:
      self._populate_label2idx()
    else:
      self._label_to_idx = label_to_idx
      self._idx_to_label = idx_to_label


if __name__ == "__main__":
  from pprint import pprint
  from data.toy_reverse_data import parse_data

  test = '../../dataset/toy_reverse/test/'
  dev = '../../dataset/toy_reverse/dev/'
  train = '../../dataset/toy_reverse/train/'

  X_test, y_test = parse_data(test)
  X_dev, y_dev = parse_data(dev)
  X_train, y_train = parse_data(train)

  train_ner_data = ToyDataFeeder(X_train, y_train)
  X_train_batches, y_train_batches, train_lens \
    = train_ner_data.naive_batch_buckets(32)

  print(train_ner_data.word_to_idx, len(train_ner_data.word_to_idx))
  print(train_ner_data.label_to_idx, len(train_ner_data.label_to_idx))
  # import pdb; pdb.set_trace()