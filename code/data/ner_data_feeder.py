"""
Given X, y, generate batches of sentences
"""
from data_feeder import DataFeeder


class NERDataFeeder(DataFeeder):
  def __init__(self, X, y, word_to_idx=None, idx_to_word=None):
    """
    Construtor for NER data 
        
    Args:
      X: all sentences 
      y: all labels corresponding to X 
      word_to_idx: only provided for test and dev set, for training set, 
      we populate from beginning.  
    """
    # super().__init__(X, y)  # python 3
    super(NERDataFeeder, self).__init__(X, y)  # python 2 compatible

    if word_to_idx is None:
      # training data
      print("Populate word2idx")
      self._populate_word2idx()
    else:
      # dev or test data
      self._word_to_idx= word_to_idx
      self._idx_to_word = idx_to_word

    self._label_to_idx = {'<s>': 0, '<e>': 1, 'I-LOC': 2, 'B-ORG': 3, 'O': 4,
                          'I-PER': 5, 'I-MISC': 6, 'B-MISC': 7, 'I-ORG': 8,
                          'B-LOC': 9 }

    self._idx_to_label = {0: '<s>', 2: 'I-LOC', 3: 'B-ORG', 4: 'O',
                          5: 'I-PER', 6: 'I-MISC', 7: 'B-MISC', 8: 'I-ORG',
                          9: 'B-LOC', 1: '<e>'}


if __name__ == "__main__":
  from data.ner_data import parse_data

  testa = "../../dataset/CoNLL-2003/eng.testa"
  testb = "../../dataset/CoNLL-2003/eng.testb"
  train = "../../dataset/CoNLL-2003/eng.train"
  X_testa, y_testa = parse_data(testa)
  X_testb, y_testb = parse_data(testb)
  X_train, y_train = parse_data(train)

  train_ner_data = NERDataFeeder(X_train, y_train)
  X_train_batches, y_train_batches, train_lens \
    = train_ner_data.naive_batch_buckets(32)

  import pdb; pdb.set_trace()