"""
Given X, y, generate batches of sentences
"""
from data_feeder import DataFeeder


class NERData(DataFeeder):
  def __init__(self, X, y, word_to_idx=None, idx_to_word=None):
    """
    Construtor for NER data 
        
    Args:
      X: all sentences 
      y: all labels corresponding to X 
      word_to_idx: only provided for test and dev set, for training set, 
      we populate from beginning.  
    """
    # super(CCGData, self).__init__(X, y)  # python 2
    super().__init__(X, y)

    if word_to_idx is None:
      # training data
      print("Populate word2idx")
      self._populate_word2idx()
    else:
      # dev or test data
      self._word_to_idx= word_to_idx
      self._idx_to_word = idx_to_word

    self._label_to_idx = {'<p>': 0, 'I-LOC': 1, 'B-ORG': 2, 'O': 3,
                          'I-PER': 4, 'I-MISC': 5, 'B-MISC': 6, 'I-ORG': 7,
                          'B-LOC': 8, '<e>': 9}

    self._idx_to_label = {0: '<p>', 1: 'I-LOC', 2: 'B-ORG', 3: 'O',
                          4: 'I-PER', 5: 'I-MISC', 6: 'B-MISC', 7: 'I-ORG',
                          8: 'B-LOC', 9: '<e>'}