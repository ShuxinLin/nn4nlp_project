"""
Given X, y, generate batches of sentences
"""
from data_feeder import DataFeeder


class CCGDataFeeder(DataFeeder):
  def __init__(self, X, y, word_to_idx=None, label_to_idx=None,
               idx_to_word=None, idx_to_label=None):
    """
    Construtor for NER data 
        
    Args:
      X: all sentences 
      y: all labels corresponding to X 
      word_to_idx: only provided for test and dev set, for training set, 
      we populate from beginning.  
      label_to_idx: same thing for labels 
    """
    super(CCGDataFeeder, self).__init__(X, y)  # python 2 compatible

    if word_to_idx is None:
      # training data
      self._populate_word2idx()
    else:
      # dev or test data
      self._word_to_idx = word_to_idx
      self._idx_to_word = idx_to_word

    if label_to_idx is None:
      self._populate_label2idx()
    else:
      self._label_to_idx = label_to_idx
      self._idx_to_label = idx_to_label
