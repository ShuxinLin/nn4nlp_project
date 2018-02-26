"""
Given X, y, generate batches of sentences
"""
import random

PAD_token = '<p>'
EOS_token = '<e>'
UNK_token = '<u>'


class DataFeeder(object):
  """Wrapper for general type of  data 
     The idea is to facilitate with batch iterations 
  """

  def __init__(self, X, y):
    # private fields
    self.__sentences = X
    self.__tags = y

    # protected fields - inherited classes must implement those 2 dictionaries
    self._word_to_idx = None
    self._label_to_idx = None

  def _populate_word2idx(self):
    """
    ---Protected method - to be inherited by subclasses.---
     
    Given a list of all available data (not incl. labels), get word2idx"""
    self._word_to_idx = {PAD_token: 0, EOS_token: 1, UNK_token: 2}
    current = 3
    for line in self.__sentences:
      for word in line:
        word = word.lower()
        if word not in self._word_to_idx:
          self._word_to_idx[word] = current
          current += 1
    print("There are totally {} unique words in this dataset".
          format(len(self._word_to_idx)))

  def _populate_label2idx(self):
    """---Protected method - to be inherited by subclasses.---
      UNK_TOKEN is used for CCG tagging where there are at least 1285 diff. tags
    """
    self._label_to_idx = {PAD_token: 0, EOS_token: 1, UNK_token: 2}
    current = 3

    for line in self.__tags:
      for tag in line:
        if tag not in self._label_to_idx:
          self._label_to_idx[tag] = current
          current += 1
    print("There are totally {} unique labels in this dataset".
          format(len(self._label_to_idx)))

  @property
  def sentences(self):
    return self.__sentences

  @property
  def tags(self):
    return self.__tags

  @property
  def label_to_idx(self):
    return self._label_to_idx

  @property
  def word_to_idx(self):
    return self._word_to_idx

  def __to_word_index(self, sentence):
    """
    Convert a sentence in to index.
    We only collect labels from training data so there would be UNK_token 
      in dev and test data for unknown words. 

    Args:
      sentence: collection of words  

    Returns:
      List of indices of all words respectively.  

    """
    words_idx = []

    for word in sentence:
      word = word.lower()
      if word not in self._word_to_idx:
        words_idx.append(self._word_to_idx[UNK_token])
      else:
        words_idx.append(self._word_to_idx[word])

    return words_idx

  def __to_label_index(self, tags):
    """Same as to_word_index but for tags"""
    labels_idx = []

    for tag in tags:
      if tag not in self._label_to_idx:
        labels_idx.append(self._label_to_idx[UNK_token])
      else:
        labels_idx.append(self._label_to_idx[tag])

    return labels_idx
    # return [self._label_to_idx[tag] for tag in tags] + \
    #        [self._label_to_idx[EOS_token]]

  @staticmethod
  def pad_seq(seq, max_length):
    """
    NOTE: padding token index of both word2idx and label2idx must be both ZERO! 
    Args:
      seq: numberic sequence 
      max_length: to-be-extended size of that sequence 

    Returns:
      zero-padded sequence 

    """
    seq += [0 for _ in range(max_length - len(seq))]
    return seq

  def generate_batch(self, batch_size):
    """
    Public interface to generate batch in generator way 
    
    Args:
      batch_size: number of sentences in a batch 

    Returns:
      a batch with equal-length sentences 
      every sentence will have the same length with the longest sentence in 
      that batch, if less then will be padded 
    """
    pass

  def naive_batch(self, batch_size, shuffle=True):
    """
    Public interface for naive batching, memory-unfriendly version.
    Fundamentally, take all data, convert into batches, do sorting, padding 
      and store back 
    
    Args:
      shuffle: or  not
      batch_size: number of sentences  
  
    Returns:
      2 lists arranged in batches
    """

    n_batches = len(self.__sentences) // batch_size
    print(n_batches)
    all_sentence_index = list(range(len(self.__sentences)))

    if shuffle:
      random.shuffle(all_sentence_index)

    X_all_batches, y_all_batches = [], []
    for i in range(n_batches):
      idx = all_sentence_index[i * batch_size:(i + 1) * batch_size]

      # record all sentences and associated lengths
      current_x, current_y, current_lengths = [], [], []
      for j in idx:
        # print("Debug: {} | {}".format(self.__sentences[j], self.__tags[j]))

        # convert to numbers for all source and targets
        current_x.append(self.__to_word_index(self.__sentences[j]))
        current_y.append(self.__to_label_index(self.__tags[j]))
        current_lengths.append(len(self.__sentences[j]))

      # sort descending based on lengths
      seq_pairs = sorted(zip(current_x, current_y, current_lengths),
                         key=lambda k: len(k[0]), reverse=True)
      input_seqs, target_seqs, current_lengths = zip(*seq_pairs)

      # debug
      # print(input_seqs)
      # print(target_seqs)
      # print(current_lengths)

      # For input and target sequences, get array of lengths and pad with 0s
      # to max length
      input_padded = [DataFeeder.pad_seq(s, max(current_lengths) + 1) for s in
                      input_seqs]
      labels_padded = [DataFeeder.pad_seq(s, max(current_lengths) + 1) for s in
                       target_seqs]

      # debug
      print(input_padded)
      print(labels_padded)
      print()

      # TODO: change to each time step, transpose 0 and 1 dimension


      # TODO: should become generator!
      X_all_batches.append(input_padded)
      y_all_batches.append(labels_padded)

    return X_all_batches, y_all_batches
