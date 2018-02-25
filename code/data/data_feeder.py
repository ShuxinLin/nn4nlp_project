"""
Given X, y, generate batches of sentences
"""
import random


PAD_token = '<p>'
SOS_token = '<s>'
EOS_token = '<e>'
label_to_idx = {'<p>': 0, '<s>': 1, 'I-LOC': 2, 'B-ORG': 3, 'O': 4,
                'I-PER': 5, 'I-MISC': 6, 'B-MISC': 7, 'I-ORG': 8,
                'B-LOC': 9, '<e>': 10}


def generate_batch(X, y, batch_size):
  """
  
  Args:
    X: training samples  
    y: labels (NER tags) 
    batch_size: number of sentences in a batch 

  Returns:
    a batch with equal-length sentences 
    every sentence will have the same length with the longest sentence in 
    that batch, if less then will be padded 
  """
  pass


def get_word2idx(all_data_set):
  """Given a list of all available data (not including labels), get word2idx"""
  word_to_idx = {PAD_token: 0, SOS_token: 1, EOS_token: 2}
  current = 3
  for segment in all_data_set:
    for line in segment:
      for word in line:
        if word.lower() not in word_to_idx:
          word_to_idx[word.lower()] = current
          current += 1

  return word_to_idx


def naive_batch(X, y, batch_size, word2index, shuffle=True):
  """
  Naive batching, memory-unfriendly version 
  Bascially take all data, convert into batches, do sorting, padding 
  and store back 
  
  Args:
    X: samples, list of list 
    y: labels, list of list, same cardinality format as X 
    batch_size: number of sentences  

  Returns:
    2 lists arranged in batches
  """

  # Return a list of indexes, one for each word in the sentence, plus EOS
  def to_traning_index(sentence):
    # print(sentence, type(sentence[0]))
    return [word2index[word.lower()] for word in sentence] + \
           [word2index[EOS_token]]

  def to_label_index(sentence):
    # print(sentence, type(sentence[0]))
    return [label_to_idx[word] for word in sentence] + \
           [label_to_idx[EOS_token]]

  # Pad a with the PAD symbol
  def pad_seq(seq, max_length):
    seq += [0 for _ in range(max_length - len(seq))]
    return seq

  l = len(X)
  n_batches = l // batch_size
  all_index = range(l)

  if shuffle:
    random.shuffle(all_index)

  X_batch, y_batch = [], []
  for i in range(n_batches):
    idx = all_index[i*batch_size:(i+1)*batch_size]

    # record all sentences and associated lengths
    current_x, current_y, current_lengths = [], [], []
    for j in idx:
      print("Debug: {} | {}".format(X[j], y[j]))
      current_x.append(to_traning_index(X[j]))
      current_y.append(to_label_index(y[j]))
      current_lengths.append(len(X[j]))

    # sort descending based on lengths
    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(current_x, current_y), key=lambda p: len(p[0]),
                       reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # debug
    print(input_seqs)
    print(target_seqs)
    print(current_lengths)

    # For input and target sequences, get array of lengths and pad with 0s
    # to max length
    input_padded = [pad_seq(s, max(current_lengths)+ 1) for s in input_seqs]
    labels_padded = [pad_seq(s, max(current_lengths)+ 1) for s in target_seqs]

    # TODO: should become generator!
    X_batch.append(input_padded)
    y_batch.append(labels_padded)

  return X_batch, y_batch