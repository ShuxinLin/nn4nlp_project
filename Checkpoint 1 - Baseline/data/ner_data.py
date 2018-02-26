"""
Process NER dataset of CoNLL-2003 
Ref: https://www.clips.uantwerpen.be/conll2003/ner/
"""
from collections import defaultdict
from pprint import pprint


def parse_data(filename=None):
  """
  Given a data file of either train, testa or testb, parse into a list. 
  Refer to the Dataset link above for the format 
  
  
  Args:
    filename: eng.testa or eng.testb or eng.train 

  Returns:
    X: list of words 
    y: list of Name Entity, of the same length as X

  """
  def reset_sentence(words, pos, tags, ne):
     """For each sentence we have Words, POS, Chunk Tag and Name Entities 
     reset"""
     return [], [], [], []

  X, y = [], []
  with open(filename) as f:
    # skip first 2 lines
    for _ in range(2):
      f.readline()

    words, pos, tags, ne = [], [], [], []
    count = 0
    for line in f:
      line = line.strip()
      if line == '':
        # print "empty line"
        count += 1
        assert len(words) == len(pos) == len(tags) == len(ne)

        # collect data
        # NOTE: we discard POS and TAGS
        # print(words, pos, tags, ne)
        if len(words) > 0:
          X.append(words)
          y.append(ne)

        # reset sentence content
        words, pos, tags, ne = reset_sentence(words, pos, tags, ne)

        # go on for the next line
        continue

      # gather content for the whole line
      w, p, t, n = line.strip().split()
      words.append(w)
      pos.append(p)
      tags.append(t)
      ne.append(n)

  print("File {} has \t{} lines".format(filename, count))
  return X, y


if __name__ == "__main__":
  testa = "../../dataset/CoNLL-2003/eng.testa"
  testb = "../../dataset/CoNLL-2003/eng.testb"
  train = "../../dataset/CoNLL-2003/eng.train"
  X_testa, y_testa = parse_data(testa)
  X_testb, y_testb = parse_data(testb)
  X_train, y_train = parse_data(train)

  # import pdb; pdb.set_trace()

  # validate by counting occurences of each NER tag
  all_ne_set = defaultdict(lambda: 0)
  for l in y_train:
    for s in l:
      all_ne_set[s] += 1
  pprint(all_ne_set)

  s = 0
  for k in all_ne_set:
    s += all_ne_set[k]
  print s