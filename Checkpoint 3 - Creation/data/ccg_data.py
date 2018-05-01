"""
Process CCG Super tagging data 
"""
import os


def parse_data(filename=None):
  all_sentences, super_tags = [], []
  tag_set = set()
  count = 0
  with open(filename) as f:
    for line in f:
      line = line.strip()
      if line != '':
        count += 1

        if len(line.split('|||')) == 1:
          print(line.split('|||'))

        sentence, annotations = line.split('|||')
        words = sentence.split()
        tags = annotations.split()
        assert len(words) == len(tags)

        # print words, tags
        for tag in tags:
          tag_set.add(tag)

        all_sentences.append(words)
        super_tags.append(tags)

  print("{} processed, there are {} lines, tag space has {} different tags".
        format(filename, count, len(tag_set)))

  return all_sentences, super_tags, tag_set


if __name__ == "__main__":
  ccg_data_dir = "../../dataset/supertag_data"
  train_words, train_tags, train_tag_set = parse_data(os.path.join(ccg_data_dir,
                                                                   "train.dat"))
  dev_words, dev_tags, dev_tag_set = parse_data(os.path.join(ccg_data_dir,
                                                             "dev.dat"))
  test_words, test_tags, test_tag_set = parse_data(os.path.join(ccg_data_dir,
                                                                "test.dat"))

  all_tag_space = train_tag_set.union(dev_tag_set).union(test_tag_set)
  print(len(all_tag_space))