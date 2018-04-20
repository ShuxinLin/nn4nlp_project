import os


def parse_data(dir_path=None):
  data_file = os.path.join(dir_path, 'data.txt')

  X, y = [], []
  with open(data_file) as f:
    for line in f:
      if line != '\n':
        source, target = line.strip().split('\t')
        print(source.split(), '\t\t\t', target.split())
        X.append(source.split())
        y.append(target.split())

  return X, y

if __name__ == "__main__":
  test = '../../dataset/toy_reverse/test/'
  dev = '../../dataset/toy_reverse/dev/'
  train = '../../dataset/toy_reverse/train/'

  X_test, y_test = parse_data(test)
  X_dev, y_dev = parse_data(dev)
  X_train, y_train = parse_data(train)

