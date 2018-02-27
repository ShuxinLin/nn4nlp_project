from sklearn.metrics import classification_report


def read_result(filepath=None):
  labels , preds = [], []
  with open(filepath) as f:
    for line in f:
      line = line.strip()
      if line == '':
        continue
      word, label, predict = line.split()
      print(word, label, predict)
      labels.append(label)
      preds.append(predict)

  return labels, preds

if __name__ == "__main__":
  labels, preds= read_result("./result/result_processed_train.txt")
  report = classification_report(labels, preds)
  print(report)
