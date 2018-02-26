#!/usr/bin/python3
from ner import ner
from preprocess import *

data_path = "../dataset/CoNLL-2003/"
train_file = "eng.train"
val_file = "eng.testa"
test_file = "eng.testb"
result_path = "../result/"
def main():
    # Temporarily generate data by hand for test purpose

    # train_X_raw = [
    #     ["The dog ate the apple".split(" "), "Everybody read that book <p>".split(" ")],
    #     ["The dog ate the apple".split(" "), "Everybody read that book <p>".split(" ")],
    #     ["The dog ate the apple".split(" "), "Everybody read that book <p>".split(" ")],
    #     ["The dog ate the apple and banana".split(" "), "Everybody read that book and book <p>".split(" ")]]
    #
    #
    # train_Y_raw = [
    #     [["DET", "NN", "V", "DET", "NN"], ["NN", "V", "DET", "NN", "<p>"]],
    #     [["DET", "NN", "V", "DET", "NN"], ["NN", "V", "DET", "NN", "<p>"]],
    #     [["DET", "NN", "V", "DET", "NN"], ["NN", "V", "DET", "NN", "<p>"]],
    #     [["DET", "NN", "V", "DET", "NN", "DET", "NN"], ["NN", "V", "DET", "NN", "DET", "NN", "<p>"]]]
    #
    #
    # word_to_idx = {"<p>": 0}
    # cur_idx = 1
    # for batch in train_X_raw:
    #     for sen in batch:
    #         for word in sen:
    #             if word not in word_to_idx:
    #                 word_to_idx[word] = cur_idx
    #                 cur_idx += 1
    #
    # label_to_idx = {"<p>": 0, "<s>": 1, "DET": 2, "NN": 3, "V": 4}
    #
    # train_X = [[[word_to_idx[w] for w in sen] for sen in batch] for batch in train_X_raw]
    # train_Y = [[[label_to_idx[t] for t in label] for label in batch] for batch in train_Y_raw]
    #
    # """
    # print(word_to_idx)
    # print(label_to_idx)
    #
    # for b_idx, batch in enumerate(train_X):
    #     print("batch index", b_idx)
    #     for idx, sen in enumerate(batch):
    #         print("instance index", idx)
    #         print("sen", sen)
    #
    # for b_idx, batch in enumerate(train_Y):
    #     print("batch index", b_idx)
    #     for idx, label in enumerate(batch):
    #         print("instance index", idx)
    #         print("label", label)
    # """
    # print train_X
    train_X, train_Y, val_X, val_Y, vocab_size, label_size = prepocess(train_file, val_file)
    ######################################
    word_embedding_dim = 16
    hidden_dim = 16
    label_embedding_dim = 16

    machine = ner(word_embedding_dim, hidden_dim, label_embedding_dim, vocab_size, label_size,
                  learning_rate=0.1, minibatch_size=64, max_epoch=300, train_X=train_X, train_Y=train_Y, test_X=val_X,
                  test_Y=val_Y)

    machine.train()
    machine.eval_on_train()
    machine.test()


if __name__ == "__main__":
    main()
