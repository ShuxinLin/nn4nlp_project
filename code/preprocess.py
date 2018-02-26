from preprocessor import Preprocessor
from operator import itemgetter
import collections

data_path = "../dataset/CoNLL-2003/"
train_file = "eng.train"
val_file = "eng.testa"
test_file = "eng.testb"

def prepocess(train, val):

    train_preprocessor = Preprocessor(data_path, train)
    train_preprocessor.read_file()
    train_preprocessor.preprocess()

    val_preprocessor = Preprocessor(data_path, val)
    val_preprocessor.read_file()
    val_preprocessor.preprocess()

    build_vocab(train_preprocessor, val_preprocessor)

    train_preprocessor.index_preprocess()
    train_X, train_Y = train_preprocessor.minibatch()
    val_preprocessor.index_preprocess()
    val_X, val_Y = val_preprocessor.minibatch()
    vocab_size = train_preprocessor.vocabulary_size
    label_size = train_preprocessor.entity_dict_size
    return train_X, train_Y, val_X, val_Y, vocab_size, label_size


def build_vocab(train_preprocessor, val_preprocessor):
    all_text = []
    for sentence in train_preprocessor.new_data['SENTENCE']:
        all_text.extend(sentence.split())
    for sentence in val_preprocessor.new_data['SENTENCE']:
        all_text.extend(sentence.split())
    all_text = filter(lambda a: a not in [train_preprocessor.EOS_TOKEN, train_preprocessor.PAD_TOKEN], all_text)
    all_words = collections.Counter(all_text).most_common()

    sorted_by_name = sorted(all_words, key=lambda x: x[0])
    all_words = sorted(sorted_by_name, key=lambda x: x[1], reverse=True)
    tokens = [(train_preprocessor.PAD_TOKEN, -1), (train_preprocessor.UNK_TOKEN, -1), (train_preprocessor.EOS_TOKEN, -1)]
    all_words = tokens + all_words

    vocab_dict = dict()
    for word in all_words:
        if word[0] not in vocab_dict:
            vocab_dict[word[0]] = len(vocab_dict)
    vocabulary_size = len(vocab_dict)
    train_preprocessor.vocab_dict = val_preprocessor.vocab_dict = vocab_dict
    train_preprocessor.vocabulary_size = val_preprocessor.vocabulary_size = vocabulary_size

    vocab_file = data_path + "vocab_dict"
    with open(vocab_file, 'w') as f:
        for word in all_words:
            f.write("%s\t%d\n" % (word[0], vocab_dict[word[0]]))
    print('Saved vocabulary to vocabulary file. vocab_size: ', vocabulary_size)

def get_index2word():
    index2word = dict()
    dict_file = '../dataset/CoNLL-2003/vocab_dict'
    with open(dict_file) as f:
        for line in f:
            (word, index) = line.split()
            index2word[int(index)] = word
    return index2word

def get_index2label():
    index2label = {0: '<PAD>', 1: '<EOS>', 2: 'I-ORG', 3: 'O', 4: 'I-MISC', 5: 'I-PER', 6: 'I-LOC', 7: 'B-LOC', 8: 'B-MISC', 9: 'B-ORG'}

    return index2label

# prepocess(train_file, val_file)
