import pandas as pd
import re
import collections
import json
import csv
from operator import itemgetter

class Preprocessor():
    def __init__(self, data_path, filename):
        self.path = data_path
        self.filename = filename
        self.UNK_TOKEN = '<UNK>'
        self.PAD_TOKEN = '<PAD>'
        # self.EOS_TOKEN = '<EOS>'

        # Token '<BEG>' is not used during preprocessing,
        # but will be used during decoding.
        self.BEG_TOKEN = '<BEG>'
        self.SEP = '\t'
        self.data = None
        # self.max_sentence_length = 300
        self.max_sentence_length = 500
        self._remove_digits = re.compile(r"""[.+-\/\\]?\d+([.,-:]\d+)?([.:]\d+)?(\.)?""")
        self.LENGTH_UNIT = 5
        self.new_data = None
        self.data_size = 0
        self.vocab_dict = dict()
        self.vocabulary_size = 0
        # self.entity_dict = {'I-LOC': 6, 'B-ORG': 9, 'O': 3, '<PAD>': 0, '<EOS>': 1, 'I-PER': 5, 'I-MISC': 4, 'B-MISC': 8, 'I-ORG': 2, 'B-LOC': 7}
        self.entity_dict = {'I-LOC': 6, 'B-ORG': 9, 'O': 3, '<PAD>': 0, '<BEG>': 1, 'I-PER': 5, 'I-MISC': 4, 'B-MISC': 8, 'I-ORG': 2, 'B-LOC': 7}
        self.entity_dict_size = len(self.entity_dict)
        self.indexed_data = None

    def read_file(self):
        raw_file = self.path + self.filename
        all_words, all_pos, all_chunk, all_entity = [], [], [], []
        all_examples = []
        inside_sentence = False
        with open(raw_file, 'r') as f:
            for line in f:
                if self._is_valid_line(line):
                    inside_sentence = True
                    word, pos, chunk, entity = line.strip().split()

                    all_words.append(self._preprocess_word(word))
                    all_pos.append(self._preprocess_pos(pos))
                    all_chunk.append(self._preprocess_chunk(chunk))
                    # all_entity.append(self._preprocess_entity(entity))
                    all_entity.append(entity)

                elif inside_sentence:   # end of sentence
                    word_in_one_sentence = ' '.join(all_words)
                    pos_in_one_sentence = ' '.join(all_pos)
                    chunk_in_one_sentence = ' '.join(all_chunk)
                    entity_in_one_sentence = ' '.join(all_entity)

                    all_examples.append([word_in_one_sentence, pos_in_one_sentence, chunk_in_one_sentence, entity_in_one_sentence])
                    all_words, all_pos, all_chunk, all_entity = [], [], [], []

                    inside_sentence = False     # end of processing one sentence
            self.data = pd.DataFrame(data=all_examples, columns=['SENTENCE', 'POS', 'CHUNK', 'ENTITY'])

    def preprocess(self, columns_to_process=['SENTENCE', 'ENTITY']):
        new_data = self.data.loc[self.data['SENTENCE'].str.len() <= self.max_sentence_length].copy()
        if 'SENTENCE' in columns_to_process:
            new_data['SENTENCE'] = new_data['SENTENCE'].apply(lambda x: self._preprocess_sentence(x))
        if 'ENTITY' in columns_to_process:
            new_data['ENTITY'] = new_data['ENTITY'].apply(lambda x: self._preprocess_entities(x))

        self.new_data = new_data.loc[:, columns_to_process]
        self.data_size = self.new_data.shape[0]
        for column in columns_to_process:
            preprocessed_file = self.path + 'processed_' + column + '_' + self.filename
            self.new_data.loc[:, [column]].to_csv(preprocessed_file, sep=self.SEP, index=False, quoting=csv.QUOTE_NONE)
        print('Successfully saved preprocessed file')

    def index_preprocess(self, columns_to_process=['SENTENCE', 'ENTITY']):
        indexed_data = self.new_data.copy()
        indexed_data['SENTENCE'] = indexed_data['SENTENCE'].apply(lambda x: self._index_sentence(x))
        indexed_data['ENTITY'] = indexed_data['ENTITY'].apply(lambda x: self._index_entity(x))
        self.indexed_data = indexed_data
        for column in columns_to_process:
            indexed_file = self.path + 'indexed_' + column + '_' + self.filename
            self.indexed_data.loc[:, [column]].to_csv(indexed_file, sep=self.SEP, index=False, quoting=csv.QUOTE_NONE)
        print('Successfully saved indexed_file file')

    def _index_sentence(self, sentence):
        words = sentence.split()
        indexes = [str(self.vocab_dict[word]) for word in words]
        return ' '.join(indexes)

    def _index_entity(self, entites):
        entities = entites.split()
        indexes = [str(self.entity_dict[e]) for e in entities]
        return ' '.join(indexes)

    """
    def _build_vocab(self, data):
        all_text = []
        for sentence in data['SENTENCE']:
            all_text.extend(sentence.split())

        all_text = filter(lambda a: a != self.PAD_TOKEN, all_text)
        all_words = collections.Counter(all_text).most_common()

        sorted_by_name = sorted(all_words, key=lambda x: x[0])
        all_words = sorted(sorted_by_name, key=lambda x: x[1], reverse=True)
        tokens = [(self.PAD_TOKEN, -1), (self.UNK_TOKEN, -1), (self.EOS_TOKEN, -1)]
        all_words = tokens + all_words
        self.vocab_dict = dict()
        for word in all_words:
            if word[0] not in self.vocab_dict:
                self.vocab_dict[word[0]] = len(self.vocab_dict)
        self.vocabulary_size = len(self.vocab_dict)
        vocab_file = self.path + "vocab_" + self.filename
        with open(vocab_file, 'w') as f:
            for word in all_words:
                f.write("%s\t%d\n" % (word[0], self.vocab_dict[word[0]]))
        print('Saved vocabulary to vocabulary file. vocab_size: ', self.vocabulary_size)
    """

    def _preprocess_sentence(self, sentence):
        sentence = self._regex_preprocess(sentence)
        #sentence = self._add_paddings_eos(sentence)
        sentence = self._add_paddings(sentence)
        return sentence

    def _regex_preprocess(self, sentence):
        sentence = self._remove_digits.sub('reg_digitz', sentence)
        sentence = sentence.replace('"', 'reg_quotes')
        return sentence

    """
    def _add_paddings_eos(self, sentence):
        words = sentence.split()
        length = len(words)
        num_of_paddings = (self.LENGTH_UNIT - length % self.LENGTH_UNIT) if (length % self.LENGTH_UNIT > 0) else 0
        words.extend([self.PAD_TOKEN] * num_of_paddings)
        words.append(self.EOS_TOKEN)
        sentence = ' '.join(words)
        return sentence
    """

    def _add_paddings(self, sentence):
        words = sentence.split()
        length = len(words)
        num_of_paddings = (self.LENGTH_UNIT - length % self.LENGTH_UNIT) if (length % self.LENGTH_UNIT > 0) else 0
        words.extend([self.PAD_TOKEN] * num_of_paddings)
        sentence = ' '.join(words)
        return sentence

    """
    def _preprocess_entities(self, entities):
        entities = entities.split()
        length = len(entities)
        num_of_paddings = (self.LENGTH_UNIT - length % self.LENGTH_UNIT) if (length % self.LENGTH_UNIT > 0) else 0
        entities.extend(['O'] * num_of_paddings)
        entities.append(self.EOS_TOKEN)
        return ' '.join(entities)
    """

    def _preprocess_entities(self, entities):
        entities = entities.split()
        length = len(entities)
        num_of_paddings = (self.LENGTH_UNIT - length % self.LENGTH_UNIT) if (length % self.LENGTH_UNIT > 0) else 0
        entities.extend(['O'] * num_of_paddings)
        return ' '.join(entities)

    def _preprocess_word(self, word):
        return word.lower()
        # add more in the future

    def _preprocess_pos(self, pos):
        result = None

        if pos == 'NN' or pos == 'NNS':
            result = 'NN'
        elif pos == 'FW':
            result = 'FW'
        elif pos == 'NNP' or pos == 'NNPS':
            result = 'NNP'
        elif 'VB' in pos:
            result = 'VB'
        else:
            result = 'OTHER'

        return result

    def _preprocess_chunk(self, chunk):
        result = None

        if 'NP' in chunk:
            result = 'NP'
        elif 'VP' in chunk:
            result = 'VP'
        elif 'PP' in chunk:
            result = 'PP'
        elif chunk == 'O':
            result = 'O'
        else:
            result = 'OTHER'

        return result

    def _preprocess_entity(self, entity):
        if entity not in self.entity_dict:
            self.entity_dict[entity] = len(self.entity_dict)

        return entity

    def _is_valid_line(self, line):
        if line.strip() == "" or len(line.split()) == 0:
            return False

        if '-DOCSTART-' in line:
            return False

        return True

    def minibatch(self, batch_size):
        print("generate mini batches.")
        X_batch = []
        Y_batch = []
        all_data = []
        for index, row in self.indexed_data.iterrows():
            splitted_sentence = list(map(int, row['SENTENCE'].split()))
            splitted_entities = list(map(int, row['ENTITY'].split()))
            assert len(splitted_entities) == len(splitted_sentence)
            all_data.append((len(splitted_sentence), splitted_sentence, splitted_entities))

        sorted_all_data = sorted(all_data, key=itemgetter(0))
        prev_len = 5
        X_minibatch = []
        Y_minibatch = []
        for data in sorted_all_data:
            if prev_len == data[0]:
                X_minibatch.append(data[1])
                Y_minibatch.append(data[2])
            else:
                X_minibatch = [X_minibatch[x:x + batch_size] for x in range(0, len(X_minibatch), batch_size)]
                Y_minibatch = [Y_minibatch[x:x + batch_size] for x in range(0, len(Y_minibatch), batch_size)]
                X_batch.extend(X_minibatch)
                Y_batch.extend(Y_minibatch)
                X_minibatch = []
                Y_minibatch = []
                X_minibatch.append(data[1])
                Y_minibatch.append(data[2])
                prev_len = data[0]
        X_minibatch = [X_minibatch[x:x + batch_size] for x in range(0, len(X_minibatch), batch_size)]
        Y_minibatch = [Y_minibatch[x:x + batch_size] for x in range(0, len(Y_minibatch), batch_size)]
        X_batch.extend(X_minibatch)
        Y_batch.extend(Y_minibatch)
        assert len(X_batch) == len(Y_batch)

        # X_batch = filter(lambda mini_batch: len(mini_batch) == batch_size, X_batch)
        # Y_batch = filter(lambda mini_batch: len(mini_batch) == batch_size, Y_batch)

        return list(X_batch), list(Y_batch)
