#!/usr/bin/python3

import argparse
import random
import torch
import yaml

from pprint import pprint
from torch import nn

from data.toy_reverse_data import parse_data
from data.toy_reverse_data_feeder import ToyDataFeeder
from trainer import Trainer

from models.word_encoder_rnn import EncoderWordRNN
from models.decoder_rnn import DecoderRNN
from models.models import TaggingSeq2Seq


def main():
  random.seed(354)
  torch.manual_seed(564598)

  # ---------------------------------------------
  #             CONFIGURATIONS
  # ---------------------------------------------
  parser = argparse.ArgumentParser(description="Train a rnn classifier")
  parser.add_argument('-cfg', '--config', type=str, nargs='?',
                      help='configuration file path',
                      default='./config/toy_config.yaml')
  args = parser.parse_args()

  with open(args.config, 'r') as f:
    config = yaml.load(f)

  print('=' * 100)
  print("CONFIGURATIONS")
  pprint(config)
  print('=' * 100)

  # ---------------------------------------------
  #             DATA FEED
  # ---------------------------------------------
  test = '../dataset/toy_reverse/test/'
  dev = '../dataset/toy_reverse/dev/'
  train = '../dataset/toy_reverse/train/'

  X_testa, y_testa = parse_data(test)
  X_testb, y_testb = parse_data(dev)
  X_train, y_train = parse_data(train)

  PAD_token = '<p>'
  SOS_token = '<s>'
  EOS_token = '<e>'
  UNK_token = '<u>'

  word_to_idx = {PAD_token: 2, EOS_token: 1, SOS_token: 0, UNK_token: 3,
                 '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10,
                 '7': 11, '8': 12, '9': 13
                 }
  idx_to_word = {2: PAD_token, 1: EOS_token, 0: SOS_token, 3: UNK_token,
                 4: '0', 5: '1', 6: '2', 7: '3', 8: '4', 9: '5', 10: '6',
                 11: '7', 12: '8', 13: '9'}

  label_to_idx = {PAD_token: 2, EOS_token: 1, SOS_token: 0, UNK_token: 3,
                  '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10,
                  '7': 11, '8': 12, '9': 13
                  }
  idx_to_label = {2: PAD_token, 1: EOS_token, 0: SOS_token, 3: UNK_token,
                  4: '0', 5: '1', 6: '2', 7: '3', 8: '4', 9: '5', 10: '6',
                  11: '7', 12: '8', 13: '9'}

  train_ner_data_feeder = ToyDataFeeder(X_train, y_train,
                                        word_to_idx=word_to_idx,
                                        idx_to_word=idx_to_word,
                                        label_to_idx=label_to_idx,
                                        idx_to_label=idx_to_label)

  pprint("In Vocab {} ({})".format(word_to_idx, len(word_to_idx)))
  pprint("Out Vocab {} ({})".format(label_to_idx, len(label_to_idx)))

  testa_ner_data_feeder = ToyDataFeeder(X_testa, y_testa)
  testb_ner_data_feeder = ToyDataFeeder(X_testb, y_testb)

  # import pdb;
  # pdb.set_trace()


  # ---------------------------------------------
  #             MODEL CREATION
  # ---------------------------------------------
  encoder_cfg = config['encoder']

  encoder = EncoderWordRNN(n_layers=encoder_cfg['n_layers'],
                           hidden_size=encoder_cfg['hidden_size'],
                           cell_type=encoder_cfg['cell_type'],
                           is_bidirectional=encoder_cfg['is_bidirectional'],
                           max_seq_len=encoder_cfg['max_seq_len'],
                           in_dropout=encoder_cfg['in_dropout'],
                           out_dropout=encoder_cfg['out_dropout'],
                           in_vocab_size=len(word_to_idx),
                           is_packing_needed=encoder_cfg['is_packing'],
                           is_batch_first=encoder_cfg['is_batch_first'],
                           is_embedding_used=encoder_cfg['is_embedding_used'],
                           out_embbed=encoder_cfg['out_embedding_dim'],
                           is_embed_pretrain=encoder_cfg['is_embed_pretrain'])

  # _decoder
  decoder_cfg = config['decoder']

  decoder = DecoderRNN(n_layers=decoder_cfg['n_layers'],
                       hidden_size=decoder_cfg['hidden_size'],
                       cell_type=decoder_cfg['cell_type'],
                       is_bidirectional=decoder_cfg['is_bidirectional'],
                       max_seq_len=decoder_cfg['max_seq_len'],
                       in_dropout=decoder_cfg['in_dropout'],
                       out_dropout=decoder_cfg['out_dropout'],
                       out_vocab_size=len(label_to_idx),
                       is_batch_first=decoder_cfg['is_batch_first'],
                       is_embedding_used=decoder_cfg['is_embedding_used'],
                       out_embbed=decoder_cfg['out_embedding_dim'],
                       is_embed_pretrain=decoder_cfg['is_embed_pretrain'],
                       is_attended=decoder_cfg['is_attended'],
                       attention_type=decoder_cfg['attention_type'],  # Ly == Lx
                       is_softmax_output_needed=
                       decoder_cfg['is_softmax_output_needed'],
                       SOS_ID=decoder_cfg['SOS_ID'],
                       EOS_ID=decoder_cfg['EOS_ID']
                       )

  # decoder = DecoderRNN(vocab_size=len(label_to_idx),
  #                      max_len=decoder_cfg['max_seq_len'],
  #                      hidden_size=decoder_cfg['hidden_size'],
  #                      sos_id=word_to_idx['0'],
  #                      eos_id=word_to_idx['1'],
  #                      n_layers=decoder_cfg['n_layers'],
  #                      rnn_cell=decoder_cfg['cell_type'],
  #                      bidirectional=decoder_cfg['is_bidirectional'],
  #                      input_dropout_p=0,
  #                      dropout_p=0,
  #                      use_attention=False)

  seq2seq = TaggingSeq2Seq(encoder, decoder)
  # seq2seq = Seq2seq(encoder, decoder)

  # ---------------------------------------------
  #             INITIALIZATION
  # ---------------------------------------------
  if torch.cuda.is_available():
    seq2seq.cuda()

  # uniformly init params
  # for param in seq2seq.parameters():
  #   param.data.uniform_(-0.08, 0.08)

  # ---------------------------------------------
  #             TRAIN, EVAL
  # ---------------------------------------------
  trainer_cfg = config['trainer']

  # create a trainer
  trainer = Trainer(task_name=config['generic']['task_name'],
                    seq2seq_model=seq2seq,
                    train_criterion=nn.NLLLoss(),
                    val_criterion=nn.NLLLoss(),
                    decode_method=trainer_cfg['decode_method'],
                    teacher_forcing_ratio=trainer_cfg['teacher_forcing_ratio'],
                    is_tagging=trainer_cfg['is_tagging'],
                    learning_rate=trainer_cfg['learning_rate'],
                    optim_type=trainer_cfg['optim_type'],
                    momentum=trainer_cfg['momentum'],
                    batch_style=trainer_cfg['batch_style'],
                    print_freq=trainer_cfg['print_freq'],
                    save_freq=trainer_cfg['save_freq'],
                    is_tensorboard=trainer_cfg['is_tensorboard'],
                    log_dir=trainer_cfg['log_dir'],
                    )

  trainer.train(is_checkpoint_retrieved=trainer_cfg['is_checkpoint_retrieved'],
                train_data_feeder=train_ner_data_feeder,
                val_data_feeder=testa_ner_data_feeder,
                test_data_feeder=testb_ner_data_feeder,
                batch_size=trainer_cfg['batch_size'],
                n_epoch=trainer_cfg['n_epoch'],
                ckpt_path=trainer_cfg['ckpt_dir'],
                n_saved_versions=trainer_cfg['num_saved_versions'],
                is_f1_reported=trainer_cfg['is_f1_reported']
                )

  # ---------------------------------------------
  #             REPORTING
  # ---------------------------------------------
  print(trainer.train_losses)
  print(trainer.val_losses)
  print(trainer.test_losses)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("User interrupting; exiting now ...")
    exit(0)
