import logging
import numpy as np
import os
import torch

from sklearn.metrics import classification_report

from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch import nn
from torch import optim

from utils import save_model
from utils import load_latest_model
from utils import clip_grad_norm


class Trainer(object):
  def __init__(self, task_name,
               seq2seq_model,
               train_criterion,
               val_criterion,
               is_tagging,
               learning_rate,
               decode_method='greedy',
               teacher_forcing_ratio=0.5,
               batch_style='naive_bucket',
               optim_type='sgd',
               momentum=0.8,
               print_freq=100,
               save_freq=2000,
               is_tensorboard=True,
               log_dir='logs'):

    self.__task_name = task_name

    self.__seq2seq_model = seq2seq_model
    self.__train_criterion = train_criterion
    self.__val_criterion = val_criterion

    if torch.cuda.is_available():
      self.__train_criterion.cuda()
      self.__val_criterion.cuda()

    self.__is_tagging = is_tagging

    self.__learning_rate = learning_rate

    self.__decode_method = decode_method
    self.__teacher_forcing_ratio = teacher_forcing_ratio
    self.__batch_style = batch_style

    self.__optim_type = optim_type.lower()
    if self.__optim_type == 'sgd':
      self.__optimizer = optim.SGD(self.__seq2seq_model.parameters(),
                                   lr=self.__learning_rate,
                                   momentum=momentum)
    elif self.__optim_type == 'adam':
      self.__optimizer = optim.Adam(self.__seq2seq_model.parameters(),
                                    lr=self.__learning_rate)
    else:
      raise ValueError("Not yet suppoted type of __optimizer!")

    self.__global_step = 0
    self.train_losses = []
    self.val_losses = []
    self.test_losses = []

    self.__print_freq = print_freq
    self.__save_freq = save_freq

    self.__writer = SummaryWriter(log_dir=log_dir) if is_tensorboard else None

    self.__logger = logging.getLogger(__name__)

  @staticmethod
  def __extract_decode_for_report(batch_decoded_seq, batch_labels):
    """
    Given a train batch: B x Ly and output (a Ly-length list of B x 1), 
      rearrange them so that we can run F1 report easily. 
        
    Args:
      batch_decoded_seq (list): Ly columnar tensors, each has dimension B x 1
      batch_labels (Tensor): B x Ly 
        
    Returns:
      2 flattened list of labels and output 
    """
    output_tensor = torch.cat(batch_decoded_seq, dim=1)  # B x Ly

    # for CUDA tensor cannot be converted to numpy, we transport to CPU first
    outputs = list(output_tensor.cpu().data.numpy().flatten())
    labels = list(batch_labels.cpu().data.numpy().flatten())

    return outputs, labels

  def __train_batch(self,
                    train_batch_x,
                    x_lengths,
                    train_batch_y,
                    decode_method,
                    teacher_forcing_ratio):
    """
    Train a batch, with SGD 
    
    Args:
      train_batch_x: input (B x Lx)
      x_lengths: if bucketing, all sentences have the same length 
      train_batch_y: labels (B x Ly) - for tagging problem: Ly == Lx  
      decode_method: greedy or beam or others 
      teacher_forcing_ratio (float): between [0., 1.]

    Returns:
      loss (float) 
      flatten_outputs (list of index) 
      flatten_labels (list of index)
    """
    # turn on TRAINING FLAG
    self.__seq2seq_model.train(True)

    result_dict = self.__seq2seq_model(encoder_input=train_batch_x,
                                       batch_source_lengths=x_lengths,
                                       decoder_input=train_batch_y,
                                       teacher_forcing_ratio=teacher_forcing_ratio,
                                       decode_method=decode_method,
                                       is_tagging=self.__is_tagging,
                                       is_training=True)

    output = result_dict['decoder_output']  # L_out x B x V, last output
    output_lengths = result_dict['decode_lengths']
    decoded_seqs = result_dict['symbols_idx_seq']  #

    flatten_outputs, flatten_labels = \
      Trainer.__extract_decode_for_report(decoded_seqs, train_batch_y)

    loss = 0

    for i in range(len(output)):
      taken_output = output[i]
      label = train_batch_y[:, i]
      # TODO: make use of output_lengths if NOT using BUCKETING batching
      loss = loss + self.__train_criterion(taken_output, label)

    # backprop
    self.__seq2seq_model.zero_grad()
    # self.__optimizer.zero_grad()
    loss.backward()

    # clip norm
    clip_grad_norm(self.__seq2seq_model.parameters(), 5., 2,
                   self.__writer, self.__global_step)

    # make a update step
    self.__optimizer.step()

    # tensor to float
    float_loss = loss / len(output)
    float_loss = float_loss.data[0]

    return float_loss, flatten_outputs, flatten_labels

  def __validate_batch(self,
                       val_batch_x,
                       x_lengths,
                       val_batch_y,
                       decode_method,
                       ):
    """ Same as __train_batch but without SGD """

    # turn on the evaluation flag
    self.__seq2seq_model.train(False)
    self.__seq2seq_model.eval()

    result_dict = self.__seq2seq_model(encoder_input=val_batch_x,
                                       batch_source_lengths=x_lengths,
                                       decoder_input=None,
                                       teacher_forcing_ratio=0.,
                                       decode_method=decode_method,
                                       is_tagging=self.__is_tagging,
                                       is_training=False)

    output = result_dict['decoder_output']  # B x Max_Len x V
    output_lengths = result_dict['decode_lengths']
    decoded_seqs = result_dict['symbols_idx_seq']

    flatten_outputs, flatten_labels = \
      Trainer.__extract_decode_for_report(decoded_seqs, val_batch_y)

    loss = 0

    # refine outputs because lengths of them are different, only take enough
    for i in range(len(output)):
      taken_output = output[i]
      label = val_batch_y[:, i]
      # update loss
      loss = loss + self.__val_criterion(taken_output, label)

    # tensor to float
    float_loss = loss / len(output)
    float_loss = float_loss.data[0]

    self.__seq2seq_model.train(True)

    return float_loss, flatten_outputs, flatten_labels

  def __validate_epoch(self, val_batches_x, val_batches_y, val_lens,
                       n_val_batches, decode_method, desc='validation'):

    # epoch-lavel monitoring variable
    epoch_val_loss = 0.
    val_reports = [[], []]

    for j in range(n_val_batches):
      val_batch_x = Variable(torch.LongTensor(val_batches_x[j]))
      val_batch_y = Variable(torch.LongTensor(val_batches_y[j]))
      val_batch_lens = Variable(torch.LongTensor(val_lens[j]))

      if torch.cuda.is_available():
        val_batch_x = val_batch_x.cuda()
        val_batch_y = val_batch_y.cuda()
        val_batch_lens = val_batch_lens.cuda()

      val_batch_loss, val_flatten_outputs, val_flatten_labels \
        = self.__validate_batch(val_batch_x,
                                val_batch_lens,
                                val_batch_y,
                                decode_method)
      # accumulate epoch-level monitoring variables
      if self.__writer:
        self.__writer.add_scalar(desc, val_batch_loss, j)

      epoch_val_loss += val_batch_loss
      val_reports[0] += val_flatten_outputs
      val_reports[1] += val_flatten_labels

    return epoch_val_loss, val_reports

  def __get_all_batches(self, batch_size,
                        train_data_feeder,
                        val_data_feeder,
                        test_data_feeder,
                        ):
    if self.__batch_style == 'naive':
      train_batches_x, train_batches_y, train_lens = \
        train_data_feeder.naive_batch(batch_size)
      val_batches_x, val_batches_y, val_lens = \
        val_data_feeder.naive_batch(batch_size)
      test_batches_x, test_batches_y, test_lens = \
        test_data_feeder.naive_batch(batch_size)
    elif self.__batch_style == 'naive_bucket':
      train_batches_x, train_batches_y, train_lens = \
        train_data_feeder.naive_batch_buckets(batch_size)
      val_batches_x, val_batches_y, val_lens = \
        val_data_feeder.naive_batch_buckets(batch_size)
      test_batches_x, test_batches_y, test_lens = \
        test_data_feeder.naive_batch_buckets(batch_size)

    return train_batches_x, train_batches_y, train_lens, \
           val_batches_x, val_batches_y, val_lens, \
           test_batches_x, test_batches_y, test_lens

  def __train_from_scratch(self, batch_size,
                           n_epoch,
                           train_data_feeder,
                           val_data_feeder,
                           test_data_feeder,
                           is_f1_reported=True,
                           ckpt_path='checkpoint',
                           n_saved_versions=3
                           ):
    """
    Training from scratch (global step 0).  
    
    Args:
      batch_size: 
      n_epoch: 
      train_data_feeder: 
      val_data_feeder: 
      test_data_feeder: 
      is_f1_reported: 
      ckpt_path:
      n_saved_versions: circular number of checkpoint versions 

    """
    for epoch in range(n_epoch):
      # draw all batches - needs some memory for naive stupid method
      # note: this should be done at every epoch since it shuffles data
      train_batches_x, train_batches_y, train_lens, \
      val_batches_x, val_batches_y, val_lens, \
      test_batches_x, test_batches_y, test_lens \
        = self.__get_all_batches(
        batch_size, train_data_feeder, val_data_feeder, test_data_feeder)

      n_train_batches = len(train_batches_x)
      n_val_batches = len(val_batches_x)
      n_test_batches = len(test_batches_x)

      # epoch-level monitoring variables
      # to be reset at the beginning of every epoch
      train_epoch_loss = 0.
      # for F1 reports: store ground truths and predicted values as 2 lists
      train_epoch_reports = [[], []]  # outputs, labels

      n_batches = 0
      for i in range(n_train_batches):
        self.__global_step += 1
        n_batches += 1

        # ------------------ TRAINING ------------------
        # draw a batch
        train_batch_x = Variable(torch.LongTensor(train_batches_x[i]))  # B x Ly
        train_batch_y = Variable(torch.LongTensor(train_batches_y[i]))  # B x Ly
        train_batch_lens = Variable(torch.LongTensor(train_lens[i]))

        if torch.cuda.is_available():
          train_batch_x = train_batch_x.cuda()
          train_batch_y = train_batch_y.cuda()
          train_batch_lens = train_batch_lens.cuda()

        # train and backprop this batch
        train_batch_loss, train_flatten_outputs, train_flatten_labels \
          = self.__train_batch(train_batch_x,
                               train_batch_lens,
                               train_batch_y,
                               self.__decode_method,
                               self.__teacher_forcing_ratio)

        # accumulate epoch-level monitoring variables
        if self.__writer:
          self.__writer.add_scalar('train_loss', train_batch_loss, self.__global_step)
        train_epoch_loss += train_batch_loss
        train_epoch_reports[0] += train_flatten_outputs
        train_epoch_reports[1] += train_flatten_labels

        # dump to console
        if n_batches % self.__print_freq == 0:
          print('Global step: %d\tEpoch: %d:\tBatch %d/%d\t\tTrain Loss = %.4f'
                % (self.__global_step, epoch + 1, n_batches, n_train_batches,
                   train_epoch_loss / n_batches))

        # save a checkpoint
        if self.__global_step % self.__save_freq == 0:
          save_model(ckpt_path, self.__seq2seq_model, self.__global_step,
                     self.__optimizer, desc=self.__task_name,
                     n_versions=n_saved_versions)

      # ------------------ EVALUATION ------------------
      val_epoch_loss, val_epoch_reports = \
        self.__validate_epoch(train_batches_x,
                              train_batches_y,
                              train_lens,
                              n_train_batches,
                              self.__decode_method,
                              desc='validation')
        # self.__validate_epoch(val_batches_x,
        #                       val_batches_y,
        #                       val_lens,
        #                       n_val_batches,
        #                       self.__decode_method)
      print('\n EVALUATION Epoch %d\t\tLoss = %.4f'
            # % (epoch + 1, val_epoch_loss / n_val_batches))
            % (epoch + 1, val_epoch_loss / n_train_batches))

      # ------------------ TESTING ------------------
      test_epoch_loss, test_epoch_reports \
        = self.__validate_epoch(test_batches_x,
                                test_batches_y,
                                test_lens,
                                n_test_batches,
                                self.__decode_method,
                                desc='test')

      print(' TESTING Epoch %d\t\tLoss = %.4f\n'
            % (epoch + 1, test_epoch_loss / n_test_batches))

      # update monitoring variables
      # if self.__writer:
      #   self.__writer.add_scalar('val_loss', val_epoch_loss / n_val_batches,
      #                            epoch)
      #   self.__writer.add_scalar('test_loss', test_epoch_loss / n_test_batches,
      #                            epoch)

      self.train_losses.append(train_epoch_loss / n_train_batches)
      self.val_losses.append(val_epoch_loss / n_val_batches)
      self.test_losses.append(test_epoch_loss / n_test_batches)

      # reporting
      if is_f1_reported:
        print("TRAINING REPORT")
        print(classification_report(y_true=train_epoch_reports[1],
                                    y_pred=train_epoch_reports[0]))

        print("VALIDATION REPORT")
        print(classification_report(y_true=val_epoch_reports[1],
                                    y_pred=val_epoch_reports[0]))

        print("TESTING REPORT")
        print(classification_report(y_true=test_epoch_reports[1],
                                    y_pred=test_epoch_reports[0]))

        print(train_epoch_reports[1][:1000])
        print(train_epoch_reports[0][:1000])
        # TODO: tensorboard

  def train(self, is_checkpoint_retrieved,
            batch_size,
            n_epoch,
            train_data_feeder,
            val_data_feeder,
            test_data_feeder,
            is_f1_reported=True,
            ckpt_path='checkpoint',
            n_saved_versions=3):
    """ Public API for training, see __train_from_scratch() for params info
      1. Try to load from checkpoint if it exists and continue to train from 
        the latest checkpoint.
      2. If checkpoint does not exists, train from scratch. 
    
    Args: 
      is_checkpoint_retrieved: try to load from checkpoint, or just train 
                                from scratch otherwise 
    """

    # try with checkpoint first
    if is_checkpoint_retrieved:
      latest_model, latest_metadata = load_latest_model(ckpt_path)

      if latest_model is not None:
        # load model
        self.__seq2seq_model = latest_model

        # load metadata
        self.__global_step = latest_metadata['global_step']
        self.__optimizer = latest_metadata['optimizer']

    # Continue with training right now
    self.__train_from_scratch(batch_size,
                              n_epoch,
                              train_data_feeder,
                              val_data_feeder,
                              test_data_feeder,
                              is_f1_reported,
                              ckpt_path,
                              n_saved_versions)


if __name__ == "__main__":
  from models.word_encoder_rnn import EncoderWordRNN
  from models.decoder_rnn import DecoderRNN
  from models.models import TaggingSeq2Seq

  encoder = EncoderWordRNN(n_layers=1, hidden_size=64, cell_type='lstm',
                           is_bidirectional=True,
                           max_seq_len=30, in_dropout=0.,
                           out_dropout=0., in_vocab_size=10,
                           is_packing_needed=False,
                           is_batch_first=True, out_embbed=7,
                           is_embed_pretrain=False)

  # input of batch=3, seqlen=5, each one is a single scalar index
  import numpy as np
  import torch
  from torch.autograd import Variable

  x = np.random.randint(0, 10, size=(3, 5)).astype(np.float)
  x = Variable(torch.LongTensor(x))
  print(x)

  # enc_out, enc_hidden = _encoder(x, [5, 5, 5])
  # print(enc_out.size())
  # print(enc_hidden[0].size(), enc_hidden[1].size())

  decoder = DecoderRNN(n_layers=1, hidden_size=128, cell_type='lstm',
                       is_bidirectional=False,
                       max_seq_len=30, in_dropout=0.,
                       out_dropout=0., out_vocab_size=200,
                       is_batch_first=True, is_embedding_used=True,
                       out_embbed=100,
                       is_embed_pretrain=False, is_attended=True,
                       attention_type='fixed'  # Ly == Lx
                       )

  y = np.random.randint(0, 10, size=(3, 5)).astype(np.float)
  y = Variable(torch.LongTensor(y))

  seq2seq = TaggingSeq2Seq(encoder, decoder)

  # uniformly init params
  for param in seq2seq.parameters():
    print(param.size())
    param.data.uniform_(-0.1, 0.1)

  # create a trainer
  train_loss = nn.NLLLoss()
  val_loss = nn.NLLLoss()
  lr = 1e-3
  is_tagging = True

  trainer = Trainer(seq2seq_model=seq2seq,
                    train_criterion=nn.CrossEntropyLoss(),
                    val_criterion=nn.CrossEntropyLoss(),
                    is_tagging=True,
                    learning_rate=lr,
                    optim_type='sgd',
                    print_freq=100)
