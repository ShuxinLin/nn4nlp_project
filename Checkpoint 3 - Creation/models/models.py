from torch import nn
import torch.nn.functional as F


class GenericSeq2Seq(nn.Module):
  """ Will flow all the way from ENC to DEC """

  def __init__(self, encoder, decoder):
    super(GenericSeq2Seq, self).__init__()
    self._encoder = encoder
    self._decoder = decoder
    print("A Generic Seq2Seq model is called!")

  def forward(self, encoder_input, batch_source_lengths, decoder_input,
              teacher_forcing_ratio):
    encoder_out, encoder_hidden = \
      self._encoder(encoder_input, batch_source_lengths)

    return self._decoder.forward_and_decode_batch(decoder_input, encoder_hidden,
                                                  encoder_out,
                                                  teacher_forcing_ratio)

  # @property
  # def encoder(self):
  #   return self._encoder
  #
  # @property
  # def decoder(self):
  #   return self._decoder

  def flatten_parameters(self):
    """ For flattening params after restoring from a checkpoint"""
    self._encoder.rnn_cell.flatten_parameters()
    self._decoder.rnn_cell.flatten_parameters()


class TaggingSeq2Seq(GenericSeq2Seq):
  def __init__(self, encoder, decoder):
    super(TaggingSeq2Seq, self).__init__(encoder, decoder)
    print("A TAGGING Seq2Seq model is created!")

  def forward(self, encoder_input, batch_source_lengths, decoder_input,
              teacher_forcing_ratio, decode_method, is_tagging, is_training):
    encoder_out, encoder_hidden = \
      self._encoder(encoder_input, batch_source_lengths)

    return self._decoder. \
      forward_and_decode_batch(decoder_input=decoder_input,
                               encoder_hidden=encoder_hidden,
                               encoder_out=encoder_out,
                               teacher_forcing_ratio=teacher_forcing_ratio,
                               decode_method=decode_method,
                               is_tagging=is_tagging,
                               batch_source_lengths=batch_source_lengths,
                               is_training=is_training)

if __name__ == "__main__":
  from word_encoder_rnn import EncoderWordRNN
  from decoder_rnn import DecoderRNN

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

  seq2seq = GenericSeq2Seq(encoder, decoder)

  result_dict = seq2seq(encoder_input=x, batch_source_lengths=[5, 5, 5],
                        decoder_input=y, teacher_forcing_ratio=0.5)

  seq2seq = TaggingSeq2Seq(encoder, decoder)
  print("Training Tagging mode")
  result_dict = seq2seq(x, [5, 4, 3], y,
                        0.5, 'greedy', True,  # is tagging
                        True)  # is train

  print("Testing Tagging mode")
  result_dict = seq2seq(x, [5, 4, 3], None,
                        0.0, 'greedy', True,
                        False)
  # import pdb; pdb.set_trace()