"""
Implementation of Encoder in Seq2Seq 
"""
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from generic_rnn import GenericRNN


class EncoderWordRNN(GenericRNN):
  """
  This Encoder will operate on batch-size basis, so the lengths of input is 
  different from batch to batch. 
  
  Sicne we use packing of batch, `is_batch_first` must always be True 
  
  """
  def __init__(self, n_layers=1, hidden_size=64, cell_type='LSTM',
               is_bidirectional=False, max_seq_len=30, in_dropout=0.,
               out_dropout=0., in_vocab_size=10, is_packing_needed=True,
               is_batch_first=True, is_embedding_used=True, out_embbed=100,
               is_embed_pretrain=False):

    super(EncoderWordRNN, self).__init__(n_layers, hidden_size, cell_type,
                                         is_bidirectional, max_seq_len,
                                         in_dropout, out_dropout)

    # embedding will turn (B x L) -> (B x L x E)
    # if not using this, remove this layer and make input B x L x E
    # and feed it directly to RNN cell
    if is_embedding_used:
      self.__embed_layer = nn.Embedding(in_vocab_size, out_embbed)
      self.__out_embed = out_embbed

    # note: we optionally apply dropout of this cell output
    # output == hidden_size
    self.__cell = self._cell_type(out_embbed, hidden_size, n_layers,
                                  batch_first=is_batch_first,
                                  bidirectional=is_bidirectional,
                                  dropout=out_dropout)


    # zero padding variable-length input of a batch or not
    self.__is_packing_needed = is_packing_needed
    self.__is_batch_first = is_batch_first

  @property
  def rnn_cell(self):
    """ For flattening parameters after restoring from a checkpoint """
    return self.__cell

  def forward(self, input_batch, batch_element_lengths=None):
    """
    Pack embedding (should be zero-padded) and produce output of RNN 
    
    Args:
      input_batch: normally (B, L, D) if batch_first=True, B = batch_size, 
                   L = max-seq-len and D is n_features 
      batch_element_lengths: individual (variable) lengths of each element 

    Returns: output and final hidden state (for GRU it's the same)
     ** NOTE ** hidden state output is a tuple `(h_T, c_T)` 

    """
    # IF IS_EMBED USED THEN
    # note: we optionally apply dropout of this embedding layer
    embedded = self.__embed_layer(input_batch)
    embedded = self._in_dropout_layer(embedded)
    # print("Embed output: {}".format(embedded.size()))

    # optionally prepare input for RNN processing
    if self.__is_packing_needed:
      embedded = pack_padded_sequence(input=embedded,
                                      lengths=batch_element_lengths,
                                      batch_first=self.__is_batch_first)
      print("Packed embedded: {}".format(embedded))

    # calculating output through hidden_size steps recurrently
    # out: B x L x (hidden_size * num_directions)
    # hidden: (num_layers * num_directions, batch, hidden_size)
    out, hidden = self.__cell(embedded)

    # unpack if previously packed
    if self.__is_packing_needed:
      out, _ = pad_packed_sequence(out, self.__is_batch_first)
      # print("Unpacked output: {}".format(out))

    # get the `true` output if bidirectional

    # TODO: return raw otuput and hidden and let DECODER to deal with that
    if self._is_bidirectional:
      out = out[:, :, :self._hidden_size] + out[:, :, self._hidden_size:]

    return out, hidden


if __name__ == "__main__":
  encoder = EncoderWordRNN(n_layers=2, hidden_size=64, cell_type='lstm',
                           is_bidirectional=True,
                           max_seq_len=30, in_dropout=0.,
                           out_dropout=0., in_vocab_size=10,
                           is_packing_needed=True,
                           is_batch_first=True, out_embbed=7,
                           is_embed_pretrain=False)

  # input of batch=3, seqlen=5, each one is a single scalar index
  import numpy as np
  import torch
  from torch.autograd import Variable

  x = np.random.randint(0, 10, size=(3, 5)).astype(np.float)
  x = Variable(torch.LongTensor(x))
  print(x)

  out, hidden = encoder(x, [5, 5, 5])
  print(out.size(), type(out.size(1)))
  print(hidden[0].size(), hidden[1].size())
