"""
Templaet for RNN which will be reused by Encoder and Decoder in Seq2Seq
"""

from torch import nn


class GenericRNN(nn.Module):
  """ This one will override torch.nn.Module"""
  def __init__(self, n_layers=1, hidden_size=64, cell_type='LSTM',
               is_bidirectional=False, max_seq_len=30, in_dropout=0.,
               out_dropout=0.):
    """
    
    Args:
      n_layers: number of stacked RNN layers 
      hidden_size: number of hidden length 
      cell_type: LSTM, GRU
      is_bidirectional: this params is to be fed into RNN cell constructor 
      max_seq_len: 
      in_dropout: probability of dropout for input
      out_dropout: probability of dropout for output 
    
    Raises:
      ValueError if cell type is not supported 
    """
    super(GenericRNN, self).__init__()

    self._n_layers = n_layers
    self._hidden_size = hidden_size
    self._max_seq_len = max_seq_len
    self._in_dropout_layer = nn.Dropout(p=in_dropout)
    self._out_dropout_prob = out_dropout

    # right now we only suppoer LTSM and GRU. Will be extending to NAS, SRU
    cells = {'lstm': nn.LSTM, 'gru': nn.GRU}
    try:
      self._cell_type = cells[cell_type.lower()]
    except:
      raise ValueError("Cell type not supported!")

    self._is_bidirectional = is_bidirectional

  def forward(self, *input):
    raise NotImplementedError("Generic RNN forward path is not implemented")


if __name__ == "__main__":
  generic_rnn = GenericRNN(1, 2, 'lstm', 3, 0.1, 0.2)
  print(generic_rnn)
  generic_rnn.forward()