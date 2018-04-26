"""
Implementation of Decoder in Seq2Seq 
"""
import numpy as np
import random
import torch
import torch.nn.functional as F
import sys

from torch import nn
from torch.autograd import Variable

from attention import Attention
from generic_rnn import GenericRNN


class DecoderRNN(GenericRNN):
  """
  This Encoder will operate on batch-size basis, so the lengths of input is 
  different from batch to batch. 

  Sicne we use packing of batch, `is_batch_first` must always be True 

  """
  # TODO: add label embed as input parameters
  def __init__(self, n_layers=1, hidden_size=64, cell_type='LSTM',
               is_bidirectional=False, max_seq_len=30, in_dropout=0.,
               out_dropout=0., out_vocab_size=10,
               is_batch_first=True,
               is_embedding_used=True, out_embbed=100,
               is_embed_pretrain=False,
               is_attended=False, attention_type='bahdanau',
               is_softmax_output_needed=True,
               SOS_ID=0, EOS_ID=1,
               ):

    super(DecoderRNN, self).__init__(n_layers, hidden_size, cell_type,
                                     is_bidirectional, max_seq_len,
                                     in_dropout, out_dropout)
    if is_embedding_used:
      self.__embed_layer = nn.Embedding(out_vocab_size, out_embbed)
      self.__out_embed = out_embbed

    # note: we optionally apply dropout of this cell output
    # output == hidden_size
    self.__cell = self._cell_type(out_embbed, hidden_size, n_layers,
                                  batch_first=is_batch_first,
                                  bidirectional=is_bidirectional,  # False
                                  dropout=out_dropout)

    self.__vocab_size = out_vocab_size

    # get the real hidden size
    multiply_factor = 2 if is_bidirectional else 1
    self.H = multiply_factor * hidden_size

    # default is bahdanau attention
    self.__attention = None
    if is_attended:
      self.__attention = Attention(attention_type, hidden_size,
                                   is_bidirectional)

    # this one is specific for _decoder, will apply for each token
    # which will produce pre-softmax scores over the whole vocabulary
    self.__last_linear_layer = nn.Linear(hidden_size, out_vocab_size)
    # print("Constructing last linear layer: {} of V={}".
    #       format(self.__last_linear_layer, out_vocab_size))

    self.__is_batch_first = is_batch_first

    # need softmax output as activations of this decoder or not?
    self.__is_softmax_output_needed = is_softmax_output_needed

    # special tokens
    self.__SOS_ID = SOS_ID
    self.__EOS_ID = EOS_ID

    # label embedding layer (10 => 8 )
    # TODO: change this
    self.__label_embed = nn.Embedding(len(self._label_to_idx), 8)

  @property
  def rnn_cell(self):
    """ For flattening parameters after restoring from a checkpoint """
    return self.__cell

  @staticmethod
  def __init_hidden(last_encoder_hidden, is_encoder_bidirectional=True):
    """
    TODO: to init hidden properly if the architecture of DEC is different from 
          ENC. 
    
    Args:
      last_encoder_hidden: is the last output of hidden from _encoder                       
      is_encoder_bidirectional: affects significantly the init process 
    Returns: init hidden layer 
    """

    # def transform_enc_hidden(h, is_encoder_bidirectional):
    #   """
    #   Because _decoder is always unidirectional, we need to transform the
    #   hidden state of _encoder in case it's bidirectional
    #   dimension: (n_layers * bi) x B x H
    #
    #   We change to (n_layers) x B x (H * bi)
    #   """
    #   if is_encoder_bidirectional:
    #     return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    #
    #   return h
    #
    # # for LSTM, hidden is a tuple of (h_n, c_n)
    # if isinstance(last_encoder_hidden, tuple):
    #   decoder_hidden = tuple([transform_enc_hidden(h, is_encoder_bidirectional)
    #                           for h in last_encoder_hidden])
    # else:
    #   # GRU case
    #   decoder_hidden = transform_enc_hidden(last_encoder_hidden,
    #                                         is_encoder_bidirectional)
    #
    # return decoder_hidden

    def init_single_hidden(h):
      # sum up the encoder hidden of size (n_layer * bi) x B x H
      half = h.size(0) // 2
      return h[:half, :, :] + h[half:, :, :]

    if is_encoder_bidirectional:
      if isinstance(last_encoder_hidden, tuple):
        first, last = last_encoder_hidden[0], last_encoder_hidden[1]
        return (init_single_hidden(first), init_single_hidden(last))
      else:
        return init_single_hidden(last_encoder_hidden)

    return last_encoder_hidden

  @staticmethod
  def __beam_decode(token_idx, token_output, symbols_idx_seq, lengths, k=1):
    raise NotImplementedError("This base class uses Greedy decoding")

  def __greedy_decode(self, token_idx, token_output, symbols_idx_seq, lengths,
                      is_tagging):
    """
    Take the best result at each time step. Period! 
    
    Args:
      token_idx: step index 
      token_output: B x 1 x V 
      symbols_idx_seq: 
      lengths: 
      is_tagging: in tagging case, Lx == Ly so we know beforehand how many 
                  tokens to be decoded (simplest case)  

    Returns: indices (== words) decoded for the batch (size B) 

    """
    decode_values, decode_idxs = token_output.topk(1)
    # print("In greedy decode: Token output size {} decode idxs {}"
    #       .format(token_output.size(), decode_idxs.size()))

    symbols_idx_seq.append(decode_idxs)  # get idx of top1 only

    # in tagging, we know beforehand how many steps
    # if not is_tagging:
    # at this step, check whether the _decoder predicts EOS token or not
    is_eos_batch = decode_idxs.data.eq(self.__EOS_ID)

    if is_eos_batch.dim() > 0:  # need to squeeze if needed
      is_eos_batch = is_eos_batch.cpu().view(-1).numpy()

      # first check whether we reach the max length or not
      # then if not, we check whether we have predicted EOS
      update_idx = ((lengths > token_idx) & is_eos_batch) == True

      # mark the sentence (in batch) where this step has EOS
      lengths[update_idx] = len(symbols_idx_seq)

    return decode_idxs

  def forward(self, decoder_input, decoder_hidden_state_0, encoder_out):
    """
    Unlike in Encoder where we can packed steps at once. This forward() 
    in the _decoder performs each time step at a time. 

    Args:
      decoder_input: should be batch_first=True format: B x Ly x D where 
                     D is the embedding dimension  
      decoder_hidden_state_0: previous hidden state, h0 will be the last 
                              hidden of _encoder
        ** NOTE ** it is a tuple of `(h_t, c_t)` 
      encoder_out: B x Ly x (H * bidirectional) 

    Returns: 
      output (B x Ly x V), 
      hidden state (h_t, c_t) if LSTM else (h_t) if GRU 
      attention scores (if there is attention, otherwise None) 
    """
    B = decoder_input.size(0)
    Ly = decoder_input.size(1)
    # print("B={} Ly={}".format(B, Ly))

    # same as encodeer
    # print('_decoder input: {}'.format(decoder_input.size()))
    embedded = self.__embed_layer(decoder_input)
    # print('embed: {}'.format(embedded.size()))

    embedded = self._in_dropout_layer(embedded)
    # print("Embed Decoder output: {}".format(embedded.size()))

    # TODO: see whether we need to apply activation (e.g. ReLU before this?
    # print("Hidden: {}".format(decoder_hidden_state[0].size()))
    out, decoder_hidden_state = self.__cell(embedded, decoder_hidden_state_0)

    # print("Out size = {}".format(out.size()))
    # print("Encoder out = {}".format(encoder_out.size()))

    attention_scores = None
    if self.__attention:
      # *** NOTE ***: for Tagging, attention_scores is None
      out, attention_scores = self.__attention(out, encoder_out)

    # make sure output has congiguous blocks of memory
    # https://discuss.pytorch.org/t/runtimeerror-input-is-not-contiguous/930
    out = out.contiguous()
    # print("Interim output: {}".format(out.size()))

    # 'expand' it to the output vocab size
    out = self.__last_linear_layer(out.view(-1, self._hidden_size))
    # print("After last linear layer: {}".format(out.size()))

    if self.__is_softmax_output_needed:
      # apply softmax along hidden dimension
      # this also work even the hidden size is double (bidirectional)
      activated_scores = F.log_softmax(out, dim=-1)

      # make the orientation to B x Ly x V
      activated_scores = activated_scores.view(B, Ly, -1)
    else:
      activated_scores = out.view(B, Ly, -1)

    # print("In forward, input={} => output={} hidden={}".
    #       format(decoder_input.size(), activated_scores.size(),
    #              decoder_hidden_state[0].size()))

    return activated_scores, decoder_hidden_state, attention_scores

  def forward_and_decode_batch(self, decoder_input, encoder_hidden, encoder_out,
                               teacher_forcing_ratio, decode_method='greedy',
                               is_tagging=False, batch_source_lengths=None,
                               is_training=True):
    """
    This public API will be called by Seq2Seq model 
    
    **NOTE:** will proceed the whole batch in 1 consistent mode: 
              teacher-forcing or free-run mode. 
    
    Args:
      decoder_input: B x Ly 
      encoder_hidden: (n_layers * bi) x B x Hx  
      encoder_out: B x Lx x (Hx * bi) 
      teacher_forcing_ratio: 
      decode_method: greedy, beam search or adaptive beam search 

      is_tagging: simple case where Ly == Lx so we copy source lengths to 
                  decoded lengths
      batch_source_lengths: only used when `__is_tagging` is turned on  
      
      is_training: if in training mode, batch size is 1, no _decoder input is 
                   provided and __teacher_forcing_ratio must be ZERO 

    Returns: a dictionary of all information 
              
    """

    # size of batch for testing mode
    B = encoder_hidden[0].size(1) if isinstance(encoder_hidden, tuple) else \
      encoder_hidden.size(1)

    # update batch size if on training mode (if using bucket batches, it varies)
    if is_training:
      B = decoder_input.size(0)
    else:
      assert teacher_forcing_ratio == 0, \
        'Testing mode, there must be strictly *** NO *** Teacher Forcing mode.'

    if decoder_input is None:
      max_decode_length = self._max_seq_len  # training free-run mode
      if is_tagging:
        # testing in Tagging mode
        # max_decode_length = torch.max(batch_source_lengths).data[0]
        max_decode_length = encoder_out.size(1)  # B x Lx x H
    else:
      # max_decode_length = decoder_input.size(1) - 1  # (Ly - 1) training mode
      max_decode_length = decoder_input.size(1)  # training mode

      # move input to GPU
      if torch.cuda.is_available():
        decoder_input = decoder_input.cuda()

    # init hidden state properly - IMPORTANT! (esp. without attention)
    decoder_hidden_state_0 = DecoderRNN.__init_hidden(encoder_hidden)

    # helper for greedy
    decoder_outputs = []  # store output for all time steps
    symbols_idx_seq = []  # store decoded sequence
    attn_outputs = []  # store attention scores of all time steps
    batch_decode_lengths = np.array([max_decode_length] * B)

    is_teacher_forcing = \
      True if random.random() < teacher_forcing_ratio else False

    input = Variable(torch.LongTensor([self.__SOS_ID] * B))
    input = input.unsqueeze(1)  # B x 1

    if torch.cuda.is_available():
      input = input.cuda()
    # -------------------------------------------------------------------------
    #                     GUIDANCE TRAINING MODE
    # -------------------------------------------------------------------------
    if is_teacher_forcing:
      # print("Teacher forcing mode {}".format(type(decoder_input)))
      # get input from Y_1 => Y_(T-1)

      # init_sos_batch = Variable(torch.LongTensor([self.__SOS_ID] * B))  # B
      # init_sos_batch = init_sos_batch.unsqueeze(1)  # B x 1

      # if torch.cuda.is_available():
      #   init_sos_batch = init_sos_batch.cuda()

      # if sentence has only 1 word - simple case
      # if decoder_input.size(1) == 1:
      #   input = init_sos_batch
      # else:
      if decoder_input.size(1) > 1:
        input = torch.cat([input, decoder_input[:, :-1]], dim=1)

      # get the whole output sequence
      activated_scores, decoder_hidden_state, attention_scores = \
        self.forward(input, decoder_hidden_state_0, encoder_out)

      # decode each token of activated scores: B x Ly x H
      for token_idx in range(activated_scores.size(1)):
        token_output = activated_scores[:, token_idx, :]  # B x V

        # populate results
        decoder_outputs.append(token_output)

        # populate attention outputs
        token_attn_scores = None
        if self.__attention:
          # B x 1 x Lx
          token_attn_scores = attention_scores[:, token_idx, :] if \
            self.__attention.attention_type != 'fixed' else None
          attn_outputs.append(token_attn_scores)

        # update symbols
        if decode_method.lower() == 'greedy':
          self.__greedy_decode(token_idx, token_output,
                               symbols_idx_seq, batch_decode_lengths,
                               is_tagging)

    # -------------------------------------------------------------------------
    #     FREE-RUN MODE: for EVALUATION (TEST) and NO_GUIDANCE TRAINING
    # -------------------------------------------------------------------------
    else:  # free-run mode, supply next decoded token as new input
      # if is_training:
      #   input = decoder_input[:, 0].unsqueeze(1)  # B x 1
      # else:
      # in testing mode, input is None, init with SOS

      # input = Variable(torch.LongTensor([self.__SOS_ID] * B))
      # input = input.unsqueeze(1)  # B x 1
      #
      # if torch.cuda.is_available():
      #   input = input.cuda()

      decoder_hidden_state = decoder_hidden_state_0  # will be overwritten
      for token_idx in range(max_decode_length):
        # print("step input: {}".format(input.size()))
        activated_scores, decoder_hidden_state, attention_scores = \
          self.forward(input, decoder_hidden_state_0, encoder_out)

        # print(activated_scores.size(), decoder_hidden_state[0].size(),
        #       attention_scores.size())

        # output size is B x 1 x V  -> return back to B x V only
        activated_scores = activated_scores.squeeze(1)
        # print(activated_scores.size())

        # populate stats
        decoder_outputs.append(activated_scores)
        attn_outputs.append(attention_scores)

        # decode symbols
        decode_idx = None
        if decode_method.lower() == 'greedy':
          decode_idx = self.__greedy_decode(token_idx, activated_scores,
                                            symbols_idx_seq,
                                            batch_decode_lengths,
                                            is_tagging)
        elif decode_method.lower() == 'beam':
          raise NotImplementedError

        # update input for the next decoding step in free-run mode
        # print(decode_idx)
        input = decode_idx

    # for tagging case, we know exactly the decoded length for each sentence
    # if is_tagging:
    #   batch_decode_lengths = batch_source_lengths

    return {'decoder_output': decoder_outputs,
            'decoder_hidden_state': decoder_hidden_state,
            'symbols_idx_seq': symbols_idx_seq,
            'attn_outputs': attn_outputs,
            'decode_lengths': batch_decode_lengths}


if __name__ == "__main__":
  from word_encoder_rnn import EncoderWordRNN

  cell_type = 'lstm'
  # cell_type = 'gru'

  encoder = EncoderWordRNN(n_layers=2, hidden_size=64, cell_type=cell_type,
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
  # print(x)

  enc_out, enc_hidden = encoder(x, [5, 4, 3])
  print(enc_out.size())
  print(enc_hidden[0].size(), enc_hidden[1].size(), type(enc_hidden))
  print("*" * 100)

  decoder = DecoderRNN(n_layers=2, hidden_size=64, cell_type=cell_type,
                       is_bidirectional=False,
                       max_seq_len=30, in_dropout=0.,
                       out_dropout=0., out_vocab_size=200,
                       is_batch_first=True, is_embedding_used=True,
                       out_embbed=100,
                       is_embed_pretrain=False, is_attended=False,
                       attention_type='bahdanau'
                       )

  y = np.random.randint(0, 10, size=(3, 7)).astype(np.float)
  y = Variable(torch.LongTensor(y))

  print("*" * 100)
  print('TRAINING MODE')
  results = decoder.forward_and_decode_batch(decoder_input=y,
                                             encoder_hidden=enc_hidden,
                                             encoder_out=enc_out,
                                             teacher_forcing_ratio=0.,
                                             decode_method='greedy',
                                             is_tagging=False,
                                             batch_source_lengths=None,
                                             is_training=True)

  for o in results['decoder_output']:
    print(o.size(), )
  print(results['decode_lengths'])

  print("*" * 100)
  print('TESTING MODE')
  results = decoder.forward_and_decode_batch(decoder_input=None,
                                             encoder_hidden=enc_hidden,
                                             encoder_out=enc_out,
                                             teacher_forcing_ratio=0,
                                             decode_method='greedy',
                                             is_tagging=False,
                                             batch_source_lengths=None,
                                             is_training=False)

  for o in results['decoder_output']:
    print(o.size(), )
  print(results['decode_lengths'])

  print("*" * 100)
  print('TAGGING MODE')
  y = np.random.randint(0, 10, size=(3, 5)).astype(np.float)
  y = Variable(torch.LongTensor(y))

  decoder = DecoderRNN(n_layers=2, hidden_size=64, cell_type=cell_type,
                       is_bidirectional=False,
                       max_seq_len=30, in_dropout=0.,
                       out_dropout=0., out_vocab_size=200,
                       is_batch_first=True, is_embedding_used=True,
                       out_embbed=100,
                       is_embed_pretrain=False, is_attended=True,
                       attention_type='fixed'  # Ly == Lx
                       )

  print("Training TAGGING mode")
  results = decoder.forward_and_decode_batch(decoder_input=y,
                                             encoder_hidden=enc_hidden,
                                             encoder_out=enc_out,
                                             teacher_forcing_ratio=0.5,
                                             decode_method='greedy',
                                             is_tagging=True,
                                             batch_source_lengths=[5, 4, 3],
                                             is_training=True)
  print(results['symbols_idx_seq'][0])

  print("*" * 100)
  print("Testing TAGGING mode")
  results = decoder.forward_and_decode_batch(decoder_input=None,
                                             encoder_hidden=enc_hidden,
                                             encoder_out=enc_out,
                                             teacher_forcing_ratio=0,
                                             decode_method='greedy',
                                             is_tagging=True,
                                             batch_source_lengths=[5, 4, 3],
                                             is_training=False)

  for o in results['decoder_output']:
    print(o.size(), )
  print(results['decode_lengths'], len(results['decode_lengths']))
