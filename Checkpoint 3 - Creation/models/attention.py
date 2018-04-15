"""
Graph component for Attention model of Batch_First style 
"""
import torch
import torch.nn.functional as F

from torch import nn


class Attention(nn.Module):
  """
  Right now we implement 3 types of attention: Bahdanau, Luong and Fixed 
  """
  def __init__(self, type='bahdanau', hidden_size=64, is_bidrectional=False):
    super(Attention, self).__init__()

    self.__type = type.lower()

    # Note: fixed attention is just a special case of Bahdanau
    if type.lower() in ['bahdanau', 'fixed']:
      multiply_factor = 2 if is_bidrectional else 1

      # luong attention does not have this layer
      self.bahdanau_layer = nn.Linear(2 * hidden_size * multiply_factor,
                                      hidden_size * multiply_factor)
    elif type.lower() != 'luong':
      raise ValueError("Not supported attention type!")

  def forward(self, dec_output, enc_output):
    """
    
    Args:
      dec_output: B x Ly x H2 
      enc_output: B x Lx x (H1 * bidirectional) (a.k.a context vector)  

    Returns:
      output: should retain the same _decoder output of B x Ly x (H * bi)  
      attention: B x Ly x Lx the relation of each word in Ly w.r.t Lx  
    """
    B = dec_output.size(0)
    Ly = dec_output.size(1)
    H = dec_output.size(2)  # actually it's (H * bi)
    Lx = enc_output.size(1)

    # requires for fixed attention Lx == Ly
    # UPdate: remove this criterion because we compare each sample of
    # dec_output and the whole input for FREE_RUN mode: Ly = 1, Lx != 1 now
    # if self.__type == 'fixed' and Lx != Ly:
    #   raise ValueError("Fixed Attention requires data and labels of the same "
    #                    "lengths.")

    # --------Phase 1: Similarity between Output and Context ----------
    activated_dec_output = None
    attention_energies = None
    if self.__type == 'fixed':
      activated_dec_output = dec_output
    else:
      attention = torch.bmm(dec_output,
                            enc_output.transpose(1,2))  # B x Ly x Lx
      attention = attention.view(-1, Lx)  # process the whole batch at once

      attention_energies = F.softmax(attention, dim=1)  # along 2nd dim (Lx)
      attention_energies = attention_energies.view(B, Ly, Lx)

      # luong attention will stop right here
      activated_dec_output = torch.bmm(attention_energies,
                                       enc_output)  # B x Ly x H

    # --------Phase 2: Learn necessary parameters on top of similarity---------
    attended_dec_output = None
    if self.__type == 'bahdanau' or self.__type == 'fixed':
      # WA + b is same as concat this activated_out and the original one and
      # learn the linear transoformation (W and b is combined and co-learned)
      concat_matrix = torch.cat((activated_dec_output, dec_output), dim=2)
      concat_matrix = concat_matrix.view(-1, 2*H)  # the whole batch at once

      # print("Concat_matrix = {}".format(concat_matrix.size()))

      # activate it with tanh after multiply with a learnable matrix
      attended_dec_output = F.tanh(self.bahdanau_layer(concat_matrix))

      # then return the previous size
      attended_dec_output = attended_dec_output.view(B, Ly, H)

    elif self.__type == 'luong':
      attended_dec_output = activated_dec_output
    else:
      raise ValueError("Not supported Attention type")

    return attended_dec_output, attention_energies

  @property
  def attention_type(self):
    return self.__type

if __name__ == "__main__":
  from torch.autograd import Variable

  # attention = Attention('bahdanau')
  attention = Attention('luong', 64)
  context = Variable(torch.randn(32, 6, 64))
  output = Variable(torch.randn(32, 1, 64))
  output, attn = attention(output, context)

  print(output.size(), attn.size())