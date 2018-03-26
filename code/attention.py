"""
Graph component for Attention model
"""
import torch
import torch.nn.functional as F

from torch import nn


class Attention(nn.Module):
  def __init__(self, type='bahdanau'):
    super(Attention, self).__init__()

    if type.lower() == 'bahdanau':
      self.__type = 'bahdanau'
      self.bahdanau_layer = None
    elif type.lower() == 'luong':
      pass
    else:
      raise ValueError("Not supported attention type!")

  def forward(self, dec_output, enc_output):
    """
    Args:
      dec_output: Ly x B x H
      enc_output: Lx x B x H

    Returns:
      output: should retain the same decoder output of Ly x B x H
      attention: B x Ly x Lx the relation of each word in Ly w.r.t. Lx
    """
    # B = dec_output.size(0)
    # Ly = dec_output.size(1)
    # H = dec_output.size(2)
    # Lx = enc_output.size(1)
    B = dec_output.size(1)
    Ly = dec_output.size(0)
    H = dec_output.size(2)
    Lx = enc_output.size(0)

    # Reshape input from shape Ly x B x H into B x Ly x H
    # for the convenience of further processing
    dec_output = dec_output.transpose(1, 0)
    enc_output = enc_output.transpose(1, 0)
    # Compute score
    attention = torch.bmm(dec_output,
                          enc_output.transpose(1,2))  # B x Ly x Lx
    attention = attention.view(-1, Lx)  # process the whole batch at once

    # The alpha_{ij} coefficients
    attention_energies = F.softmax(attention, dim=1)  # along 2nd dimension (Lx)
    attention_energies = attention_energies.view(B, Ly, Lx)

    # Compute context vector
    activated_dec_output = torch.bmm(attention_energies, enc_output)  # B x Ly x H

    # WA + b is same as concat this activated_out and the original one and
    # learn the linear transformation (W and b is combined and co-learned)
    concat_matrix = torch.cat((activated_dec_output, dec_output), dim=2)
    # Process the whole batch at once
    concat_matrix = concat_matrix.view(-1, 2*H)

    # Activate it with tanh after multiplying with a learnable matrix
    if self.__type == 'bahdanau':
      self.bahdanau_layer = nn.Linear(2*H, H)
      attended_dec_output = F.tanh(self.bahdanau_layer(concat_matrix))

      # Then recover the previous shape
      attended_dec_output = attended_dec_output.view(B, Ly, H)

    elif self.__type == 'luong':
      attended_dec_output = None

    # Return attended_dec_output, attention_energies
    # Reshape back into Ly x B x H, Ly x B x Lx
    return attended_dec_output.transpose(1, 0), \
           attention_energies.transpose(1, 0)


if __name__ == "__main__":
  from torch.autograd import Variable

  attention = Attention('bahdanau')
  # context = Variable(torch.randn(32, 6, 64))
  # output = Variable(torch.randn(32, 1, 64))
  context = Variable(torch.randn(6, 32, 64))
  output = Variable(torch.randn(1, 32, 64))
  output, attn = attention(output, context)

  print (output.size(), attn.size())
