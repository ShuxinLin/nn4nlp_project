#!/usr/bin/python3

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import os


class DetermAgent():
  def __init__(self):
    self.DECREASE = 0
    self.SAME = 1
    self.INCREASE = 2


  # state is expected to be:
  # [max_beam_size of accumulated logP's (sorted from high to low),
  #  max_beam_size of (non-accumulated) logP's (sorted from high to low),
  #  current (incoming) beam size]
  # May later be appended with (part of the following):
  # [input sequence length,
  #  current time step in the decoding process,
  #  the final output hidden vector from encoder (a representation of X),
  #  the output hidden vectors of decoder at this time step from the first incoming beam]
  #
  # action is:
  # 1: no action; keep the same beam size
  # 2: increase the beam size by 1 (would be no effect in environment if the current beam size is already max_beam_size; the agent can in principle output this action---especially later when it is trained by RL---but it simply won't cause the beam size to really increase by 1 in the environment.)
  # 0: decrease the beam size by 1 (similar to stated above: would be no effect if the current beam size is already 1.)
  def get_action(self, state, max_beam_size, accum_logP_ratio_low, logP_ratio_low):
    accum_logP = state[0:max_beam_size]
    logP = state[max_beam_size:2 * max_beam_size]
    beam_size_in = state[2 * max_beam_size]

    # Determine whether to decrease the beam size
    # Only observe the accum_logP at (beam_size_in - 1)
    # If it is lower than or equal to the highest accum_logP (at position 0) by the ratio accum_logP_ratio_low, then decrease the size by 1
    #
    # Note that in log space it is:
    # log P_B - log P_1 <= log ratio
    if beam_size_in > 1:
      if accum_logP[beam_size_in - 1] - accum_logP[0] <= np.log(accum_logP_ratio_low):
        return self.DECREASE
      # Do similar with logP
      if logP[beam_size_in - 1] - logP[0] <= np.log(logP_ratio_low):
        return self.DECREASE

    # Determine whether to increase the beam size
    # Only observe the accum_logP at beam_size_in (that is, the new candidate in the increased beam)
    # If it is higher than the highest accum_logP (at position 0) by the ratio accum_logP_ratio_low, then increase the size by 1 (kind of like: if the current beam size was one size larger (beam_size_in + 1), it would not be pruned)
    if beam_size_in < max_beam_size:
      if accum_logP[beam_size_in] - accum_logP[0] > np.log(accum_logP_ratio_low):
        return self.INCREASE
      # Do similar with logP
      if logP[beam_size_in] - logP[0] > np.log(logP_ratio_low):
        return self.INCREASE

    # Otherwise, keep the same beam size
    return self.SAME
