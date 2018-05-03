import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
  out = torch.randn(weights.size())
  out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
  return out


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    weight_shape = list(m.weight.data.size())
    fan_in = np.prod(weight_shape[1:4])
    fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
    w_bound = np.sqrt(6. / (fan_in + fan_out))
    m.weight.data.uniform_(-w_bound, w_bound)
    m.bias.data.fill_(0)
  elif classname.find('Linear') != -1:
    weight_shape = list(m.weight.data.size())
    fan_in = weight_shape[1]
    fan_out = weight_shape[0]
    w_bound = np.sqrt(6. / (fan_in + fan_out))
    m.weight.data.uniform_(-w_bound, w_bound)
    m.bias.data.fill_(0)


class AdaptiveActorCritic(torch.nn.Module):
  def __init__(self, max_beam_size, action_space=3):
    """ Format of a state (a vector): 
          1. accum_logP = state[0:self.max_beam_size]
          2. logP = state[self.max_beam_size:2 * self.max_beam_size]
          3. beam_size_in = int(state[2 * self.max_beam_size])
      Args
        max_beam_size
        action_space: number of possible actions, default is 3 
    """
    super(AdaptiveActorCritic, self).__init__()

    # FC layers
    hidden_size = max_beam_size * 2 + 1
    self.critic_linear = nn.Linear(hidden_size, 1)
    self.actor_linear = nn.Linear(hidden_size, action_space)

  def forward(self, state):
    state = Variable(torch.FloatTensor(state))

    return self.critic_linear(state), self.actor_linear(state)


if __name__ == "__main__":
  state = np.random.rand(21)
  print(state.shape)

  ac = AdaptiveActorCritic(max_beam_size=10, action_space=3)
  c, a = ac(state)
  print(a, c)