import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ActorCritic(torch.nn.Module):
  def __init__(self, num_inputs, action_space):
    super(ActorCritic, self).__init__()
    self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

    self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

    num_outputs = action_space.n
    self.critic_linear = nn.Linear(256, 1)
    self.actor_linear = nn.Linear(256, num_outputs)

    self.apply(weights_init)
    self.actor_linear.weight.data = normalized_columns_initializer(
      self.actor_linear.weight.data, 0.01)
    self.actor_linear.bias.data.fill_(0)
    self.critic_linear.weight.data = normalized_columns_initializer(
      self.critic_linear.weight.data, 1.0)
    self.critic_linear.bias.data.fill_(0)

    self.lstm.bias_ih.data.fill_(0)
    self.lstm.bias_hh.data.fill_(0)

    self.train()

  def forward(self, inputs):
    inputs, (hx, cx) = inputs
    x = F.elu(self.conv1(inputs))
    x = F.elu(self.conv2(x))
    x = F.elu(self.conv3(x))
    x = F.elu(self.conv4(x))

    x = x.view(-1, 32 * 3 * 3)
    hx, cx = self.lstm(x, (hx, cx))
    x = hx

    return self.critic_linear(x), self.actor_linear(x), (hx, cx)


class AdativeActorCritic(torch.nn.Module):
  def __init__(self, max_beam_size, action_space=3):
    """ Format of a state (a vector): 
          1. accum_logP = state[0:self.max_beam_size]
          2. logP = state[self.max_beam_size:2 * self.max_beam_size]
          3. beam_size_in = int(state[2 * self.max_beam_size])
      Args
        max_beam_size
        action_space: number of possible actions, default is 3 
    """
    super(AdativeActorCritic, self).__init__()
    # self.max_beam_size = max_beam_size
    #
    # # RNN layer - for the whole sequence
    # hidden_size = 32
    # self.logp_lstm = nn.LSTM(max_beam_size, hidden_size)
    # self.accum_lstm = nn.LSTM(max_beam_size, hidden_size)
    #
    # # FC layers
    hidden_size = max_beam_size * 2 + 1
    self.critic_linear = nn.Linear(hidden_size, 1)
    self.actor_linear = nn.Linear(hidden_size, action_space)

  def forward(self, state):
    # accum_logP = state[0:self.max_beam_size].reshape(1, 1, -1)
    # logP = state[self.max_beam_size:2 * self.max_beam_size].reshape(1, 1, -1)
    #
    # # concatenate 2 scores
    # cat_scores = np.concatenate([accum_logP, logP], axis=1)
    # print(cat_scores.shape)
    #
    # beam_size_in = int(state[2 * self.max_beam_size])
    #
    # # to variables
    # accum_logP = Variable(torch.FloatTensor(accum_logP))
    # logP = Variable(torch.FloatTensor(logP))
    #
    # hx, cx = self.logp_lstm(accum_logP)
    # print(hx)
    # print(cx)

    # TODO: turn to RNN
    state = Variable(torch.FloatTensor(state))

    return self.critic_linear(state), self.actor_linear(state)


if __name__ == "__main__":
  state = np.random.rand(21)
  print(state.shape)

  ac = AdativeActorCritic(max_beam_size=10, action_space=3)
  c, a = ac(state)
  print(a, c)