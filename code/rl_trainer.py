import itertools
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import AdaptiveActorCritic
from torch.autograd import Variable


def ensure_shared_grads(model, shared_model):
  for param, shared_param in zip(model.parameters(),
                                 shared_model.parameters()):
    if shared_param.grad is not None:
      return
    shared_param._grad = param.grad


def train_adaptive(rank,
                   machine,
                   max_beam_size,
                   shared_model,
                   counter,
                   lock,
                   optimizer,
                   data_X, data_Y, index2word, index2label,
                   suffix, decode_method, beam_size,
                   reward_coef_fscore, reward_coef_beam_size,
                   f_score_index_begin,
                   args):
  torch.manual_seed(123 + rank)

  logfile = open(os.path.join(args.logdir, "log_" + str(rank) + ".txt"), "w+")

  # create adaptive model
  model = AdaptiveActorCritic(max_beam_size=max_beam_size, action_space=3)
  # If a shared_optimizer is not passed in
  if optimizer is None:
    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
  # torch.nn.modules.module:
  # Sets the module in training mode
  model.train()

  batch_num = len(data_X)

  for epoch in range(0, args.n_epochs):
    reward_list = []

    # shuffle
    batch_idx_list = range(batch_num)
    batch_idx_list = np.random.permutation(batch_idx_list)

    time_begin = time.time()
    for batch_idx in batch_idx_list:
      sen = data_X[batch_idx]
      label = data_Y[batch_idx]

      current_batch_size = len(sen)
      current_sen_len = len(sen[0])

      # DEBUG
      # print(batch_idx, current_sen_len)
      if current_sen_len < 3:  # ignore sentence having tiny length
        continue

      sen_var = Variable(torch.LongTensor(sen))
      label_var = Variable(torch.LongTensor(label))

      if machine.gpu:
        sen_var = sen_var.cuda()
        label_var = label_var.cuda()

      # Initialize the hidden and cell states
      # The axes semantics are
      # (num_layers * num_directions, batch_size, hidden_size)
      # So 1 for single-directional LSTM encoder,
      # 2 for bi-directional LSTM encoder.
      init_enc_hidden = Variable(
        torch.zeros((2, current_batch_size, machine.hidden_dim)))
      init_enc_cell = Variable(
        torch.zeros((2, current_batch_size, machine.hidden_dim)))

      if machine.gpu:
        init_enc_hidden = init_enc_hidden.cuda()
        init_enc_cell = init_enc_cell.cuda()

      enc_hidden_seq, (enc_hidden_out, enc_cell_out) = machine.encode(sen_var,
                                                                   init_enc_hidden,
                                                                   init_enc_cell)

      # The semantics of enc_hidden_out is (num_layers * num_directions,
      # batch, hidden_size), and it is "tensor containing the hidden state
      # for t = seq_len".
      #
      # Here we use a linear layer to transform the two-directions of the dec_hidden_out's into a single hidden_dim vector, to use as the input of the decoder
      init_dec_hidden = machine.enc2dec_hidden(
        torch.cat([enc_hidden_out[0], enc_hidden_out[1]], dim=1))
      init_dec_cell = machine.enc2dec_cell(
        torch.cat([enc_cell_out[0], enc_cell_out[1]], dim=1))

      # ===================================
      if decode_method == "adaptive":
        # the input argument "beam_size" serves as initial_beam_size here
        # TODO: implement this here
        label_pred_seq, accum_logP_pred_seq, logP_pred_seq, \
        attention_pred_seq, episode, sen_beam_size_seq, total_reward = \
          decode_one_sentence_adaptive_rl(machine,
          current_sen_len, init_dec_hidden, init_dec_cell, enc_hidden_seq,
          beam_size, max_beam_size, model, shared_model, reward_coef_fscore,
          reward_coef_beam_size, label_var, f_score_index_begin, counter, lock,
          optimizer, args)

        reward_list.append(total_reward)

      else:
        raise Exception("Not implemented!")
      # ===================================

      # update beam seq
      #beam_size_seqs.append(sen_beam_size_seq)

      ### Debugging...
      # print("input sentence =", sen)
      # print("true label =", label)
      # print("predicted label =", label_pred_seq)
      # print("episode =", episode)
    # End for batch_idx
    time_end = time.time()
    time_used = time_end - time_begin

    reward_list = np.array(reward_list)
    reward_mean = np.mean(reward_list)
    reward_std = np.std(reward_list)
    log_msg = "%d\t%f\t%f\t%f" % (epoch, reward_mean, reward_std, time_used)
    print(log_msg)
    #print(log_msg, file=logfile, flush=True)
    logfile.write(log_msg + '\n')
    logfile.flush() 

    # Save shared model and (supposedly) shared optimizer
    # Purposely possibly over-writing other threads' model for the same epoch
    checkpoint_filename = os.path.join(args.logdir, "ckpt_" + str(epoch) + ".pth")
    with lock:
      torch.save({'epoch': epoch,
                  'state_dict': shared_model.state_dict(),
                  'optimizer' : optimizer.state_dict()},
                  checkpoint_filename)
  # End for epoch
  logfile.close()


# Update the thread-specific model once at each sentence
def decode_one_sentence_adaptive_rl(machine, seq_len, init_dec_hidden,
                                    init_dec_cell,
                                    enc_hidden_seq, initial_beam_size, max_beam_size,
                                    model, shared_model,
                                    reward_coef_fscore, reward_coef_beam_size,
                                    label_true_seq, f_score_index_begin,
                                    counter, lock, optimizer,
                                    args):
  # Currently, batch size can only be 1
  batch_size = 1

  # Each beta is (batch size, beam size) matrix,
  # and there will be T_y of them in the sequence
  # y => same
  beta_seq = []
  y_seq = []

  logP_seq = []
  accum_logP_seq = []

  if machine.attention:
    # This would be the attention alpha_{ij} coefficients
    # in the shape of (output seq len, batch size, beam size, input seq len)
    attention_seq = []
  else:
    attention_seq = None

  # For RL episode
  episode = []

  # init_label's shape => (batch size, 1),
  # with all elements machine.BEG_INDEX
  if machine.gpu:
    init_label_emb = \
      machine.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()).cuda() \
        + machine.BEG_INDEX) \
        .view(batch_size, machine.label_embedding_dim)
  else:
    init_label_emb = \
      machine.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()) \
        + machine.BEG_INDEX) \
        .view(batch_size, machine.label_embedding_dim)

  # t = 0, only one input beam from init (t = -1)
  # Only one dec_hidden_out, dec_cell_out
  # => dec_hidden_out has shape (batch size, hidden dim)
  dec_hidden_out, dec_cell_out = \
    machine.decoder_cell(init_label_emb,
                         (init_dec_hidden, init_dec_cell))

  # Attention
  if machine.attention:
    dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
    dec_hidden_out, attention = \
      machine.attention(dec_hidden_out, enc_hidden_seq, 0, machine.enc2dec_hidden)

    # remove the added dim
    dec_hidden_out = dec_hidden_out.view(batch_size, machine.hidden_dim)
    attention = attention.view(batch_size, seq_len)

  # dec_hidden_beam shape => (1, batch size, hidden dim),
  # 1 because there is only 1 input beam
  dec_hidden_beam = torch.stack([dec_hidden_out], dim=0)
  dec_cell_beam = torch.stack([dec_cell_out], dim=0)

  # This one is for backtracking (need permute)
  if machine.attention:
    # For better explanation, see in the "for t" loop below
    #
    # Originally attention has shape (batch size, input seq len)
    #
    # At t = 0, there is only 1 beam, so formally attention is actually
    # in shape (1, batch size, input seq len), where 1 is beam size.
    attention_beam = torch.stack([attention], dim=0)

    # We need to permute (swap) the dimensions into
    # the shape (batch size, 1, input seq len)
    attention_beam = attention_beam.permute(1, 0, 2)

  # score_out.shape => (batch size, |V^y|)
  score_out = machine.hidden2score(dec_hidden_out) \
    .view(batch_size, machine.label_size)
  logP_out = machine.score2logP(score_out).view(batch_size, machine.label_size)

  # Initial step, accumulated logP is the same as logP
  accum_logP_out = logP_out

  logP_out_list = [logP_out]
  accum_logP_out_list = [accum_logP_out]

  # This one is for backtracking (need permute)
  logP_output_beam = torch.stack(logP_out_list, dim=0).permute(1, 0, 2)
  accum_logP_output_beam = \
    torch.stack(accum_logP_out_list, dim=0).permute(1, 0, 2)

  # score_matrix.shape => (batch size, |V^y| * 1)
  # * 1 because there is only 1 input beam
  logP_matrix = torch.cat(logP_out_list, dim=1)
  accum_logP_matrix = torch.cat(accum_logP_out_list, dim=1)

  # Just for code consistency (about reward calculation)
  cur_beam_size_in = 1

  # Just for code consistency (about experience tuple)
  cur_state = machine.make_state(accum_logP_matrix, logP_matrix, 1,
                                 max_beam_size)
  action = None

  # All beta^{t=0, b} are actually 0
  # beta_beam.shape => (batch size, beam size),
  # each row is [y^{t, b=0}, y^{t, b=1}, ..., y^{t, b=B-1}]
  # y_beam, score_beam => same

  action_seq = []
  beam_size_seq = []
  beam_size = initial_beam_size
  beam_size_seq.append(beam_size)
  accum_logP_beam, index_beam = torch.topk(accum_logP_matrix, beam_size,
                                           dim=1)

  beta_beam = torch.floor(index_beam.float() / machine.label_size).long()
  y_beam = torch.remainder(index_beam, machine.label_size)

  # This one is for backtracking
  beta_seq.append(beta_beam)
  y_seq.append(y_beam)
  if machine.attention:
    attention_seq.append(attention_beam)
  logP_seq.append(logP_output_beam)
  accum_logP_seq.append(accum_logP_output_beam)

  # Just for sentence with length = 1
  label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq = machine.backtracking(
    1, batch_size, y_seq, beta_seq, attention_seq, logP_seq, accum_logP_seq)

  # -----------------
  # Sync params with the shared model
  with lock:
    model.load_state_dict(shared_model.state_dict())

  values = []
  log_probs = []
  rewards = []
  entropies = []
  # -----------------

  # t = 1, 2, ..., (T_y - 1 == seq_len - 1)
  for t in range(1, seq_len):
    # print("At time step {} seq_len={}".format(t, seq_len))


    # We loop through beam because we expect that
    # usually batch size > beam size
    #
    # DESIGN: This may not be true anymore in adaptive beam search,
    # since we expect batch size = 1 in this case.
    # So is beam operations vectorizable?

    accum_logP_matrix, logP_matrix, dec_hidden_beam, dec_cell_beam, attention_beam, accum_logP_output_beam, logP_output_beam = \
      machine.decode_beam_step(beam_size, y_beam, beta_beam,
                                  dec_hidden_beam, dec_cell_beam, accum_logP_beam,
                                  enc_hidden_seq, seq_len, t)

    # Actually, at t = T_y - 1 == seq_len - 1,
    # you don't have to take action (you don't have to pick a beam of predictions anymore), because at this last output step, you would pick only the highest result, and do the backtracking from it to determine the best sequence.
    # However, in the current version of this code, we temporarily keep doing one more beam picking, just to be compatible with the backtracking function and the rest of the code.
    # We delay the improvement to the future work.
    #
    # Note that this state is actually the output state at t
    state = machine.make_state(accum_logP_matrix, logP_matrix,
                               beam_size, max_beam_size)

    # For experience tuple
    prev_state = cur_state
    cur_state = state
    prev_action = action

    # For reward calculation
    prev_beam_size_in = cur_beam_size_in
    cur_beam_size_in = beam_size

    # policy network showtime
    value, logit = model(state)
    prob = F.softmax(logit, dim=-1)
    log_prob = F.log_softmax(logit, dim=-1)


    # TODO: for naive MLP policy network only
    prob = prob.view(1, -1)
    log_prob = log_prob.view(1, -1)

    entropy = -(log_prob * prob).sum(1, keepdim=True)
    if t <= seq_len - 2:
      entropies.append(entropy)

    action = prob.multinomial().data
    log_prob = log_prob.gather(1, Variable(action))

    # state, reward, done, _ = env.step(action.numpy())
    # done = done or episode_length >= args.max_episode_length
    # reward = max(min(reward, 1), -1)

    with lock:
      counter.value += 1

    # populate data
    if t <= seq_len - 2:
      values.append(value)
      log_probs.append(log_prob)
      action_seq.append(action)

    # print(type(action))
    # TODO: review this
    action = action.numpy()[0]

    # update beam size w.r.t to the action chosen
    if action == 0 and beam_size > 1:
      beam_size -= 1
    elif action == 2 and beam_size < max_beam_size:
      beam_size += 1

    if t <= seq_len - 2:
      beam_size_seq.append(beam_size)

    accum_logP_beam, index_beam = \
      torch.topk(accum_logP_matrix, beam_size, dim=1)

    beta_beam = torch.floor(
      index_beam.float() / machine.label_size).long()
    y_beam = torch.remainder(index_beam, machine.label_size)
    beta_seq.append(beta_beam)
    y_seq.append(y_beam)
    if machine.attention:
      attention_seq.append(attention_beam)
    logP_seq.append(logP_output_beam)
    accum_logP_seq.append(accum_logP_output_beam)

    # Compute the F-score for the sequence [0, 1, ..., t] (length t+1) using y_seq, betq_seq we got so far. This is the ("partial", so to speak) F-score at this t.
    label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq = \
      machine.backtracking(
      t + 1, batch_size, y_seq, beta_seq, attention_seq, logP_seq,
      accum_logP_seq)

    cur_fscore = machine.get_fscore(label_pred_seq, label_true_seq,
                                    f_score_index_begin)

    # If t >= 2, compute the reward,
    # and generate the experience tuple ( s_{t-1}, a_{t-1}, r_{t-1}, s_t )
    # reward = None
    if t >= 2:
      reward = machine.get_reward(cur_fscore, fscore, cur_beam_size_in,
                                  prev_beam_size_in, reward_coef_fscore,
                                  reward_coef_beam_size)
      experience_tuple = (prev_state, prev_action, reward, cur_state)
      episode.append(experience_tuple)
      rewards.append(reward)

    fscore = cur_fscore
  # End for t

  # print("rewards: {}".format(rewards))
  # print("actions: {}".format(action_seq))

  total_reward = sum(rewards)

  ################
  # backprop now with actor-critic
  R = torch.zeros(1, 1)
  R = Variable(R)
  policy_loss = 0
  value_loss = 0
  gae = torch.zeros(1, 1)

  # This is just a coding trick
  values.append(Variable(torch.zeros(1, 1)))

  for i in reversed(range(len(rewards))):
    R = args.gamma * R + rewards[i]
    advantage = R - values[i]
    value_loss = value_loss + 0.5 * advantage.pow(2)

    # Generalized Advantage Estimation
    # Here is where the coding trick is used
    delta_t = rewards[i] + args.gamma * \
                           values[i + 1].data - values[i].data
    gae = gae * args.gamma * args.tau + delta_t

    policy_loss = policy_loss - \
                  log_probs[i] * Variable(gae) - args.entropy_coef * \
                                                 entropies[i]
  # print(policy_loss)

  optimizer.zero_grad()

  (policy_loss + args.value_loss_coef * value_loss).backward()
  torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

  ensure_shared_grads(model, shared_model)
  # Do I need to do the update with lock?
  optimizer.step()

  return label_pred_seq, accum_logP_pred_seq, logP_pred_seq, \
         attention_pred_seq, episode, beam_size_seq, total_reward


def eval_adaptive(machine,
                  max_beam_size,
                  model,
                  data_X, data_Y, index2word, index2label,
                  suffix, write_result, decode_method, beam_size,
                  reward_coef_fscore, reward_coef_beam_size,
                  f_score_index_begin,
                  args):
  torch.manual_seed(123)
  batch_num = len(data_X)

  if write_result:
    f_sen = open(os.path.join(args.logdir,
                              "sen_" + suffix + ".txt"), 'w')
    f_pred = open(os.path.join(args.logdir,
                               "pred_" + suffix + ".txt"), 'w')
    f_label = open(os.path.join(args.logdir,
                                "label_" + suffix + ".txt"), 'w')
    f_result_processed = \
      open(os.path.join(args.logdir,
                        "result_processed_" + suffix + ".txt"), 'w')
    f_beam_size = \
      open(os.path.join(args.logdir,
                        "beam_size_" + suffix + ".txt"), 'w')
  # End if write_result

  beam_size_seqs = []

  true_pos_count = 0
  pred_pos_count = 0
  true_pred_pos_count = 0

  batch_idx_list = range(batch_num)

  time_begin = time.time()
  for batch_idx in batch_idx_list:
    sen = data_X[batch_idx]
    label = data_Y[batch_idx]

    current_batch_size = len(sen)
    current_sen_len = len(sen[0])

    sen_var = Variable(torch.LongTensor(sen))
    label_var = Variable(torch.LongTensor(label))

    if machine.gpu:
      sen_var = sen_var.cuda()
      label_var = label_var.cuda()

    # Initialize the hidden and cell states
    # The axes semantics are
    # (num_layers * num_directions, batch_size, hidden_size)
    # So 1 for single-directional LSTM encoder,
    # 2 for bi-directional LSTM encoder.
    init_enc_hidden = Variable(
      torch.zeros((2, current_batch_size, machine.hidden_dim)))
    init_enc_cell = Variable(
      torch.zeros((2, current_batch_size, machine.hidden_dim)))

    if machine.gpu:
      init_enc_hidden = init_enc_hidden.cuda()
      init_enc_cell = init_enc_cell.cuda()

    enc_hidden_seq, (enc_hidden_out, enc_cell_out) = machine.encode(sen_var,
                                                                 init_enc_hidden,
                                                                 init_enc_cell)

    # The semantics of enc_hidden_out is (num_layers * num_directions,
    # batch, hidden_size), and it is "tensor containing the hidden state
    # for t = seq_len".
    #
    # Here we use a linear layer to transform the two-directions of the dec_hidden_out's into a single hidden_dim vector, to use as the input of the decoder
    init_dec_hidden = machine.enc2dec_hidden(
      torch.cat([enc_hidden_out[0], enc_hidden_out[1]], dim=1))
    init_dec_cell = machine.enc2dec_cell(
      torch.cat([enc_cell_out[0], enc_cell_out[1]], dim=1))

    # ===================================
    if decode_method == "adaptive":
      # the input argument "beam_size" serves as initial_beam_size here
      # TODO: implement this here
      label_pred_seq, accum_logP_pred_seq, logP_pred_seq, \
      attention_pred_seq, sen_beam_size_seq = \
        decode_one_sentence_adaptive_rl_eval(
        machine, current_sen_len, init_dec_hidden, init_dec_cell,
        enc_hidden_seq, beam_size, max_beam_size,
        model,
        reward_coef_fscore, reward_coef_beam_size,
        label_var, f_score_index_begin,
        args)

    else:
      raise Exception("Not implemented!")
    # ===================================

    # update beam seq
    beam_size_seqs.append(sen_beam_size_seq)

    ### Debugging...
    # print("input sentence =", sen)
    # print("true label =", label)
    # print("predicted label =", label_pred_seq)
    # print("episode =", episode)

    for label_index in range(f_score_index_begin, machine.label_size):
      true_pos = (label_var == label_index)
      true_pos_count += true_pos.float().sum()

      pred_pos = (label_pred_seq == label_index)
      pred_pos_count += pred_pos.float().sum()

      true_pred_pos = true_pos & pred_pos
      true_pred_pos_count += true_pred_pos.float().sum()

    # Write result into file
    if write_result:
      if machine.gpu:
        label_pred_seq = label_pred_seq.cpu()

      label_pred_seq = label_pred_seq.data.numpy().tolist()

      # Here label_pred_seq.shape = (batch size, sen len)

      # sen, label, label_pred_seq are list of lists,
      # thus I would like to flatten them for iterating easier

      sen = list(itertools.chain.from_iterable(sen))
      label = list(itertools.chain.from_iterable(label))
      label_pred_seq = list(itertools.chain.from_iterable(label_pred_seq))
      assert len(sen) == len(label) and len(label) == len(label_pred_seq)
      for i in range(len(sen)):
        f_sen.write(str(sen[i]) + '\n')
        f_label.write(str(label[i]) + '\n')
        f_pred.write(str(label_pred_seq[i]) + '\n')

        # clean version (does not print <PAD>, print a newline instead of <EOS>)
        # if sen[i] != 0 and sen[i] != 2: # not <PAD> and not <EOS>
        # if sen[i] != 0: # not <PAD>

        result_sen = index2word[sen[i]]
        result_label = index2label[label[i]]
        result_pred = index2label[label_pred_seq[i]]
        f_result_processed.write(
          "%s %s %s\n" % (result_sen, result_label, result_pred))

        f_sen.flush()
        f_label.flush()
        f_pred.flush()
        f_result_processed.flush()

      if decode_method == "adaptive":
        beam_size_seq_str = ' '.join(map(str, sen_beam_size_seq))
        f_beam_size.write(beam_size_seq_str + '\n')
        f_beam_size.flush()
  # End for batch_idx
  time_end = time.time()
  time_used = time_end - time_begin

  if machine.gpu:
    true_pos_count = true_pos_count.cpu()
    pred_pos_count = pred_pos_count.cpu()
    true_pred_pos_count = true_pred_pos_count.cpu()

  true_pos_count = true_pos_count.data.numpy()[0]
  pred_pos_count = pred_pos_count.data.numpy()[0]
  true_pred_pos_count = true_pred_pos_count.data.numpy()[0]

  precision = true_pred_pos_count / pred_pos_count if pred_pos_count > 0 else 0

  recall = true_pred_pos_count / true_pos_count if true_pos_count > 0 else 0
  fscore = 2 / (1 / precision + 1 / recall) if (
    precision > 0 and recall > 0) else 0
  fscore = fscore * 100

  total_beam_number_in_dataset = sum([sum(beam_size_seq) for beam_size_seq in beam_size_seqs])
  avg_beam_sizes = [(sum(beam_size_seq) / len(beam_size_seq) if len(beam_size_seq) else 0) for beam_size_seq in beam_size_seqs]
  avg_beam_sizes = list(filter(lambda xx: xx > 0, avg_beam_sizes))
  avg_beam_size = sum(avg_beam_sizes) / len(avg_beam_sizes)

  if write_result:
    f_sen.close()
    f_pred.close()
    f_label.close()
    f_result_processed.close()
    f_beam_size.close()

  return fscore, total_beam_number_in_dataset, avg_beam_size, time_used


def decode_one_sentence_adaptive_rl_eval( \
  machine, seq_len, init_dec_hidden, init_dec_cell,
  enc_hidden_seq, initial_beam_size, max_beam_size,
  model,
  reward_coef_fscore, reward_coef_beam_size,
  label_true_seq, f_score_index_begin,
  args):
  # Currently, batch size can only be 1
  batch_size = 1

  # Each beta is (batch size, beam size) matrix,
  # and there will be T_y of them in the sequence
  # y => same
  beta_seq = []
  y_seq = []

  logP_seq = []
  accum_logP_seq = []

  if machine.attention:
    # This would be the attention alpha_{ij} coefficients
    # in the shape of (output seq len, batch size, beam size, input seq len)
    attention_seq = []
  else:
    attention_seq = None

  # For RL episode
  #episode = []

  # init_label's shape => (batch size, 1),
  # with all elements machine.BEG_INDEX
  if machine.gpu:
    init_label_emb = \
      machine.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()).cuda() \
        + machine.BEG_INDEX) \
        .view(batch_size, machine.label_embedding_dim)
  else:
    init_label_emb = \
      machine.label_embedding(
        Variable(torch.LongTensor(batch_size, 1).zero_()) \
        + machine.BEG_INDEX) \
        .view(batch_size, machine.label_embedding_dim)

  # t = 0, only one input beam from init (t = -1)
  # Only one dec_hidden_out, dec_cell_out
  # => dec_hidden_out has shape (batch size, hidden dim)
  dec_hidden_out, dec_cell_out = \
    machine.decoder_cell(init_label_emb,
                         (init_dec_hidden, init_dec_cell))

  # Attention
  if machine.attention:
    dec_hidden_out = dec_hidden_out[None, :, :]  # add 1 nominal dim
    dec_hidden_out, attention = \
      machine.attention(dec_hidden_out, enc_hidden_seq, 0, machine.enc2dec_hidden)

    # remove the added dim
    dec_hidden_out = dec_hidden_out.view(batch_size, machine.hidden_dim)
    attention = attention.view(batch_size, seq_len)

  # dec_hidden_beam shape => (1, batch size, hidden dim),
  # 1 because there is only 1 input beam
  dec_hidden_beam = torch.stack([dec_hidden_out], dim=0)
  dec_cell_beam = torch.stack([dec_cell_out], dim=0)

  # This one is for backtracking (need permute)
  if machine.attention:
    # For better explanation, see in the "for t" loop below
    #
    # Originally attention has shape (batch size, input seq len)
    #
    # At t = 0, there is only 1 beam, so formally attention is actually
    # in shape (1, batch size, input seq len), where 1 is beam size.
    attention_beam = torch.stack([attention], dim=0)

    # We need to permute (swap) the dimensions into
    # the shape (batch size, 1, input seq len)
    attention_beam = attention_beam.permute(1, 0, 2)

  # score_out.shape => (batch size, |V^y|)
  score_out = machine.hidden2score(dec_hidden_out) \
    .view(batch_size, machine.label_size)
  logP_out = machine.score2logP(score_out).view(batch_size, machine.label_size)

  # Initial step, accumulated logP is the same as logP
  accum_logP_out = logP_out

  logP_out_list = [logP_out]
  accum_logP_out_list = [accum_logP_out]

  # This one is for backtracking (need permute)
  logP_output_beam = torch.stack(logP_out_list, dim=0).permute(1, 0, 2)
  accum_logP_output_beam = \
    torch.stack(accum_logP_out_list, dim=0).permute(1, 0, 2)

  # score_matrix.shape => (batch size, |V^y| * 1)
  # * 1 because there is only 1 input beam
  logP_matrix = torch.cat(logP_out_list, dim=1)
  accum_logP_matrix = torch.cat(accum_logP_out_list, dim=1)

  # Just for code consistency (about reward calculation)
  cur_beam_size_in = 1

  # Just for code consistency (about experience tuple)
  cur_state = machine.make_state(accum_logP_matrix, logP_matrix, 1,
                                 max_beam_size)
  action = None

  # All beta^{t=0, b} are actually 0
  # beta_beam.shape => (batch size, beam size),
  # each row is [y^{t, b=0}, y^{t, b=1}, ..., y^{t, b=B-1}]
  # y_beam, score_beam => same

  action_seq = []
  beam_size_seq = []
  beam_size = initial_beam_size
  beam_size_seq.append(beam_size)
  accum_logP_beam, index_beam = torch.topk(accum_logP_matrix, beam_size,
                                           dim=1)

  beta_beam = torch.floor(index_beam.float() / machine.label_size).long()
  y_beam = torch.remainder(index_beam, machine.label_size)

  # This one is for backtracking
  beta_seq.append(beta_beam)
  y_seq.append(y_beam)
  if machine.attention:
    attention_seq.append(attention_beam)
  logP_seq.append(logP_output_beam)
  accum_logP_seq.append(accum_logP_output_beam)

  # Just for sentence with length = 1
  label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq = machine.backtracking(
    1, batch_size, y_seq, beta_seq, attention_seq, logP_seq, accum_logP_seq)

  # t = 1, 2, ..., (T_y - 1 == seq_len - 1)
  for t in range(1, seq_len):
    # print("At time step {} seq_len={}".format(t, seq_len))


    # We loop through beam because we expect that
    # usually batch size > beam size
    #
    # DESIGN: This may not be true anymore in adaptive beam search,
    # since we expect batch size = 1 in this case.
    # So is beam operations vectorizable?

    accum_logP_matrix, logP_matrix, dec_hidden_beam, dec_cell_beam, attention_beam, accum_logP_output_beam, logP_output_beam = \
      machine.decode_beam_step(beam_size, y_beam, beta_beam,
                                  dec_hidden_beam, dec_cell_beam, accum_logP_beam,
                                  enc_hidden_seq, seq_len, t)

    # Actually, at t = T_y - 1 == seq_len - 1,
    # you don't have to take action (you don't have to pick a beam of predictions anymore), because at this last output step, you would pick only the highest result, and do the backtracking from it to determine the best sequence.
    # However, in the current version of this code, we temporarily keep doing one more beam picking, just to be compatible with the backtracking function and the rest of the code.
    # We delay the improvement to the future work.
    #
    # Note that this state is actually the output state at t
    state = machine.make_state(accum_logP_matrix, logP_matrix,
                               beam_size, max_beam_size)

    # For experience tuple
    prev_state = cur_state
    cur_state = state
    prev_action = action

    # For reward calculation
    prev_beam_size_in = cur_beam_size_in
    cur_beam_size_in = beam_size

    # policy network showtime
    _, logit = model(state)
    prob = F.softmax(logit, dim=-1)

    # TODO: for naive MLP policy network only
    prob = prob.view(1, -1)

    action = prob.multinomial().data

    # state, reward, done, _ = env.step(action.numpy())
    # done = done or episode_length >= args.max_episode_length
    # reward = max(min(reward, 1), -1)

    # populate data
    #if t <= seq_len - 2:
    #  action_seq.append(action)

    # print(type(action))
    # TODO: review this
    action = action.numpy()[0]
    if t <= seq_len - 2:
      action_seq.append(action)

    # update beam size w.r.t to the action chosen
    if action == 0 and beam_size > 1:
      beam_size -= 1
    elif action == 2 and beam_size < max_beam_size:
      beam_size += 1

    if t <= seq_len - 2:
      beam_size_seq.append(beam_size)

    accum_logP_beam, index_beam = \
      torch.topk(accum_logP_matrix, beam_size, dim=1)

    beta_beam = torch.floor(
      index_beam.float() / machine.label_size).long()
    y_beam = torch.remainder(index_beam, machine.label_size)
    beta_seq.append(beta_beam)
    y_seq.append(y_beam)
    if machine.attention:
      attention_seq.append(attention_beam)
    logP_seq.append(logP_output_beam)
    accum_logP_seq.append(accum_logP_output_beam)

    if t == seq_len - 1:
      label_pred_seq, accum_logP_pred_seq, logP_pred_seq, attention_pred_seq = \
        machine.backtracking(
        t + 1, batch_size, y_seq, beta_seq, attention_seq, logP_seq,
        accum_logP_seq)
  # End for t

  return label_pred_seq, accum_logP_pred_seq, logP_pred_seq, \
         attention_pred_seq, beam_size_seq

