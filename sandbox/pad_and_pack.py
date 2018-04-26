# https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
import itertools

def flatten(l):
	return list(itertools.chain.from_iterable(l))

seqs = ['ghatmasala','nicela','c-pakodas']

# make <pad> idx 0
# ['<pad>', '-', 'a', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o',
# 'p', 's', 't']
vocab = ['<pad>'] + sorted(list(set(flatten(seqs))))

# make model
embed = nn.Embedding(len(vocab), 10)
lstm = nn.LSTM(10, 5)

vectorized_seqs = [[vocab.index(tok) for tok in seq]for seq in seqs]
print(vectorized_seqs)

# get the length of each seq in your batch
seq_lengths = torch.LongTensor(map(len, vectorized_seqs))
print("Seq len = {}".format(seq_lengths))

# dump padding everywhere, and place seqs on the left.
# NOTE: you only need a tensor as big as your longest sequence
seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
	seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

# SORT YOUR TENSORS BY LENGTH!
seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
seq_tensor = seq_tensor[perm_idx]

print(seq_tensor)  # 3 x 10 = B x L


# utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
# Otherwise, give (L,B,D) tensors
seq_tensor = seq_tensor.transpose(0, 1) # (B,L,D) -> (L,B,D)

print(seq_tensor)  # 10 x 3 = L x B

# embed your sequences
seq_tensor = embed(seq_tensor)

print(seq_tensor)  # 10 x 3 x 10  = L x B x Emb

# pack them up nicely
packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())

print(packed_input)  # 25x10, batch_sizes=[3, 3, 3, 3, 3, 3, 2, 2, 2, 1]

# throw them through your LSTM (remember to give batch_first=True here if you packed with it)
packed_output, (ht, ct) = lstm(packed_input)
print(packed_output) # 25 x 5, batch_sizes=[3, 3, 3, 3, 3, 3, 2, 2, 2, 1]
print(ht) # 1 x 3 x 5 = n_layer x B x out_feats
print(ct) # 1 x 3 x 5

# unpack your output if required
output, _ = pad_packed_sequence(packed_output)
# print output  # 10 x 3 x 5

# Or if you just want the final hidden state?
print(ht[-1])