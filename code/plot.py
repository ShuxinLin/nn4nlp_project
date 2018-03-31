import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import re


logfile = open("../gitignore/result_lrn_0p001_beam_10/log.txt", "r")

train_loss = []
val_loss = []

train_f = []
val_f = []

for line in logfile:
  tmp = re.split("\t", line)
  train_loss.append(float(tmp[2]))
  val_loss.append(float(tmp[3]))
  train_f.append(float(tmp[4]))
  val_f.append(float(tmp[5]))

epoch = list(range(len(train_loss)))

plt.figure(1)
plt.plot(epoch, train_loss, "r-", epoch, val_loss, "b-")
#plt.xlim([0, 11])
#plt.ylim([0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('Average cross-entropy error')
plt.savefig('../gitignore/result_lrn_0p001_beam_10/fig_lrn_0p001_beam_10_loss.pdf')



plt.figure(2)
plt.plot(epoch, train_f, "r-", epoch, val_f, "b-")
#plt.xlim([0, 11])
#plt.ylim([0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('F-score')
plt.savefig('../gitignore/result_lrn_0p001_beam_10/fig_lrn_0p001_beam_10_f.pdf')













logfile.close()


