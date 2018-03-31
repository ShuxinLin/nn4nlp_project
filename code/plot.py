import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import re


logfile = open("../result_lrn_0p001/log.txt", "r")

epoch = list(range(len(train_loss)))

train_loss = []
val_loss = []

for line in logfile:
  tmp = re.split("\t", line)
  train_loss.append(float(tmp[2]))
  val_loss.append(float(tmp[3]))


plt.figure(1)
plt.plot(epoch, train_loss, "r-", epoch, val_loss, "b-")
#plt.xlim([0, 11])
#plt.ylim([0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('Average cross-entropy error')
plt.savefig('fig_lrn_0p001.pdf')





