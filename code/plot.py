import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import re


logfile = open("../gitignore/result_lrn_0p001_beam_10/log.txt", "r")

train_loss = []
val_loss = []
test_loss = []

train_f = []
val_f = []
test_f = []

for line in logfile:
  tmp = re.split("\t", line)
  train_loss.append(float(tmp[2]))
  val_loss.append(float(tmp[3]))
  test_loss.append(float(tmp[4]))
  train_f.append(float(tmp[5]))
  val_f.append(float(tmp[6]))
  test_f.append(float(tmp[7]))

epoch = list(range(len(train_loss)))

epochToPlot = 42

plt.figure(1)
plt.plot(epoch[:epochToPlot], train_loss[:epochToPlot], "r-", epoch[:epochToPlot], val_loss[:epochToPlot], "g-", epoch[:epochToPlot], test_loss[:epochToPlot], "b-")
#plt.xlim([0, 11])
#plt.ylim([0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('Average cross-entropy error')
plt.savefig('../gitignore/result_lrn_0p001_beam_10/fig_lrn_0p001_beam_10_loss.pdf')



plt.figure(2)
plt.plot(epoch[:epochToPlot], train_f[:epochToPlot], "r-", epoch[:epochToPlot], val_f[:epochToPlot], "g-", epoch[:epochToPlot], test_f[:epochToPlot], "b-")
#plt.xlim([0, 11])
#plt.ylim([0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('F-score')
plt.savefig('../gitignore/result_lrn_0p001_beam_10/fig_lrn_0p001_beam_10_f.pdf')













logfile.close()


