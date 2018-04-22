import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import re


logfile = open("../gitignore/result_lrn_0p001_atten/eval_greedy.txt", "r")

train_f = []
val_f = []
test_f = []
greedy_test_f = []

for line in logfile:
  tmp = re.split("\t", line)
  train_f.append(float(tmp[4]))
  val_f.append(float(tmp[5]))
  test_f.append(float(tmp[6]))
  greedy_test_f.append(float(tmp[6]))

epoch = list(range(len(train_f)))

epochToPlot = len(train_f)

plt.figure(1)
plt.plot(epoch[:epochToPlot], train_f[:epochToPlot], "r-", epoch[:epochToPlot], val_f[:epochToPlot], "g-", epoch[:epochToPlot], test_f[:epochToPlot], "b-")
#plt.xlim([0, 11])
#plt.ylim([0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('F-score')
plt.savefig('../gitignore/result_lrn_0p001_atten/fig_lrn_0p001_greedy_f.pdf')


logfile.close()





logfile = open("../gitignore/result_lrn_0p001_atten/eval_beam_1.txt", "r")

train_f = []
val_f = []
test_f = []
beam_1_test_f = []

for line in logfile:
  tmp = re.split("\t", line)
  train_f.append(float(tmp[4]))
  val_f.append(float(tmp[5]))
  test_f.append(float(tmp[6]))
  beam_1_test_f.append(float(tmp[6]))

epoch = list(range(len(train_f)))

epochToPlot = len(train_f)

plt.figure(2)
plt.plot(epoch[:epochToPlot], train_f[:epochToPlot], "r-", epoch[:epochToPlot], val_f[:epochToPlot], "g-", epoch[:epochToPlot], test_f[:epochToPlot], "b-")
#plt.xlim([0, 11])
#plt.ylim([0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('F-score')
plt.savefig('../gitignore/result_lrn_0p001_atten/fig_lrn_0p001_beam_1_f.pdf')


logfile.close()







logfile = open("../gitignore/result_lrn_0p001_atten/eval_beam_3.txt", "r")

train_f = []
val_f = []
test_f = []
beam_3_test_f = []

for line in logfile:
  tmp = re.split("\t", line)
  train_f.append(float(tmp[4]))
  val_f.append(float(tmp[5]))
  test_f.append(float(tmp[6]))
  beam_3_test_f.append(float(tmp[6]))

epoch = list(range(len(train_f)))

epochToPlot = len(train_f)

plt.figure(3)
plt.plot(epoch[:epochToPlot], train_f[:epochToPlot], "r-", epoch[:epochToPlot], val_f[:epochToPlot], "g-", epoch[:epochToPlot], test_f[:epochToPlot], "b-")
#plt.xlim([0, 11])
#plt.ylim([0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('F-score')
plt.savefig('../gitignore/result_lrn_0p001_atten/fig_lrn_0p001_beam_3_f.pdf')


logfile.close()





plt.figure(4)
#plt.plot(epoch[:epochToPlot], greedy_test_f[:epochToPlot], "r^", epoch[:epochToPlot], beam_1_test_f[:epochToPlot], "g-", epoch[:epochToPlot], beam_3_test_f[:epochToPlot], "bo")
plt.plot(epoch[:epochToPlot], greedy_test_f[:epochToPlot], "r-", epoch[:epochToPlot], beam_3_test_f[:epochToPlot], "b-")
#plt.xlim([0, 11])
#plt.ylim([0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('F-score')
plt.savefig('../gitignore/result_lrn_0p001_atten/fig_lrn_0p001_combine_f.pdf')



diff = [(beam_3_test_f[i] - greedy_test_f[i]) for i in range(len(greedy_test_f))]

plt.figure(5)
plt.plot(epoch[:epochToPlot], diff[:epochToPlot], "b-")
plt.plot(epoch[:epochToPlot], np.zeros(epochToPlot), "k--")
#plt.xlim([0, 11])
plt.ylim([-2.5, 2.5])
plt.xlabel('Epoch')
plt.ylabel('F-score diff (beam_3 - greedy)')
plt.savefig('../gitignore/result_lrn_0p001_atten/fig_lrn_0p001_diff_f.pdf')








"""
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
"""

