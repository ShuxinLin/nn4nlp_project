import numpy as np

logfile = open("../result_toy/eval_beam_3.txt", "r")

data = []
for line in logfile:
  data.append(list(map(float, line.strip().split("\t"))))
data = np.array(data)

logfile.close()

val_f_vec = data[:, 1]
prev = -1
wait_threshold = 3
for epoch, f in enumerate(val_f_vec):
  if f < prev:
    decrease_counter += 1
    if decrease_counter >= wait_threshold:
      break
  else:
    decrease_counter = 0
  prev = f

print("best epoch =", epoch - wait_threshold)
