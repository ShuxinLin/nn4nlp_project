import numpy as np
import torch
from torch.autograd import Variable

x = np.random.randint(0, 10, size=(3, 5)).astype(np.float)
x = Variable(torch.LongTensor(x))
print(x)


print(torch.max(x).data[0])