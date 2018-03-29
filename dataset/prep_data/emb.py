import sys
import numpy as np
dic={}
vectors=[]
for line in open('embeddings'):
  ls = line.strip().split()
  dic[ls[0]]= ls[1:]
  vectors.append(ls[1:])
#for line in open('vocab1.de'):
#  wrd = line.strip().split()[0]
#  if wrd in dic:
#    vectors.append(dic[wrd])
#  else:
#    vectors.append([0.0]*512)
a= np.asarray(vectors)
np.save('emb.npy',a)
