import matplotlib.pyplot as plt
import numpy as np
from src import makeData as md
from src import Nets
from src.Layers import *
import time

all,maxV,maxLen,step = md.sin();
lr = 0.001;
## RNN Model
#Layers = [Rnn(step,100,lr),Affine(100,1,lr,'linear'),Loss()];
## Peephole LSTM Model
#Layers = [PeepLstm(step,100,lr),Affine(100,1,lr'linear'),Loss()];
## LSTM Model
Layers = [Lstm(step,100,lr),Affine(100,1,lr,'linear'),Loss()];
net = Nets.LSTM(1,1,1,layers=Layers,purpose=0.00001);

loop = len(all[1]);
res_no = np.array([float(net.predict(all[0][t])) for t in range(loop)]);

st = time.time();
epoc = 600;
net.train(all,epoch=epoc);
print('time:',time.time()-st);

st = time.time();
row = np.array([all[1][t][0] for t in range(loop)]);
res = np.array([float(net.predict(all[0][t])) for t in range(loop)]);

#plt.figure(figsize=(20,15));
plt.plot(row,label='row');
plt.plot(res,label='LSTM');
plt.plot(res_no,label='no train');
plt.legend();
plt.savefig('predict.jpg');
print('figure time:',time.time()-st);
