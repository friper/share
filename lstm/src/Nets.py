'''
author: firper
'''
from src.Layers import *

class LSTM():
  def __init__(self,inodes,hnodes,onodes,lr=0.0001,layers=1,purpose=1):
    self.purpose = purpose;
    if 1==layers:
      self.Layers = [
        Embedding(inodes,hnodes,lr),
        LstmPeep(hnodes,hnodes,lr),
        Affine(hnodes,onodes,lr,'sigmoid'),
        Softmax()
       ];
    else:
      self.Layers = layers;
    self.lyLen = len(self.Layers)-1;

  def feedforward(self,res):
    for layer in self.Layers:
      res = layer.feedforward(res);

    return res;

  def backprop(self,res,teach,state):
    for layer in self.Layers:
      res = layer.backforward(res,state=state);
    loop = self.Layers[::-1];
    for layer in loop:
      teach = layer.backprop(teach);

  def predict(self,inputData):
    x = np.array(inputData);
    res = self.feedforward(x);
    return res;

  def evaluate(self):
    res = self.Layers[self.lyLen].get();
    return res;

  def optimain(self):
    for layer in self.Layers[:-1]:
      layer.optimain();

  def train(self,datas,epoch=1):
    inputList = np.array(datas[0]);
    teachList = np.array(datas[1]);
    maxLen = len(teachList);
    state = True;
    self.epoch = 0;
    batch = (maxLen*epoch)+1;
    print('batch:',batch,'epoch:',epoch);

    for b in range(batch):
      rnd = b%maxLen;

      if 0==rnd and 0!=b:
        self.epoch += 1;
        loss = self.evaluate();
        print('epoch:',self.epoch,'Loss: {:.8f}'.format(loss));
        if self.purpose>loss:
          print('train loss reach the purpose value');
          break;
        state = True;
        self.optimain();

      self.backprop(inputList[rnd],teachList[rnd],state);
      state = False;
#=====================================================================
class RNN():
  def __init__(self,inodes,hnodes,onodes,lr=0.001,layers=1,purpose=1):
    self.purpose = purpose;
    if 1==layers:
      self.Layers = [
        Embedding(inodes,hnodes,lr),
        Rnn(hnodes,hnodes,lr),
        Affine(hnodes,onodes,lr,'sigmoid'),
        Softmax()
       ];
    else:
      self.Layers = layers;
    self.lyLen = len(self.Layers)-1;

  def feedforward(self,res):
    for layer in self.Layers:
      res = layer.feedforward(res);

    return res;

  def backprop(self,res,teach,state):
    for layer in self.Layers:
      res = layer.backforward(res,state=state);
    loop = self.Layers[::-1];
    for layer in loop:
      teach = layer.backprop(teach);

  def predict(self,inputData):
    x = np.array(inputData);
    res = self.feedforward(x);
    return res;

  def evaluate(self):
    res = self.Layers[self.lyLen].get();
    return res;

  def optimain(self):
    for layer in self.Layers[:-1]:
      layer.optimain();

  def train(self,datas,epoch=1):
    inputList = np.array(datas[0]);
    teachList = np.array(datas[1]);
    maxLen = len(teachList);
    state = True;
    self.epoch = 0;
    batch = (maxLen*epoch)+1;
    print('batch:',batch,'epoch:',epoch);

    for b in range(batch):
      rnd = b%maxLen;

      if 0==rnd and 0!=b:
        self.epoch += 1;
        loss = self.evaluate();
        print('epoch:',self.epoch,'Loss: {:.8f}'.format(loss));
        if self.purpose>loss:
          print('train loss reach the purpose value');
          break;
        state = True;
        self.optimain();

      self.backprop(inputList[rnd],teachList[rnd],state);
      state = False;
