'''
author: friper
'''
import numpy as np

def sin():
  datas = list(np.sin(np.arange(-2*np.pi,8*np.pi,0.1)+2*np.pi)+1);

  step = 200;
  maxV = 2;
  inputList = [];
  teachList = [];
  maxLen = len(datas);
  maxLen = 300;
  print('len:max ',maxLen,maxV);
  for i in range(step,maxLen):
    inputData = [];
    teachData = [];
    for k in range(i-step-1,i-1):
      inputData.append((datas[k]/maxV));
    teachData = [(datas[i]/maxV)];

    inputList.append(inputData);
    teachList.append(teachData);
  all = [inputList,teachList];

  return all,maxV,maxLen,step;

def classes():
  inputList = [];
  teachList = [];

  print('inodes:onodes','8,2');
  a = [0.01,0.01,0.99,0.99,0.01,0.99,0.99,0.99];
  a_r = [0.99,0.01];
  b = [0.99,0.99,0.01,0.01,0.99,0.99,0.01,0.99];
  b_r = [0.01,0.99];

  for i in range(1000):
    if 0==i%50:
      inputList.append(a);
      teachList.append(a_r);
    elif 0==i%60:
      inputList.append(b);
      teachList.append(b_r);
    else:
      inputList.append(list(np.random.rand(8)));
      teachList.append([0.01,0.01]);

  all = [inputList,teachList];

  return all,a,b;
