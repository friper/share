'''
author: firper
'''
import numpy as np

def sigmoid(x):
  return np.nan_to_num(1/(1+np.exp(-x)));

# x should be x through sigmoid function.
def derivSig(x):
  return x*(1.0-x);

def tanh(x):
  return np.tanh(x);
# x should be x through tanh function.
def derivTanh(x):
  return 1-x**2;

def linear(x):
  return x;

def derivLine(x):
  return 1.0;

def relu(x):
  return x*(x>0.0);

def derivRelu(x):
  return 1.0*(x>0.0);

def activate(active):
  if 'sigmoid'==active:
    return sigmoid,derivSig;
  elif 'tanh'==active:
    return np.tanh,derivTanh;
  elif 'linear'==active:
    return linear,derivLine;
  elif 'relu'==active:
    return relu,derivRelu;

def initWeight(inodes,onodes):
  return np.random.normal(0.0,1.0,(onodes,inodes))/100;

def initPeep(inodes,onodes):
  return np.random.normal(0.0,1.0,(onodes,inodes))/np.sqrt(onodes);

def zeroL(weight):
  return np.zeros_like(weight);

class Embedding():
  def __init__(self,inodes,onodes,lr):
    self.lr = lr;
    self.weights = initWeight(onodes,inodes);
    #self.errors = np.zeros_like(self.weights);
    self.results = None;
    self.id = None;
    self.Grands = zeroL(self.weights);

  def feedforward(self,Wid):
    return np.sum(self.weights[Wid],axis=0);
    #return self.weights[Wid];

  def backforward(self,Wid,state=False):
    self.id = Wid;
    self.results = np.sum(self.weights[Wid],axis=0);
    #self.results = self.weights[Wid];

    return self.results;

  def backprop(self,Error):
    #self.weights[self.id] += self.lr*(Error+self.weights[self.id]-self.results);
    #self.weights[self.id] += self.lr*(Error*self.weights[self.id]);
    self.Grands[self.id] += Error*self.weights[self.id];
    #self.Grands[self.id] += np.dot(Error,self.weights[self.id].T);

  def optimain(self):
    self.weights += self.Grands;
    self.Grands = zeroL(self.weights);

class Affine():
  def __init__(self,inodes,onodes,lr,active):
    self.lr = lr;
    self.weights = initWeight(onodes,inodes);
    self.Contents = np.array([[],[]]);
    self.active,self.derivAct = activate(active);
    self.Grands = zeroL(self.weights);

  def feedforward(self,inputMat):
     return self.active(np.dot(inputMat,self.weights));

  def backforward(self,inputMat,state=False):
    self.Contents = [np.array([inputMat]),self.active(np.dot(inputMat,self.weights))];
    return self.Contents[1];

  def backprop(self,Error):
    deriv = Error*self.derivAct(self.Contents[1]);
    self.Grands += deriv*self.Contents[0].T;
    return np.dot(deriv,self.weights.T);

    #self.Grands += np.dot(np.array([Error*self.derivAct(self.Contents[1])]).T,self.Contents[0]);
    #return np.dot(self.weights.T,Error);

  def optimain(self):
    self.weights += self.lr*self.Grands;
    self.Grands = zeroL(self.weights);

class Loss():
  def __init__(self):
    self.all = 0;
    self.results = None;

  def feedforward(self,inputMat):
    return inputMat;

  def backforward(self,inputMat,state=False):
    self.results = inputMat;
    return inputMat;

  def backprop(self,teachMat):
    error = teachMat-self.results;
    self.all += error;
    return error;

  def get(self):
    loss = (np.sum(self.all)**2)/2;
    self.all = 0;
    return loss;

class Softmax():
  def __init__(self):
    self.all = 0;
    self.inputs = None;
    self.results = None;

  def feedforward(self,inputMat):
    return np.exp(inputMat)/np.sum(np.exp(inputMat));

  def backforward(self,inputMat,state=False):
    self.inputs = inputMat;
    self.results = np.exp(inputMat)/np.sum(np.exp(inputMat));
    return self.results;

  def backprop(self,teachMat):
    error = -np.sum((teachMat*np.log(self.results+1e-7)));
    self.all += error;
    res = (teachMat-self.results)

    return res;

  def get(self):
    loss = np.sum(self.all);
    self.all = 0;
    return loss;

class Rnn():
  def __init__(self,inodes,onodes,lr):
    self.lr = lr;
    self.Wx = initPeep(onodes,inodes);
    self.Wh = initPeep(onodes,onodes);
    self.Grands = [zeroL(self.Wh),zeroL(self.Wx)];

    self.Contents = [1,1,1,1];
    self.Prevs = np.ones(onodes);
    self.inodes = onodes;

  def feedforward(self,inputMat):
    res = np.tanh(np.dot(inputMat,self.Wx)+np.dot(self.Prevs,self.Wh));

    self.Prevs = res;
    return res;

  def backforward(self,inputMat,state=False):
    if state:
      self.Prevs = np.ones(self.inodes);
 
    res = np.tanh(np.dot(inputMat,self.Wx)+np.dot(self.Prevs,self.Wh));
    
    self.Contents[:3] = [inputMat,self.Prevs,res];
    self.Prevs = res;
    return res;

  def backprop(self,Error):
    x,prevs,res,nEH = self.Contents;

    deriv = (Error+nEH)*derivTanh(res);
    errorX = np.dot(deriv,self.Wx.T);
    self.Contents[3] = np.dot(deriv,self.Wh.T);

    self.Grands[0] += deriv*np.array([prevs]).T;
    self.Grands[1] += deriv*np.array([x]).T;

    return errorX;

  def optimain(self):
    self.Wh += self.lr*self.Grands[0];
    self.Wx += self.lr*self.Grands[1];
    self.Grands = [zeroL(self.Wh),zeroL(self.Wx)];

class LstmPeep():
  def __init__(self,inodes,onodes,lr):
    self.lr = lr;
    self.Wx = initPeep(4*onodes,inodes);
    self.Rh = initPeep(4*onodes,onodes);
    self.Pc = initPeep(onodes,3);

    self.Grands = [zeroL(self.Wx),zeroL(self.Rh),zeroL(self.Pc)];
    self.Contents = [np.ones(onodes) for i in range(15)];
    self.PrevH = np.ones(onodes);
    self.PrevC = np.ones(onodes);
    self.inodes = onodes;

  def feedforward(self,inputMat):
    IND = self.inodes;
    A = np.dot(inputMat,self.Wx)+np.dot(self.PrevH,self.Rh);
    B = self.PrevC*self.Pc[:2];

    z = np.tanh(A[:IND]);
    i = sigmoid(A[IND:2*IND]+B[0]);
    f = sigmoid(A[2*IND:3*IND]+B[1]);
    c = z*i+self.PrevC*f;
    cb = np.tanh(c);
    o = sigmoid(A[3*IND:]+c*self.Pc[2]);
    h = o*cb;

    self.PrevH,self.PrevC = h,c;
    return h;

  def backforward(self,inputMat,state=False):
    IND = self.inodes;
    if state:
      self.PrevH = np.ones(IND);
      self.PrevC = np.ones(IND);

    A = np.dot(inputMat,self.Wx)+np.dot(self.PrevH,self.Rh);
    B = self.PrevC*self.Pc[:2];

    z = np.tanh(A[:IND]);
    i = sigmoid(A[IND:2*IND]+B[0]);
    f = sigmoid(A[2*IND:3*IND]+B[1]);
    c = z*i+self.PrevC*f;
    cb = np.tanh(c);
    o = sigmoid(A[3*IND:]+c*self.Pc[2]);
    h = o*cb;

    self.Contents[:9] = [inputMat,z,i,f,c,cb,o,self.PrevC,self.PrevH];
    self.PrevH,self.PrevC = h,c;
    return h;

  def backprop(self,Error):
    IND = self.inodes;
    x,z,i,f,c,cb,o,prevC,prevH = self.Contents[:9];
    nF,nEZ,nEI,nEF,nEC,nEO = self.Contents[9:];

    errorH = Error+np.dot(nEZ,self.Rh[:,:IND].T)+np.dot(nEI,self.Rh[:,IND:2*IND].T) \
            +np.dot(nEF,self.Rh[:,2*IND:3*IND].T)+np.dot(nEO,self.Rh[:,3*IND:4*IND].T);
    errorO = errorH*np.tanh(c)*derivSig(o);
    errorC = errorH*o*derivTanh(cb)+errorO*self.Pc[2] \
            +nEI*self.Pc[0]+nEF*self.Pc[1]+nEC*nF;
    errorZ = errorC*i*derivTanh(z);
    errorI = errorC*z*derivSig(i);
    errorF = errorC*prevC*derivSig(f);
    errorX = np.dot(errorZ,self.Wx[:,:IND].T)+np.dot(errorI,self.Wx[:,IND:2*IND].T) \
            +np.dot(errorF,self.Wx[:,2*IND:3*IND].T)+np.dot(errorO,self.Wx[:,3*IND:4*IND].T);

    self.Contents[9:] = [f,errorZ,errorI,errorF,errorC,errorO];

    #x = np.array([np.concatenate([x,x,x,x])]);
    #prevH = np.array([np.concatenate([prevH,prevH,prevH,prevH])]);
    x = np.array([x]).T;
    prevH = np.array([prevH]).T;
    #prevC = np.array([prevC]).T;
    #c = np.array([c]).T;
    combo = np.concatenate([errorZ,errorI,errorF,errorO]);
    gx = combo*x;
    gh = combo*prevH;
    if 0==np.sum(np.isnan(gx)):
      self.Grands[0] += combo*x;
    if 0==np.sum(np.isnan(gh)):
      self.Grands[1] += combo*prevH;
    #self.Grands[2] += np.concatenate([np.dot(errorI,prevC.T),np.dot(errorF,prevC.T),np.dot(errorO,c.T)]);
    '''
    self.Grands[0][:,:IND] += errorZ*x;
    self.Grands[0][:,IND:2*IND] += errorI*x;
    self.Grands[0][:,2*IND:3*IND] += errorF*x;
    self.Grands[0][:,3*IND:4*IND] += errorO*x;
    

    self.Grands[1][:,:IND] += errorZ*prevH;
    self.Grands[1][:,IND:2*IND] += errorI*prevH;
    self.Grands[1][:,2*IND:3*IND] += errorF*prevH;
    self.Grands[1][:,3*IND:4*IND] += errorO*prevH;
    '''

    self.Grands[2][0] += errorI*prevC;
    self.Grands[2][1] += errorF*prevC;
    self.Grands[2][2] += errorO*c;

    return errorX;

  def optimain(self):
    self.Wx += self.lr*self.Grands[0];
    self.Rh += self.lr*self.Grands[1];
    self.Pc += self.lr*self.Grands[2];

    self.Grands = [zeroL(self.Wx),zeroL(self.Rh),zeroL(self.Pc)];

class Lstm():
  def __init__(self,inodes,onodes,lr):
    self.lr = lr;
    self.Wx = initPeep(4*onodes,inodes);
    self.Rh = initPeep(4*onodes,onodes);

    self.Grands = [zeroL(self.Wx),zeroL(self.Rh)];
    self.Contents = [np.ones(onodes) for i in range(15)];
    self.PrevH = np.ones(onodes);
    self.PrevC = np.ones(onodes);
    self.inodes = onodes;

  def feedforward(self,inputMat):
    IND = self.inodes;
    A = np.dot(inputMat,self.Wx)+np.dot(self.PrevH,self.Rh);

    z = np.tanh(A[:IND]);
    i = sigmoid(A[IND:2*IND]);
    f = sigmoid(A[2*IND:3*IND]);
    c = z*i+self.PrevC*f;
    cb = np.tanh(c);
    o = sigmoid(A[3*IND:]);
    h = o*cb;

    self.PrevH,self.PrevC = h,c;
    return h;

  def backforward(self,inputMat,state=False):
    IND = self.inodes;
    if state:
      self.PrevH = np.ones(IND);
      self.PrevC = np.ones(IND);

    A = np.dot(inputMat,self.Wx)+np.dot(self.PrevH,self.Rh);

    z = np.tanh(A[:IND]);
    i = sigmoid(A[IND:2*IND]);
    f = sigmoid(A[2*IND:3*IND]);
    c = z*i+self.PrevC*f;
    cb = np.tanh(c);
    o = sigmoid(A[3*IND:]);
    h = o*cb;

    self.Contents[:9] = [inputMat,z,i,f,c,cb,o,self.PrevC,self.PrevH];
    self.PrevH,self.PrevC = h,c;
    return h;

  def backprop(self,Error):
    IND = self.inodes;
    x,z,i,f,c,cb,o,prevC,prevH = self.Contents[:9];
    nF,nEZ,nEI,nEF,nEC,nEO = self.Contents[9:];

    errorH = Error+np.dot(nEZ,self.Rh[:,:IND].T)+np.dot(nEI,self.Rh[:,IND:2*IND].T) \
            +np.dot(nEF,self.Rh[:,2*IND:3*IND].T)+np.dot(nEO,self.Rh[:,3*IND:4*IND].T);
    errorO = errorH*np.tanh(c)*derivSig(o);
    errorC = errorH*o*derivTanh(cb)+nEC*nF;
    errorZ = errorC*i*derivTanh(z);
    errorI = errorC*z*derivSig(i);
    errorF = errorC*prevC*derivSig(f);
    errorX = np.dot(errorZ,self.Wx[:,:IND].T)+np.dot(errorI,self.Wx[:,IND:2*IND].T) \
            +np.dot(errorF,self.Wx[:,2*IND:3*IND].T)+np.dot(errorO,self.Wx[:,3*IND:4*IND].T);

    self.Contents[9:] = [f,errorZ,errorI,errorF,errorC,errorO];

    #x = np.array([np.concatenate([x,x,x,x])]);
    #prevH = np.array([np.concatenate([prevH,prevH,prevH,prevH])]);
    x = np.array([x]).T;
    prevH = np.array([prevH]).T;
    #prevC = np.array([prevC]).T;
    #c = np.array([c]).T;
    combo = np.concatenate([errorZ,errorI,errorF,errorO]);
    gx = combo*x;
    gh = combo*prevH;
    if 0==np.sum(np.isnan(gx)):
      self.Grands[0] += combo*x;
    if 0==np.sum(np.isnan(gh)):
      self.Grands[1] += combo*prevH;
    #self.Grands[2] += np.concatenate([np.dot(errorI,prevC.T),np.dot(errorF,prevC.T),np.dot(errorO,c.T)]);

    return errorX;

  def optimain(self):
    self.Wx += self.lr*self.Grands[0];
    self.Rh += self.lr*self.Grands[1];

    self.Grands = [zeroL(self.Wx),zeroL(self.Rh)];

