FULL BPTT LSTM
===

Overview

## Requirement

- Python 3.6.9
- numpy 1.18.3
- matplotlib 2.1.1

## Install
```
git clone https://github.com/friper/share.git
mv ./share/lstm <your path>/LSTM
rm -r share
```

## Run
```
python3 test.py
```

## Information

### RNN

**RNN Model**
![RNN Model](./images/RNN.jpg)

### LSTM

**LSTM Model**
![LSTM Model](./images/LSTM.jpg)

**Peephole LSTM Model**
![Peephole LSTM Model](./images/PeepholeLSTM.jpg) 

**LSTM Calculate**
![LSTM Calculate](./images/PeepholeLSTMcal.jpg)

### Format

* train/test Data: [[[trainData1],[trainData2],...],[[teachData1],[teachData2],...]]

* predict Data: [predict Data]
