import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
# import keras
#from two_layer_net import TwoLayerNet

#0と1のデータ0から1000個を取得する。
#検証用データ1000から1100

#normalize 正規化
#one_hot_label t_trainに関して「”true”の時one-hot表現 ”false”の時数値で扱う」
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=False) 
t_train_0=t_train[np.where(t_train==0)] #0のデータx
x_train_0 = x_train[np.where(t_train==0)] #0のデータt
print("x_train_0",x_train_0[0])
# with open('train_out.txt', mode='w') as f:
#     f.write(str(x_train_0[0]))
t_train_1 = t_train[np.where(t_train==1)] #1のデータx
x_train_1 = x_train[np.where(t_train==1)] #1のデータt
print("x_train_1",x_train_1)
t_train_0to1 = np.concatenate([t_train_0, t_train_1])
x_train_0to1 = np.concatenate([x_train_0,x_train_1])

print(t_train_0to1)
train_loss_list = []

iters_num = 10 #何回勾配法使うか（イテレータ）
train_size = x_train_0to1.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784,hidden_size=2,output_size=784)

for i in range(iters_num): #1バッチに対して勾配法を用いる回数
    print("here")
    batch_mask = np.random.choice(train_size,batch_size)#(sizeの中からランダムな値,個数)[a,b,c,d,e,f.....]
    x_batch = x_train[batch_mask]
    t_batch = x_batch
    
    grad = network.numerical_gradient(x_batch,t_batch)
    
    for key in ('W1','b1','W2','b2'):
        network.params[key] = network.params[key] - learning_rate * grad[key]
        
        
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

print(train_loss_list)

