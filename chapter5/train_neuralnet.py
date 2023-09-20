import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num=10000 #iters_numとは？？？1バッチに対する試行回数　バッチ x iters_num　で学習していく。10000回繰り返す
train_size = x_train.shape[0] #何個のデータがあるのか x_train.shape[1]と[2]はそれぞれ28（28*28の画像）
batch_size = 100 #まとめて処理するデータの数
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []


iter_per_epoch =max(train_size/batch_size,1)#x_train.shpae[0]=60000/batch_size=100 600回学習させれば全部学習させたことになる

for i in range(iters_num):
    #print("here")
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.gradient(x_batch,t_batch) #誤差逆伝播法によって勾配を求める
    
    for key in ('W1','b1','W2','b2'):
        #print("a",network.params[key].shape) #(784,50),(50,),(50,10),(10,)となって欲しい
        #print("b",grad[key].shape)#(784,50),(50,),(50,10),(10,)となればいい →ggradientがおかしそう grad[b1]→[50]のはずなのに[10]になってる？　解決
        network.params[key] -= learning_rate*grad[key]  #W1とW2の型がちがう
        #print("gradの変化",key,grad[key][0]) 
    
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
     #lossは変化している
    
    if i % iter_per_epoch == 0: #データ使い切った時 どこで学習を記録してるかっていうだけ
        #print(network.params["W1"])
        train_acc = network.accuracy(x_train,t_train)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("loss | "+str(loss))
        print(i,iter_per_epoch,i/iter_per_epoch,train_acc,test_acc)
    
        
    
    
#1万回繰り返すうち、600回で全データを学習させられる。(このデータの場合)
#だから600を境目にして精度を測ろうとしている

    


