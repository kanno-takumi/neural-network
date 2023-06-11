import numpy as np
import sys,os
import time
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

train_loss_list =[]

#ハイパーパラメータ
iters_num = 1000 #バッチデータに対する試行回数
train_size = x_train.shape[0]
batch_size = 100
learning_rate =0.1 #イータη

network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

time_start = time.time()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #勾配の計算
    grad = network.numerical_gradient(x_batch,t_batch)
    #パラメータの更新(ちょっとずらす)
    for key in('W1','b1','W2','b2'):
        network.params[key] =network.params[key]-learning_rate * grad[key]
        
    #学習経過の記録
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
    
time_end = time.time()

result_time = time_end - time_start
print(result_time)
print(train_loss_list)    
    
    