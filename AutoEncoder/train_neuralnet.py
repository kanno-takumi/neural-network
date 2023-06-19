import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
# import keras
#from two_layer_net import TwoLayerNet

#0と1のデータ0から1000個を取得する。
#検証用データ1000から1100

def train_neuralnet():

#normalize 正規化
#one_hot_label t_trainに関して「”true”の時one-hot表現 ”false”の時数値で扱う」
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=False) 
    t_train_0=t_train[np.where(t_train==0)] #0のデータx
    x_train_0 = x_train[np.where(t_train==0)] #0のデータt
    print("x_train_0[0]",x_train_0[0])
# with open('train_out.txt', mode='w') as f:
#     f.write(str(x_train_0[0]))
    t_train_1 = t_train[np.where(t_train==1)] #1のデータx
    x_train_1 = x_train[np.where(t_train==1)] #1のデータt
    print("x_train_1",x_train_1)
    t_train_0to1 = np.concatenate([t_train_0, t_train_1])
    x_train_0to1 = np.concatenate([x_train_0,x_train_1])#0と1のデータx
    print(t_train_0to1)
    train_size = x_train_0to1.shape[0]
    print("train_size",train_size) #12665
    batch_size = 100
    iters_num = 1000 #何回勾配法使うか（イテレータ）
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = max(train_size/batch_size,1) #1エポックあたりの繰り返し数  12665/100 （1エポック=12665）の1まとまりで何回繰り返すか。足りなければ次のエポック
    print("iter_per_epoch",iter_per_epoch)#126.65



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
    
        if i % iter_per_epoch == 0: #iter_per_epoch=126.65 i=126になった時だけ発動 252 
            train_acc = network.accuracy(x_train,t_train)
            test_acc = network.accuracy(x_test,t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
        
        if i == 1000:
            #predictの出力結果に対して、255をかければ大丈夫？？？何ならかけなくても大丈夫？
            #line見ればわかる
            print("処理")
        


    #精度を確認しているだけ
    print("lossの出力",train_loss_list)
    print("train acc,test acc |" + str(train_acc)+","+str(test_acc))

