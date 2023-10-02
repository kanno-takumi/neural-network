import numpy as np
#np.set_printoptions(threshold=np.inf)
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from image_show import make_img
# import keras
#from two_layer_net import TwoLayerNet

#0と1のデータ0から1000個を取得する。
#検証用データ1000から1100

def train_neuralnet():

#normalize 正規化
#one_hot_label t_trainに関して「”true”の時one-hot表現 ”false”の時数値で扱う」
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=False) 
    #t_train_0 = t_train[np.where(t_train==0)] #0のデータt 教師データはいらない
    x_train_0 = x_train[np.where(t_train==0)] #0のデータx　
    x_test_0 = x_test[np.where(t_test==0)]
    #t_test_0 = t_test[np.where(t_test==0)]　教師データはいらない
    print("x_train_0[0]",x_train_0[0])
    #make_img(x_train_0[0],"sample.png")
# with open('train_out.txt', mode='w') as f:
#     f.write(str(x_train_0[0]))
    #t_train_1 = t_train[np.where(t_train==1)] #1のデータ
    x_train_1 = x_train[np.where(t_train==1)] #1のデータt 
    print("x_train_1",x_train_1)
    
    #t_train_0to1 = np.concatenate([t_train_0, t_train_1])
    #x_train_0to1 = np.concatenate([x_train_0,x_train_1])#0と1のデータx
    #print(t_train_0to1)
    train_size = x_train_1.shape[0]
    print("train_size",train_size) #12665
    batch_size = 100
    iters_num = 100 #何回勾配法使うか（イテレータ）
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    #iter_per_epoch = max(train_size/batch_size,1) #1エポックあたりの繰り返し数  12665/100 （1エポック=12665）の1まとまりで何回繰り返すか。足りなければ次のエポック
    #print("iter_per_epoch",iter_per_epoch)#126.65



    network = TwoLayerNet(input_size=784,hidden_size=2,output_size=784)

    for i in range(iters_num): #1バッチに対して勾配法を用いる回数
        print("here")
        batch_mask = np.random.choice(train_size,batch_size)#例 5207 5424 4372 133 114 4522 4704　ランダムにデータを撮ってきたいだけ
        x_batch = x_train_1[batch_mask]
        #print(batch_mask)
        #print("x_batch",x_batch[0])
        t_batch = x_train_1[batch_mask] #入出力で比べるのが同じものでないと困る
    
        grad = network.numerical_gradient(x_batch,t_batch)
    
        for key in ('W1','b1','W2','b2'):
            network.params[key] = network.params[key] - learning_rate * grad[key]
        
        
        loss = network.loss(x_batch,t_batch)
        train_loss_list.append(loss)
        # with open('train_out.txt', mode='a',newline='\n') as f:
        #     f.writelines(str(loss)+",")
    
        # if i % iter_per_epoch == 0: #iter_per_epoch=126.65 i=126になった時だけ発動 252 
        #     train_acc = network.accuracy(x_train_0,t_train_0)
        #     test_acc = network.accuracy(x_test_0,t_test_0)
        #     train_acc_list.append(train_acc)
        #     test_acc_list.append(test_acc)
        
        if (i == iters_num- 1):
            #predictの出力結果に対して、255をかければ大丈夫？？？何ならかけなくても大丈夫？
            #line見ればわかる
            #make_img(network.predict(x_batch[0]),"x_out.png")
            print("x_batch[0]",x_batch[0])
            print("t_batch[0]",t_batch[0]) #ここまでは普通にbatchデータが機能している
            make_img(network.predict(x_batch[0]),"学習済みデータ.png") #predictした状態のものを引数で与えている
            print("make_imgが完了しました。")
            make_img(t_batch[0],"元のデータの例.png")

    #精度を確認しているだけ
    print("lossの出力",train_loss_list)
    #print("accuracy",network.accuracy(t_batch[0]))
    # print("W1",network.params['W1'])
    # print("W2",network.params['W2'])
    with open('W1.txt', mode='w') as f:
        f.write(str(network.params['W1']))
    with open('W2.txt', mode='w') as f:
        f.write(str(network.params['W2']))
   # print("train acc,test acc |" + str(train_acc)+","+str(test_acc))
    
train_neuralnet()

