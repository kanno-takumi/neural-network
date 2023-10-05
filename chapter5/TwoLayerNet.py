import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from Affine import Affine
from ReLU import Relu

class TwoLayerNet:
    params=None
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        #レイヤの作成
        self.layers = OrderedDict() #pythonに元々用意されている
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1']) #Affine層は行列の計算(X W)　ここではAffineクラスのインスタンスを持った状態
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self,x):
        for layer in self.layers.values(): #layersの中の値に対して繰り返し作業を行う(affine1の中にはたくさんの重みとバイアスを持っている) values→例えばAffine1とか
            #print("layer",layer)
            x =layer.forward(x) 
        #print("予想値:",x)
        return x
    
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    def accuracy(self,x,t):
        #print("accuracy計算")
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim  !=1 : t=np.argmax(t,axis=1)
        
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy
    
    # def numerical_gradient(self,x,t):#数値微分によって勾配を求める
    #     loss_W = lambda W:self.loss(x,t)   #lambdaの意味
        
    #     grads = {}
    #     grads['W1'] = numerical_gradient(loss_W,self.params['W1']) #勾配
    #     grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
    #     grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
    #     grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        
    #     return grads
    
    def gradient(self,x,t):#　誤差逆伝播方によって勾配を求める。
        #forward
        self.loss(x,t)
        
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)#最後の層だけここで直接計算している。
        # print("ここのdoutは表示されている",dout)
        layers = list(self.layers.values())
        # print("layers",layers)
        #print("layers",layers)
        layers.reverse() #リストを逆順に使う #reverse()が返すのはNone
        #print("layers",layers)
        for layer in layers:
            #print("layer変わるタイミング",layer)
            # print("ここのdoutは？",dout)
            dout = layer.backward(dout)#層によってbackwardの処理が変わってくるので計算はそれぞれの層で任せている。流れはここで定義している。
            
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW #[784,50]   Affine層のフィールドにdW,dbが用意されている
        grads['b1'] = self.layers['Affine1'].db #[50,]
        grads['W2'] = self.layers['Affine2'].dW #[50,10]
        grads['b2'] = self.layers['Affine2'].db #[10,]
        
        #print("grads[b1]",grads['b1'])
        
        return grads