#二層のニューラルネットワークを実装 #隠れ層が1つ
import numpy as np
import sys,os 
sys.path.append(os.pardir+"/..")
from common.functions import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from common.layers import *

#params ニューラルネットワークのパラメータを保持するディクショナリ変数
#params['W1']は1層目の重み
#params['b1' ]は1層目のバイアス

#grads 勾配を保持するディクショナリー変数 
#numerical_gradient()の返り値
#grads['W1']は1層目の重みの勾配
#grads['b1']は1層目のバイアスの勾配

#random.randn(入力データの個数,ニューロンの個数)

class TwoLayerNet:
    params=None
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params={} #ディクショナリ
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size) #平均0、分散1（標準偏差1）引数で配列の次元を返す
        self.params['b1'] = np.zeros(hidden_size) #0で初期化した配列を作る
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        #print(self.params)
        
        #レイヤの作成：誤差逆伝播法特有
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Sigmoid'] = Sigmoid()
        
        self.lastLayer = Reconstruction_Error()
        
    def predict(self,x):
        #誤差逆使わない時
        # W1,W2 = self.params['W1'],self.params['W2']
        # b1,b2 = self.params['b1'],self.params['b2']
        # a1 = np.dot(x,W1) + b1 #入力層からの計算
        # z1 = relu(a1) #活性化関数
        # a2 = np.dot(z1,W2) + b2      
        # y = sigmoid(a2) #出力は0〜1　本来は0から255の値が入ることになっているが正規化しているため
        #print(self.layers)
        #誤差逆使う時
        for layer in self.layers.values():
            # print(self.layers.values())
            #print("どこのlayerでエラーが起きているのか？",layer)
            x = layer.forward(x)
        return x
        
    
    def loss(self,x,t): #損失関数 #引数はbatchで入る
        y = self.predict(x)#784ニューロンとか
        #print("yのサイズ",y.shape) (100,784)
        return self.lastLayer.forward(y,t)
    #reconstruction内で、batchデータをまとめて処理してlossを計算している。
    
    def accuracy(self,x,t): #どれくらい合っているか
        y = self.predict(x)
        y = np.argmax(y,axis = 1) #配列の最大を返す
        t = np.argmax(t,axis = 1) 
        
        accuracy = np.sum( y==t ) / float(x.shape[0])
        return accuracy
    
    
    #順伝播の重みつけ
    # def numerical_gradient(self,x,t): #損失関数における勾配　＃x→入力データのこと t→教師データ
    #     loss_W = lambda : self.loss(x,t) #predictの結果と正解ラベルの交差エントロピー誤差  #無名関数
    #     grads = {} #ディクショナリー
        
    #     #なぜ分ける必要があるのか
    #     grads['W1'] = numerical_gradient(loss_W,self.params['W1']) #引数(損失関数、その関数に入れる入力)
    #     grads['b1'] = numerical_gradient(loss_W,self.params['b1'])#サイズを決めるためだけにparams['b1'] ['W1'] ['W2'] ['b2']
    #     grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
    #     grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
    #     return grads
    
    #誤差逆伝播法による重みづけ
    
    def gradient(self,x,t):
        self.loss(x,t)
        dout =1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db    
        
        return grads