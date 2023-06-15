#二層のニューラルネットワークを実装 #隠れ層が1つ
import numpy as np
import sys,os 
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

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
        
    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']
        #ここで再度W1,W2,b1,b2を定義して入れている
        
        a1 = np.dot(x,W1) + b1 #入力層からの計算
        z1 = sigmoid(a1) #活性化関数
        a2 = np.dot(z1,W2) + b2      
        y = softmax(a2)    
        #print("xのサイズ",x.shape) 
        #print("W1のサイズ",W1.shape)
        #print("aのサイズ",a1.shape)
        #print("yのサイズ",y.shape)
        return y # AIが出した結果
    
    def loss(self,x,t): #損失関数
        y = self.predict(x)
        return cross_entropy_error(y,t) #2.302944356298707のような値
    
    def accuracy(self,x,t): #どれくらい合っているか
        y = self.predict(x)
        y = np.argmax(y,axis = 1) #配列の最大を返す
        t = np.argmax(t,axis = 1) 
        
        accuracy = np.sum( y==t ) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t): #損失関数における勾配　＃x→入力データのこと t→教師データ
        loss_W = lambda : self.loss(x,t) #predictの結果と正解ラベルの交差エントロピー誤差  #無名関数
        grads = {} #ディクショナリー
        
        #なぜ分ける必要があるのか
        grads['W1'] = numerical_gradient(loss_W,self.params['W1']) #引数(損失関数、その関数に入れる入力)
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])#サイズを決めるためだけにparams['b1'] ['W1'] ['W2'] ['b2']
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        return grads
    
#net =  TwoLayerNet(input_size=784,hidden_size=100,output_size=10)
#print(net.params['W1'].shape) #(784,100) 784個のニューロンから100個のニューロンにそれぞれ繋がる
#print(net.params['b1'].shape) #(100,)
#print(net.params['W2'].shape) #(100,10)
#print(net.params['b2'].shape) #(10)

#x = np.random.rand(100,784) #ダミーの入力データ100枚 1枚あたりの入力784ニューロンかな???
#print("predict、出力層の出力サイズ",net.predict(x).shape) #(100,10)
 
#ここまでは勾配を求めるまで。
#これを使って学習させる（Wとbをずらしていく）
#別のファイルで



        
        
        
        
    
     