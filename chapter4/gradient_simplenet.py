#単純なニューラルネットワークを例にして勾配を求める実装　※ただのニューラルネットワークの実装だから、出力で何か得られるわけではない。

import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) #重みをランダムに入れる
        
    def predict(self,x): #次の層に送る計算をする(W*X)
        return np.dot(x,self.W)
    
    def loss(self, x, t): #教師データt 損失関数
        z = self.predict(x) #計算結果から正解と誤差を求める
        y = softmax(z)
        loss = cross_entropy_error(y,t) #交差エントロピー誤差
        
        return loss

net = SimpleNet()
#print(net.W)
x = np.array([0.6,0.9]) #入力層(手書き文字の認識から一度離れる) AND,OR,NORゲートのような数値のものを想定する 
p = net.predict(x)

t = np.array([0,0,1])
net.loss(x,t)

#ここから誤差関数の勾配を求める
def f(W):
    return net.loss(x,t)

dW = numerical_gradient(f,net.W)#(関数=どの関数について勾配を求めるのか,勾配における入力=重み 配列)
#自分で作ったnumerical_gradientは入力が1限配列にしか対応していない
#今numerical_gradientでの入力（引数）は 2 * 3　の配列だから使えない 

#このあと実際にずらしていく作業があるが、参考書にないので省略
#gradient_descentへとほんとは続く
    
