#commonの方にもlayers/Affineがあるからそっちから呼び出されている
import numpy as np
import sys, os
sys.path.append(os.pardir)

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
        
    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b #xが入力値、Wが重み　xに対してそれぞれWがあるため行列で計算する　bも多分配列
        
        return out
    
    def backward(self,dout):
        #print("Affineクラス内でdoutの確認",dout)
        dx = np.dot(dout,self.W.T)#転置    順伝播では (X,W)の順番
        #print("確認1")
        self.dW = np.dot(self.x.T,dout)
        #print("確認2")
        self.db = np.sum(dout,axis = 0) #行で合計している→この場合は self.db = doutと同じ
        #print("確認3")
        return dx