import numpy as np

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        #ここは動いている
        self.mask = (x<=0) #xが0以下の時trueとなる
        out = x.copy() #最初に初期化している。
        out[self.mask] = 0 #0以下のやつだけ0へと変換される。
        #print(self.mask)
        #print(out[self.mask])  
        return out #返すのは数値
    
    def backward(self,dout): #d_out
        #print("ReLUの中でdoutの確認",dout)
        #print(type(self.mask))
        #print(type(dout))
        #print("shape",dout.shape)
        #print("selfmask確認",self.mask)
        dout[self.mask] = 0 #0以下のやつだけ0へと変換される。
        dx = dout
        
        return dx
    
# relu = Relu()
# print(relu.forward(np.array([[1.0,-0.5],[2.0,3.0]])))