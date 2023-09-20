import sys,os
sys.path.append(os.pardir)
from common.functions import softmax
from common.functions import cross_entropy_error

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self,t)
        
        return self.loss
    
    def backward(self,dout=1): #softmaxWithLossの偏微分した時、δL/δa=y(k)-t(k)となる aはパラメータとする。
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size #なぜbatch_sizeで割る必要があるのか→forwardのlossの計算では、cross_entropy_error内で、lossの合計を取り、それをbatchsizeで割っているため、逆の処理を行う。
        
        return dx