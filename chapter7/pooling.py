#解像度を1落とす

import sys,os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad
    
    def forward(self, X):
        N, C, H, W = X.shape
        #指定してあげないと割り算がfloatになっちゃう
        OH=int(((H-self.pool_h)/self.stride)+1) #スライドする回数(縦)=行列にした時のデータ一1つの出力Hのサイズ
        OW=int(((W-self.pool_w)/self.stride)+1) #スライドする回数(横)行列にした時のデータ1つ枚の出力Wのサイズ
        col_X=im2col(X,self.pool_h,self.pool_w,self.stride,self.pad) #変形？
        # N * OH * OW, C * PH * PW → N * OH * OW * C, PH * PW
        col_X=col_X.reshape(-1, self.pool_h * self.pool_w) # -1とした次元は他の次元から推測されて自動的に決まる
        # -1→ N * OH * OW *C 
        # print(col_X)
        
        #最大値を求める
        col_max=np.max(col_X, axis=1)
        
        #行列を元の形に戻す
        out = col_max.reshape(N,OH,OW,C).transpose(0,3,1,2) #im2colの特性上、out[N][OH][OW][C]の順番で計算される。（決まりと思ったほうがいい）
        print(out)
        return out
        
data = np.array([[[[1,2,1,0],[0,1,2,3],[3,0,1,2],[2,4,0,1]],[[3,0,6,5],[1,2,2,3],[2,6,1,0],[4,5,4,1]],[[1,0,2,3],[3,1,0,2],[2,3,4,0],[4,2,1,3]]],[[[1,2,1,0],[0,1,2,3],[3,0,1,2],[2,4,0,1]],[[3,0,6,5],[1,2,2,3],[2,6,1,0],[4,5,4,1]],[[1,0,2,3],[3,1,0,2],[2,3,4,0],[4,2,1,3]]],[[[1,2,1,0],[0,1,2,3],[3,0,1,2],[2,4,0,1]],[[3,0,6,5],[1,2,2,3],[2,6,1,0],[4,5,4,1]],[[1,0,2,3],[3,1,0,2],[2,3,4,0],[4,2,1,3]]]])
pooling_sample = Pooling(2,2)
pooling_sample.forward(data)
# print(pooling_sample.forward(data))