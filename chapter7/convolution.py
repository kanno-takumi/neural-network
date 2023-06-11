import sys,os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np

class Convolution:
    #初期値
    W=None
    b=None
    stride=None
    pad=None
    
    #コンストラクタ
    def __init__(self,W,b,stride=1,pad=0):#(self=Convollutionを指している,そのあとは引数)
        self.W=W #フィルターの中身（FN,C,FH,FW）
        self.b=b #バイアス
        self.stride=stride #stride
        self.pad=pad #padding
    
    #関数(順伝播)
    def forward(self,X):
        FN, C, FH, FW = self.W.shape #shape→各次元ごとの要素数 フィルタ数、チャンネル数、フィルタサイズ縦、フィルタサイズ横
        N,C, H, W = X.shape  #入力x(バッチ数、チャンネル数、高さ、幅)
        OH = int(1+(H+2*self.pad-FH)/self.stride)#出力結果(たてサイズ)　パディングとストライドを考えると複雑な式に見えるけど具体例で考えれば簡単
        OW = int(1+(W+2*self.pad-FW)/self.stride)#出力結果（よこサイズ）
        
        #col=>行列　
        col_X = im2col(X,FH,FW,self.stride,self.pad)   #入力データの展開 (バッチ数N,チャンネル数C,高さH,幅w)→（C*H*W,N）
        col_W = self.W.reshape(FN,-1).T #フィルタの展開 (FN,C,FH,FW)→(C*FH*FW,FN)　箱がいっぱいあるイメージ -1を指定すると自動で決定される
        out = np.dot(col_X,col_W) + self.b
    
        out = out.reshape(N,OH,OW,-1).transpose(0,3,1,2) #行列結果を3次元に変換
        
        return out
        
        
batch_data_3 = np.array([[[[1,2,1,0],[0,1,2,3],[3,0,1,2],[2,4,0,1]],[[3,0,6,5],[1,2,2,3],[2,6,1,0],[4,5,4,1]],[[1,0,2,3],[3,1,0,2],[2,3,4,0],[4,2,1,3]]],[[[1,2,1,0],[0,1,2,3],[3,0,1,2],[2,4,0,1]],[[3,0,6,5],[1,2,2,3],[2,6,1,0],[4,5,4,1]],[[1,0,2,3],[3,1,0,2],[2,3,4,0],[4,2,1,3]]],[[[1,2,1,0],[0,1,2,3],[3,0,1,2],[2,4,0,1]],[[3,0,6,5],[1,2,2,3],[2,6,1,0],[4,5,4,1]],[[1,0,2,3],[3,1,0,2],[2,3,4,0],[4,2,1,3]]]])
filter_3 = np.array([[[[2,3],[4,2]],[[3,6],[5,2]],[[4,4],[1,2]]],[[[2,3],[4,2]],[[3,6],[5,2]],[[4,4],[1,2]]],[[[2,3],[4,2]],[[3,6],[5,2]],[[4,4],[1,2]]]])
b = np.array([1,2,3])

        
    
self1 = Convolution(filter_3,b)
out = self1.forward(batch_data_3)
print(out)

