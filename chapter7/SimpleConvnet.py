#単純な畳み込みニューラルネットワークの実装
import numpy as np
from collections import OrderedDict
from convolution import Convolution 
from pooling import Pooling
from affine import Affine
from softmax_with_loss import SoftmaxWithLoss
from relu import Relu

class SimpleConvNet:
    def __init__(self,input_dim=(1,28,28),#引数 #input_dim 入力データの（チャンネル、高さ、幅）
                 conv_param={'filter_num':30,'filter_size':5,'pad':0,'stride':1},# "conv_param" という名前のdirectionary(filter_num,filtersize,pad,strideが存在する)
                 hidden_size=100,
                 output_size=10,
                 weight_init_std=0.01 #初期化の際の重みの標準偏差
                 ):
        #引数として得た情報をfieldとして保存するだけ
        filter_num=conv_param['filter_num']
        filter_size=conv_param['filter_size']
        filter_pad=conv_param['pad']
        filter_stride=conv_param['stride']
        input_size=input_dim[1]
        #畳み込み層、プーリング層の出力サイズを計算
        conv_output_size=(int(filter_num*(conv_output_size/2)*(conv_output_size/2)))
        pool_output_size=int(filter_num*(conv_output_size/2)*(conv_output_size/2))       
        
        #重みパラメータの初期化
        #学習に必要なparameterは1層目の畳み込み層と、2つの全結合層の重み
        self.params = {}
        #畳み込み層  \この記号は改行
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num,input_dim[0],
                                            filter_size,filter_size)
        self.params['b1'] = np.zeros(filter_num)
        
        #全結合層 \は改行
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size,
                                            hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        
        #全結合層
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        #最後に必要なレイヤを作成していく
        self.layers = OrderedDict() #OrderedDictとは→順序付き辞書
        #OrderDictの中に畳み込み層、プーリング層、全結合層などを入れていく
        self.layers['Conv1'] = Convolution(
                                            #引数 #保存したdictionaryから呼び出す
                                           self.params['W1'], 
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad']
                                           )
        self.layers['Relu1'] = Relu()   
        
        self.layers['pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)#引数  
        
        self.layers['Affine1'] = Affine(self.params['W2'],self.params['b2'])#引数
        self.layers['Relu2'] = Relu()  
        
        self.layers['Affine2'] = Affine(self.params['W3'],self.params['b3'])#引数
        
        self.last_layer = SoftmaxWithLoss()
        
    #ここまでがSimpleConvNetの初期化で行う処理
    #初期化が終わったら推論を行うpedictメソッドと、損失関数の値を求めるlossメソッドを実装すればOK
    
    def predict(self,X): #Xは入力データ predictとは、入力と重みから次の層の値を計算する
        for layer in self.layers.values():
            X =layer.forward(X)
        return  X
         
    def loss(self,X,t): # tは教師ラベル
        y = self.predict(X) #出力
        return self.lastLayer.forward(y,t) #出力と教師データを損失関数にぶち込む
        
        
    #損失関数を元に勾配を求める(誤差逆伝播法を使う) 
    #損失関数が最小になるようにパラメータを設定したい→それを見つけるのが勾配法
    def gradient(self,X,t):
        #forward
        self.loss(X,t)
        
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        #設定
        #勾配は進む方向を教えてくれる 
        grads= {} # gradsというディクショナリに各重みパラメータの勾配を格納する
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conb1'].db 
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        
        return grads
    
    #あとは学習するためのコード
        
    
        
                            
        
         
         
         
# 質問 
# 変な記号　/ 
# OrderdDictについて                  
         