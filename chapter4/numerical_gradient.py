#勾配の実装 (偏微分をベクトルとしてまとめたもの)

import numpy as np
def numerical_gradient(f,x): #(function,入力)　＃ここでのxは損失関数に入れるxの配列
    h = 1e-4 #0.0001 1^(-4)
    grad = np.zeros_like(x) #xと同じサイズの配列を作る
    
    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h)の計算
        x[idx]=tmp_val+h
        fxh1 = f(x) #x0,x1,x2,x3....を関数に入れている
        
        #f(x-h)の計算
        x[idx] = tmp_val-h
        fxh2 = f(x) 
        
        grad[idx] = (fxh1-fxh2)/(2*h) #傾き
        x[idx] = tmp_val #x[idx]色々変えちゃったから元に戻しただけ
        
    return grad

#２変数 #損失関数ではない #ただの勾配を求める関数.
def function_sample_1(x):
    return x[0]**2+x[1]**2

#3変数　（関数を変えれば変数の数は問題ではない）
def function_sample_2(x):
    return x[0]**2+x[1]**2+x[2]**2

#print("傾き",numerical_gradient(function_sample_1,np.array([-1.0,4.0])))
#print("傾き",numerical_gradient(function_sample_2,np.array([1.0,2.0,3.0])))


#出力の数値    プラスマイナスにより(→,↓)など向きを矢印で、大きさを数値で表してくれる　例（100,5）だったらx軸の正の方にめちゃくちゃ上がっている、yの正にちょっとだけ上がっていることを表している

#最終的なパラメータの値は求めてない（ここでは移動すべき方向と向きだけ）