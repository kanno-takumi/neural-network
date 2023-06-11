#勾配法（実際に移動した数値）の実装
import numpy as np
from numerical_gradient import numerical_gradient 

def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x = init_x
    #f→最適化したい関数,初期値、学習率、繰り返しの数
    for i in range(step_num): # 100回繰り返す
        grad = numerical_gradient(f,x) 
        x = x-lr*grad #ちょっとずつずらしていく処理を行う
        # x -= lr * grad
        
    return x

def function_sample(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0,4.0])
print(gradient_descent(function_sample,init_x,lr=0.0001,step_num=1000000)) #lr学習率は低く回数が多いほうが精度は高いが、時間がかかる。