import numpy as np
import matplotlib.pyplot as plt 
from sklearn  import datasets

img_size = 8
n_noise =16
eta = 0.001
n_learn = 10001
interval = 1000
batch_size =32

#訓練データの用意
digits_data =datasets.load_digits()
x_train = np.asarray(digits_data.data)
x_train = x_train/15 * 2 - 1

#全結合層の継承元
class BaseLayer:
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
        
class MiddleLayer(BaseLayer):
    def __init__(self,n_upper,n):
        self.w = np.random.randn(n_upper,n) * np.sqrt(2/n_upper)
        self.b = np.zeros(n)
        
    def forward(self,x):
        self.x = x
        self.u = np.dot(x,self.w) + self.b
        
    def backward(self,grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1) #ここは何をしている？
        self.grad_w = np.dot(self.x.T,delta)
        self.grad_b = np.sum(delta,axis = 0)
        self.grad_x = np.dot(delta,self.w.T)