import numpy as np

class ParamsLayer:
    def __init__(self,n_upper,n): #n_upper→上の層のニューロン数
        self.w = np.random.randn(n_upper,n)/np.sqrt(n_upper)
        self.b = np.zeros(n)
        
    def forward(self,x):
        self.x = x
        u = np.dot(x,self.x)+ self.b
        self.y = u #self.y (u) = 潜在変数 
        
    def backward(self,grad_y):
        delta = grad_y
    
        #誤差逆伝播法　行列版
        self.grad_w = np.dot(self.x.T,delta) #grad_wはwを少しずらした時の変化。特に
        self.grad_b = np.sum(delta,axis=0)
        self.grad_x = np.dot(delta,self.w.T)