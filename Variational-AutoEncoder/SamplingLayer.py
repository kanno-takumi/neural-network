import numpy as np

class SamplingLayer:
    def forward(self,average,log_var):#(平均,分散)
        self.average = average #平均
        self.log_var = log_var #分散のlog
        
        self.epsilon = np.random.randn(*log_var.shape)
        self.z = average +self.epsilon*np.exp(log_var/2)#平均u+ε*分散σシグマ/2 = 潜在変数の正規分布？
        
    def backward(self,grad_z):
        self.grad_average = grad_z + self.average
        self.grad_log_var = grad_z * self.epsilon/2*np.exp(self.log_var/2) - 0.5*(1-np.exp(self.log_var))
        