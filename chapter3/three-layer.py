import numpy as np
import sigmoid
import identify

X=np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])#一層目のそれぞれのバイアス（しきい値）
A1=np.dot(X,W1)+B1

Z1=sigmoid.sigmoid(A1) #sigmoid function 1層目のXと同じ
print(Z1)

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])
A2=np.dot(Z1,W2)+B2

Z2=sigmoid.sigmoid(A2)
print(Z2)

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])
A3=np.dot(Z2,W3)+B3
Y=identify.identify_function(A3)
print(Y)#result