#行列の計算
import numpy as np

X = np.array([1,2])
print(X.shape) #shape ->行列の次元数

W = np.array([[1,3,5],[2,4,6]])#3x2の配列
print(W.shape)
W.shape

Y = np.dot(X,W)
print(Y)
