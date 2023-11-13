import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(50,10,10000) #(平均,標準偏差,個数)
#random.normalは正規分布に従う乱数を生成している

plt.hist(x,bins=50)#binsは棒の数
plt.show()


