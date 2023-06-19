import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

#2次元配列から画像を作る
def make_img(img_array1):
    img_array2=reshape_img(img_array1)
    pil_img = Image.fromarray(np.uint8(img_array2))
    pil_img.show()
    pil_img.save("image.png")
    
    
#1次元(784)を2次元配列(28x28)に変換する（0〜255の値）
def reshape_img(img_array1):
    print(img_array1.shape)
    img_array2 = img_array1.reshape(28,28)
    print(img_array2.shape)
    return img_array2

def normalize_img():
    print("処理")
    
    
(x_train,t_train),(x_test,t_test) =load_mnist(flatten=True,normalize=False) #flattenはデフォルトでtrue
(x_train_normalize,t_train_normalize) = load_mnist(flatten=True,normalize=True)
    
img = x_train[0]
print(img)
label =t_train[0]
print(label)
make_img(img)

    
    