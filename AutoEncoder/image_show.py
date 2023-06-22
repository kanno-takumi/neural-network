import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

#1次元配列の正規化されたデータを元に戻す
def denormalize(img_normalized):
    img_denormalized=img_normalized*255
    print("まえ",img_normalized)
    print("あと",img_denormalized)
    
    
    return img_denormalized

#1次元(784)を2次元配列(28x28)に変換する（0〜255の値）
def reshape_img(img_array1):
    img_array1 = denormalize(img_array1)
    print(img_array1.shape)
    img_array2 = img_array1.reshape(28,28)
    print(img_array2.shape)
    return img_array2

#2次元配列から画像を作る
def make_img(img_array1,imgname):#.pngで保存するのを忘れない
    print("1次元の配列(784)のはず",img_array1)
    print("img_array1.shape",img_array1.shape)
    img_array2=reshape_img(img_array1)
    pil_img = Image.fromarray(np.uint8(img_array2))
    #pil_img.show()
    pil_img.save(imgname)
    
    

    
    
    
# (x_train,t_train),(x_test,t_test) =load_mnist(flatten=True,normalize=False) #flattenはデフォルトでtrue
(x_train_normalize,t_train_normalize),(x_test_normalize,t_test_normalize) = load_mnist(flatten=True,normalize=True)
    
# img = x_train[0]
# img2 = x_train_normalize[0]
# print("img",img)
# print("img2",img2)
# label =t_train[0]
# print(label)
# print("img.shape",img.shape)
# print("img2.shape",img2.shape)
#make_img(img)


#sampleで入れてみただけ
print("x_train_normalize",x_train_normalize[0])
make_img(x_train_normalize[0],"sample.png")

    
    