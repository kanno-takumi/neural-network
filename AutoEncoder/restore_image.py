#入力画像、復元画像の表示

#数値を画像データに変換しないといけない。
#毎回やるわけにはいかない。最後だけ表示したいとしたらどうする？？？？
#Wとbをどこかに保存しておかなければならない？
#最後に対してのみ数値データ分繰り返して画像として表示させる必要がある。

#➀1次元配列を2次元配列に変換する。28 x 28
#➁plt.imshow(img_rgb, cmap = 'gray', vmin = 0, vmax = 255, interpolation = 'none')
from PIL import Image
# def restore(): 


#未実装
def img_show(img):
    pil_img = Image.fromarray(np.unit)