import numpy as np
# import cupy as np  # GPUの場合
import matplotlib.pyplot as plt
from sklearn import datasets

# -- 各設定値 --
img_size = 8  # 画像の高さと幅
n_in_out = img_size * img_size  # 入出力層のニューロン数
n_mid = 16  # 中間層のニューロン数
n_z = 2

eta = 0.001  # 学習係数
epochs = 201 
batch_size = 32
interval = 20  # 経過の表示間隔

# -- 訓練データ --
digits_data = datasets.load_digits()
x_train = np.asarray(digits_data.data)
x_train /= 15  # 0-1の範囲に
t_train = digits_data.target

# -- 全結合層の継承元 --
class BaseLayer:
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

# -- 中間層 --
class MiddleLayer(BaseLayer):
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) * np.sqrt(2/n_upper)  # Heの初期値
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u) # ReLU
    
    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T) 

# -- 正規分布のパラメータ（平均,分散）を求める層 --
class ParamsLayer(BaseLayer):
    def __init__(self, n_upper, n):#今回は16,2
        #正規分布の乱数生成（平均0分散1）
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)  # 初期値
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = u  # 恒等関数
        # print("x,u",x,u)
    
    def backward(self, grad_y):
        delta = grad_y

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T) 

# -- 出力層 --
class OutputLayer(BaseLayer):
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)  # Xavierの初期値
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = 1/(1+np.exp(-u))  # シグモイド関数

    def backward(self, t):
        delta = self.y - t
        
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T) 

# -- 潜在変数をサンプリングする層 --
class LatentLayer:
    def forward(self, mu, log_var):
        self.mu = mu  # 平均値
        #print("mu",self.mu)
        self.log_var = log_var  # 分散の対数
        #print("var",log_var)

        self.epsilon = np.random.randn(*log_var.shape)
        self.z = mu + self.epsilon*np.exp(log_var/2)
    
    def backward(self, grad_z):
        self.grad_mu = grad_z + self.mu
        self.grad_log_var = grad_z*self.epsilon/2*np.exp(self.log_var/2) \
         - 0.5*(1-np.exp(self.log_var))

# -- 各層の初期化 --
# Encoder
middle_layer_enc = MiddleLayer(n_in_out, n_mid)
mu_layer = ParamsLayer(n_mid, n_z)
log_var_layer = ParamsLayer(n_mid, n_z)
z_layer = LatentLayer()
# Decoder
middle_layer_dec = MiddleLayer(n_z, n_mid)
output_layer = OutputLayer(n_mid, n_in_out)

# -- 順伝播 --
def forward_propagation(x_mb):
    # Encoder
    middle_layer_enc.forward(x_mb)
    mu_layer.forward(middle_layer_enc.y)
    log_var_layer.forward(middle_layer_enc.y)
    z_layer.forward(mu_layer.y, log_var_layer.y)
    # Decoder
    middle_layer_dec.forward(z_layer.z)
    output_layer.forward(middle_layer_dec.y)

# -- 逆伝播 --
def back_propagation(t_mb):
    # Decoder
    output_layer.backward(t_mb)
    middle_layer_dec.backward(output_layer.grad_x)
    # Encoder
    z_layer.backward(middle_layer_dec.grad_x)
    log_var_layer.backward(z_layer.grad_log_var)
    mu_layer.backward(z_layer.grad_mu)
    middle_layer_enc.backward(mu_layer.grad_x + log_var_layer.grad_x)

# -- パラメータの更新 --
def update_params():
    middle_layer_enc.update(eta)
    mu_layer.update(eta)
    log_var_layer.update(eta)
    middle_layer_dec.update(eta)
    output_layer.update(eta)

# -- 誤差を計算 --
def get_rec_error(y, t):
    eps = 1e-7
    return -np.sum(t*np.log(y+eps) + (1-t)*np.log(1-y+eps)) / len(y) 

def get_reg_error(mu, log_var):
    return -np.sum(1 + log_var - mu**2 - np.exp(log_var)) / len(mu)

rec_error_record = []
reg_error_record = []
total_error_record = []
n_batch = len(x_train) // batch_size  # 1エポックあたりのバッチ数
for i in range(epochs):
        
    # -- 学習 -- 
    index_random = np.arange(len(x_train))
    np.random.shuffle(index_random)  # インデックスをシャッフルする
    for j in range(n_batch):
        
        # ミニバッチを取り出す
        mb_index = index_random[j*batch_size : (j+1)*batch_size]
        x_mb = x_train[mb_index, :]
        
        # 順伝播と逆伝播
        forward_propagation(x_mb)
        back_propagation(x_mb)
        
        # 重みとバイアスの更新
        update_params()

    # -- 誤差を求める --
    forward_propagation(x_train)

    rec_error = get_rec_error(output_layer.y, x_train)
    reg_error = get_reg_error(mu_layer.y, log_var_layer.y)
    total_error = rec_error + reg_error

    rec_error_record.append(rec_error)
    reg_error_record.append(reg_error)
    total_error_record.append(total_error)

    # -- 経過の表示 -- 
    if i%interval == 0:
        print("Epoch:", i, "Rec_error:", rec_error, "Reg_error", reg_error, "Total_error", total_error)

plt.plot(range(1, len(rec_error_record)+1), rec_error_record, label="Rec_error")
plt.plot(range(1, len(reg_error_record)+1), reg_error_record, label="Reg_error")
plt.plot(range(1, len(total_error_record)+1), total_error_record, label="Total_error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

# 潜在変数を計算
forward_propagation(x_train)

# 潜在変数を平面にプロット
plt.figure(figsize=(8, 8))
for i in range(10):
    zt = z_layer.z[t_train==i]
    z_1 = zt[:, 0]  # y軸
    z_2 = zt[:, 1]  # x軸
    marker = "$"+str(i)+"$"  # 数値をマーカーに
    plt.scatter(z_2.tolist(), z_1.tolist(), marker=marker, s=75)

plt.xlabel("z_2")
plt.ylabel("z_1")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid()
plt.show()

# 画像の設定
n_img = 16  # 画像を16x16並べる
img_size_spaced = img_size + 2
matrix_image = np.zeros((img_size_spaced*n_img, img_size_spaced*n_img))  # 全体の画像

# 潜在変数
z_1 = np.linspace(3, -3, n_img)  # 行
z_2 = np.linspace(-3, 3, n_img)  # 列

#  潜在変数を変化させて画像を生成
for i, z1 in enumerate(z_1):
    for j, z2 in enumerate(z_2):
        x = np.array([float(z1), float(z2)])
        middle_layer_dec.forward(x)  # Decoder
        output_layer.forward(middle_layer_dec.y)  # Decoder
        image = output_layer.y.reshape(img_size, img_size)
        top = i*img_size_spaced
        left = j*img_size_spaced
        matrix_image[top : top+img_size, left : left+img_size] = image

plt.figure(figsize=(8, 8))
plt.imshow(matrix_image.tolist(), cmap="Greys_r")
plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  # 軸目盛りのラベルと線を消す
plt.show()