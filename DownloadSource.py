# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys

plt.style.use('ggplot')

# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 100

# 学習の繰り返し回数
n_epoch   = 1

# 中間層の数
n_units   = 1000

# MNISTの手書き数字データのダウンロード
# #HOME/scikit_learn_data/mldata/mnist-original.mat にキャッシュされる
print 'fetch MNIST dataset'
mnist = fetch_mldata('MNIST original')
# mnist.data : 70,000件の784次元ベクトルデータ
# sample img in SourceImg folder
mnist.data   = mnist.data.astype(np.float32)
mnist.data  /= 255     # 0-1のデータに変換
# mnist.target : 正解データ（教師データ）
mnist.target = mnist.target.astype(np.int32)

# 手書き数字データを描画する関数
def draw_digit(data,i):
    size = 28
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size),range(size))
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    plt.savefig("./SourceImg/"+str(i))
    #plt.show()

#draw_digit(mnist.data[5],5)
#draw_digit(mnist.data[12345],12345)
#draw_digit(mnist.data[33456],34356)


#np.savetxt('test.txt', mnist.data[:60000],fmt="%i", delimiter=',')
r_base = 55000
for idx in range(r_base,r_base+10):
    draw_digit(mnist.data[idx],idx)
