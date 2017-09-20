# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
import sys

plt.style.use('ggplot')

# MNISTの手書き数字データのダウンロード
# #HOME/scikit_learn_data/mldata/mnist-original.mat にキャッシュされる
print 'fetch MNIST dataset'
mnist = fetch_mldata('MNIST original')
# mnist.data : 70,000件の784次元ベクトルデータ
# sample img in SourceImg folder
mnist.data   = mnist.data.astype(np.float32)
mnist.data  /= 255     # 0-1のデータに変換


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
    plt.clf()
    plt.cla()
    plt.close()

r_base = 1
for idx in range(r_base,r_base+70000,5): # I just want 1/5 of training set
    draw_digit(mnist.data[idx],idx) 
    print str(idx)
