#coding:utf-8
import theano
import theano.tensor as T
import numpy as np

class Conv2d(object):
    def __init__(self, input, out_c, in_c, k_size):
        self._input = input #入力されるシンボル
        self._out_c = out_c #出力チャネル数
        self._in_c = in_c #入力チャネル数
        w_shp = (out_c, in_c, k_size, k_size) #重みのshape
        w_bound = np.sqrt(6. / (in_c * k_size * k_size + \
                        out_c * k_size * k_size)) #重みの制約
        #重みの定義
        self.W = theano.shared( np.asarray(
                        np.random.uniform( #乱数で初期化
                            low=-w_bound,
                            high=w_bound,
                            size=w_shp),
                        dtype=self._intype.dtype), name ='W', borrow=True)
        b_shp = out_c, #バイアスのshape
        #バイアスの定義(ゼロで初期化)
        self.b = theano.shared(np.zeros(b_shp,
                        dtype=self._input.dtype), name ='b', borrow=True)
        #畳み込みのシンボルの定義
        self.output = T.nnet.conv.conv2d(self._input, self.W) \
                        + self.b.dimshuffle('x', 0, 'x', 'x')
        #更新されるパラメータを保存
        self.params = [self.W, self.b]

class relu(object):
    def __init__(self, input):
        self._input = input
        self.output  = T.switch(self._input < 0, 0, self._input)


from theano.tensor.signal import pool

class Pool2d(object):
    def __init__(self, input, k_size, st, pad=0, mode='max'):
        self._input = input
        #プーリング層のシンボルの定義
        self.output = pool.pool_2d(self._input, 
                            (k_size, k_size), #カーネルサイズ
                            ignore_border=True, #端の処理(基本的にTrueでok,詳しくは公式Documentへ)
                            st=(st, st), #ストライド
                            padding=(pad, pad), #パディング
                            mode=mode) #プーリングの種類('max', 'sum', 'average_inc_pad', 'average_exc_pad')

class FullyConnect(object):
    def __init__(self, input, inunit, outunit):
        self._input = input
        #重みの定義
        W = np.asarray(
            np.random.uniform(
            low=-np.sqrt(6. / (inunit + outunit)),
            high=np.sqrt(6. / (inunit + outunit)),
            size=(inunit, outunit)
            ),
            dtype=theano.config.floatX)
        self.W = theano.shared(value=W, name='W', borrow=True)
        #バイアスの定義
        b = np.zeros((outunit,), dtype=theano.config.floatX) #ゼロで初期化
        self.b = theano.shared(value=b, name='b', borrow=True)
        #全結合層のシンボルの定義
        self.output = T.dot(self._input, self.W) + self.b
        #更新されるパラメータを保存
        self.params = [self.W, self.b]

class softmax(object):
    def __init__(self, input, y):
        self._input = input
        #softmaxのシンボル定義
        self.output = nnet.softmax(self._input)
        #cross entropyのシンボル定義(数式的にはsumですがここではmeanを用います)
        self.cost = -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

import gzip
import cPickle

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    set_x = theano.shared(np.asarray(data_x,
                  dtype=theano.config.floatX).reshape(-1,1,28,28),
                  borrow=True)
    set_y = T.cast(theano.shared(np.asarray(data_y,
                  dtype=theano.config.floatX), borrow=True), 'int32')
    return set_x, set_y

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)

train_set_x, train_set_y = shared_dataset(train_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
test_set_x, test_set_y = shared_dataset(test_set)

x = T.tensor4() #入力データのシンボル
y = T.ivector() #出力データのシンボル

conv1 = Conv2d(x, 20, 1, 5) #xを入力とし、出力が20チャネル、入力が1チャネル、カーネルサイズ5
relu1 = relu(conv1.output) #conv1の出力を入力とする
pool1 = Pool2d(relu1.output, 2, 2) #relu1の出力を入力とし, カーネルサイズ2、ストライド2

conv2 = Conv2d(pool1.output, 50, 20, 5) #poo1の出力を入力とし、出力が50チャネル、入力が20チャネル、カーネルサイズ5
relu2 = relu(conv2.output) #conv2の出力を入力とする
pool2 = Pool2d(relu2.output, 2, 2) #relu2の出力を入力とし, カーネルサイズ2、ストライド2

fc1_input = pool2.output.flatten(2) #pool2の出力のシンボルはT.tensor4のため、flatten()を使って全結合層の入力のシンボルに合わせる
fc1 = FullyConnect(fc1_input, 50*4*4, 500) #入力のユニット数が50*4*4(チャネル数*縦*横)、出力のユニット数が500
relu3 = relu(fc1.output)
fc2 = FullyConnect(relu3.output, 500, 10) #入力のユニット数が500、出力のユニット数が10(10クラス分類のため)
loss = softmax(fc2.output, y)

#学習される全パラメータをリストにまとめる
params = conv1.params + conv2.params + fc1.params + fc2.params 
#各パラメータに対する微分を計算
grads = T.grad(loss.cost, params)
#学習率の定義
learning_rate = 0.001
#更新式を定義
updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]
#学習のtheano.functionを定義
index = T.lscalar()
batch_size = 128
train_model = theano.function(inputs=[index], #入力は学習データのindex
                       outputs=loss.cost, #出力はloss.cost
                       updates=updates, #更新式
                       givens={
                            x: train_set_x[index: index + batch_size], #入力のxにtrain_set_xを与える
                            y: train_set_y[index: index + batch_size] #入力のyにtrain_set_yを与える
                       })
for i in range(0, train_set_y.get_value().shape[0], batch_size):
    train_model(i)

pred = T.argmax(loss.output, axis=1) #予測された確率が最も高いクラスを返す
error = T.mean(T.neq(pred,y)) #予測されたクラスを正解ラベルと比較
test_model = theano.function(inputs=[index],
                             outputs=error,
                             givens={
                             x: test_set_x[index: index + batch_size],
                             y: test_set_y[index: index + batch_size]
                             })

val_model = theano.function(inputs=[index],
                             outputs=error,
                             givens={
                             x: test_set_x[index: index + batch_size],
                             y: test_set_y[index: index + batch_size]
                             })

test_losses = [test_model(i)
               for i in range(0, test_set_y.get_value().shape[0], batch_size] #バッチごとの平均のlossをリストに保存
mean_test_loss = np.mean(test_losses) #全体の平均を算出
