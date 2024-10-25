import tensorflow as tf
from tensorflow import keras as ks
import abc
"""
时间 23年3月5日
内容 定义网络结构文件
     后续用koopman可能需要更复杂的网络结构
作者 冯胖子 
时间 23年4月25日
更改 删去 relu_4 sigmoid_3 MLP
     在koopman 的net.py文件 基础上修改，结合昨天写的 RPv1中的编码器解码器结构，实现比较自由的定义神经网络的结构
     以后直接调用Mlp就可以实现relu_4的功能
     调用Koopman 就可以实现基于koopman理论的XXX

时间 23年7月18日
1.更改 根据官网，自己实现mish函数，
2.把选择激活函数的部分抽象成一个函数
"""
#
def mish(x):
    softplus = tf.math.log(tf.math.exp(x) + tf.ones(shape=x.shape))
    return x * tf.keras.activations.tanh(softplus)

def activation_fun(act_fun):
    if act_fun == 'relu':
        function = tf.nn.relu
    elif act_fun == "sigmoid":
        function = tf.nn.sigmoid
    elif act_fun == "elu":
        function = tf.nn.elu
    elif act_fun == "gelu":
        function = tf.nn.gelu
    elif act_fun == "crelu":
        function = tf.nn.crelu
    elif act_fun == "mish":
        # 因为这个版本没有 mish 自己实现一下
        function = mish
    elif act_fun == "tanh":
        function = tf.keras.activations.tanh
    else:
        print('the activate fun is wrong')
        print('act_fun')
        function = 0 
    return function
# 定义一个简单的神经网络  单隐层 
# 可以把这个mlp 以及后面涉及的其他网络 还有数据读取 分别放到其他py文件中
# 后续可以把其他网络结构也放在这里
class Mlp(ks.Model):
    def __init__(self, mlp_list, act_fun='relu', name='mlp', w_init=tf.random_normal_initializer(stddev=0.05),
                b_init=tf.zeros_initializer()):
        super(Mlp, self).__init__()
        self.mlp_list = mlp_list
        self.net_len = len(mlp_list)
        # self.act_fun = act_fun
        self.name_ = name
        self.w_dict = dict()
        self.b_dict = dict()

        # 选择激活函数 根据字符串
        self.act_fun = activation_fun(act_fun) 

        # net_len = int(len(mlp_list))
        for i in range(self.net_len - 1):
            self.w_dict[self.name_ + '_w_%d' % (i + 1)] = tf.Variable(
                initial_value=w_init(shape=(self.mlp_list[i], self.mlp_list[i + 1]),
                                     dtype='float32', ),
                trainable=True,
                name=self.name_ + '_w_%d' % (i + 1))
            self.b_dict[self.name_ + '_b_%d' % (i + 1)] = tf.Variable(
                initial_value=b_init(shape=[self.mlp_list[i + 1], ],
                                     dtype='float32'),
                trainable=True,
                name=self.name_ + '_b_%d' % (i + 1))
        print(mlp_list)



    def call(self, inputs):
        x = inputs

        for i in range(self.net_len - 2):  # 处理最后一层不用relu
            x = self.act_fun(
                tf.matmul(x, self.w_dict[self.name_ + '_w_%d' % (i + 1)]) + self.b_dict[self.name_ + '_b_%d' % (i + 1)])

        y = tf.matmul(x, self.w_dict[self.name_ + '_w_%d' % (self.net_len - 1)]) + self.b_dict[self.name_ + '_b_%d' % (self.net_len - 1)]
        return y 


# 尝试一下用继承的思路做  包括输入输出一共四层
# https://www.runoob.com/w3cnote/python-extends-init.html
class MLPLayers(ks.layers.Layer):
    def __init__(self, mlp_list, act_fun='relu', name='mlp', w_init=tf.random_normal_initializer(stddev=0.05),
                    b_init=tf.zeros_initializer()):
        super(MLPLayers, self).__init__()
        self.mlp_list = mlp_list
        self.net_len = len(mlp_list)
        # self.act_fun = act_fun
        self.name_ = name
        self.w_dict = dict()
        self.b_dict = dict()
        
        self.act_fun = activation_fun(act_fun) 
        
        # net_len = int(len(mlp_list))
        for i in range(self.net_len - 1):
            self.w_dict[self.name_ + '_w_%d' % (i + 1)] = tf.Variable(
                initial_value=w_init(shape=(self.mlp_list[i], self.mlp_list[i + 1]),
                                     dtype='float32', ),
                trainable=True,
                name=self.name_ + '_w_%d' % (i + 1))
            self.b_dict[self.name_ + '_b_%d' % (i + 1)] = tf.Variable(
                initial_value=b_init(shape=[self.mlp_list[i + 1], ],
                                     dtype='float32'),
                trainable=True,
                name=self.name_ + '_b_%d' % (i + 1))
        
        
    def call(self, inputs):
        x = inputs

        for i in range(self.net_len - 2):  # 处理最后一层不用relu
            x = self.act_fun(
                tf.matmul(x, self.w_dict[self.name_ + '_w_%d' % (i + 1)]) + self.b_dict[self.name_ + '_b_%d' % (i + 1)])

        y = tf.matmul(x, self.w_dict[self.name_ + '_w_%d' % (self.net_len - 1)]) + self.b_dict[self.name_ + '_b_%d' % (self.net_len - 1)]
        return y 


# 定义一个矩阵，在koopman线性空间 感觉还是矩阵方便  因为后面可能会用到这个矩阵进行乘法和除法
# 如果只有一种矩阵运算的话 可以用类的方法封装起来
def k_matrix(shape):
    k_init = tf.random_normal_initializer(stddev=0.05)
    return tf.Variable(initial_value=k_init(shape=(shape, shape), dtype='float32'), trainable=True)


class Koopman(ks.Model, abc.ABC):
    """
    sp : 内部循环的次数 
    需要注意的是网络输出是什么，这影响这损失函数应该如何写
    """
    def __init__(self, encoder_list, decoder_list, active_f='relu', sp=1):
        super(Koopman, self).__init__()

        self.encoder_list = encoder_list
        self.decoder_list = decoder_list
        self.active_f = active_f
        self.sp = sp

        self.encoder = MLPLayers(self.encoder_list, act_fun=self.active_f, name='encoder')
        self.k_mat = k_matrix(self.encoder_list[-1])
        self.decoder = MLPLayers(self.decoder_list, act_fun=self.active_f, name='decoder')

    def call(self, inputs, training=None, mask=None):
        # 第一个y_b是没有乘法k_mat 的 可以用来构建复现误差 
        # 之后的 y_b_list 中的值是可以用来 计算线性化误差 和预报误差的 
        # y_all_state 是把说有输入都转换到隐藏空间
        y_all_state = self.encoder(inputs)
        y_b = self.encoder(inputs[:, 0, :])
        y_b_list = [y_b]
        if self.sp > 1:
            for i in range(self.sp - 1):
                y_b = tf.matmul(y_b, self.k_mat)
                y_b_list.append(y_b)

            x_b_list = self.decoder(y_b_list)
            return {'x_b_list':x_b_list, 'y_all_state':y_all_state}
        elif self.sp == 1:
            y_b = tf.matmul(y_b, self.k_mat)
            x_b = self.decoder(y_b)
            return x_b
        else:
            assert sp == 1
            return 0

class Encoder_decoder(ks.Model, abc.ABC):
    """
    # 时间 7月24
    # 简介 在Koopman基础上更改的编码器解码器结构，带词嵌入。
    sp : 这里的sp是模型最多管几个时刻数据的意思
    需要注意的是网络输出是什么，这影响这损失函数应该如何写
    """
    def __init__(self, encoder_list, decoder_list, embdding_list, active_f='relu', sp=0):
        super(Encoder_decoder, self).__init__()

        self.encoder_list = encoder_list
        self.decoder_list = decoder_list
        self.embdding_list = embdding_list
        self.active_f = active_f
        self.sp = sp
        
        self.encoder = MLPLayers(self.encoder_list, act_fun=self.active_f, name='encoder')
        self.embdding = MLPLayers(self.embdding_list, name='embdding')
        self.decoder = MLPLayers(self.decoder_list, act_fun=self.active_f, name='decoder')

    def call(self, inputs, t, training=None, mask=None, test=False):
        # 这里的t是 一个随机的sp 应该来说 inputs 为给encoder
        # 然后 t 的输入之前 要做一个变化 把它的shape 变成 none, 1
        encoder_output = self.encoder(inputs)
        t_embdding = self.embdding(t)
        
        # 两个合成 
        decoder_intput = tf.raw_ops.Concat(values=[encoder_output, t_embdding], concat_dim=1, )
        decoder_output = self.decoder(decoder_intput)
        
        if test:
            print('inputs encoder_output t_embdding decoder_intput decoder_output')
            print(inputs.shape,encoder_output.shape, t_embdding.shape, decoder_intput.shape, decoder_output.shape)
        return decoder_output