import tensorflow as tf
from tensorflow import keras as ks
"""
时间 23年3月5日
内容 定义网络结构文件
     后续用koopman可能需要更复杂的网络结构
作者 冯胖子 
"""
# 定义一个简单的神经网络  单隐层 
# 可以把这个mlp 以及后面涉及的其他网络 还有数据读取 分别放到其他py文件中
# 后续可以把其他网络结构也放在这里
class Mlp(ks.Model):
    # 时间 23年4月9日
    # 增加 增加内容 增加激活函数，默认为sigmoid 
    def __init__(self, inputs_shape, h_state, output_shape, active_fun='sigmoid'):
        super(Mlp, self).__init__()
        w_init = tf.random_normal_initializer(stddev=0.05)
        b_init = tf.zeros_initializer()
        
        # h_state = 
        self.ww1 = tf.Variable(initial_value=w_init(shape=(inputs_shape, h_state), dtype='float32'), trainable=True)
        self.wb1 = tf.Variable(initial_value=b_init(shape=[h_state, ], dtype='float32'), trainable=True)
        self.ww2 = tf.Variable(initial_value=w_init(shape=(h_state, output_shape), dtype='float32'), trainable=True)
        self.wb2 = tf.Variable(initial_value=b_init(shape=[output_shape, ], dtype='float32'), trainable=True)
        
        if active_fun == 'sigmoid':
            self.act_fun = tf.nn.sigmoid
        elif active_fun == 'relu':
            self.act_fun = tf.nn.relu
        else:
            assert 1==0

    def call(self, inputs, training=None, mask=None):
        # x = inputs

        xx = self.act_fun(tf.matmul(inputs, self.ww1) + self.wb1)  # 先用sigmoid 看下 如果不行可以用relu todo
        y = tf.matmul(xx, self.ww2) + self.wb2
        return y
    
class MLP_4(ks.Model):
    # 增加对 active fun的支持 
    def __init__(self, inputs_shape, h_state, output_shape, active_fun='sigmoid'):
        self.inputs_shape = inputs_shape
        self.h_state = h_state
        self.output_shape = output_shape
        self.active_fun = active_fun
        self.model = tf.keras.models.Sequential()
        
        self.model.add(tf.keras.Input(shape=(self.inputs_shape,)))
        
        self.model.add(tf.keras.layers.Dense(self.h_state, activation=self.active_fun))
        self.model.add(tf.keras.layers.Dense(self.h_state, activation=self.active_fun))

        self.model.add(tf.keras.layers.Dense(self.output_shape))
        

        
# 这两个是是主要给ppmv_v4系列设计的，这样主要可以对比relu_4和sigmoid
def mlp_relu4(inputs_shape, h_state, output_shape,):
    mlp = tf.keras.models.Sequential()
        
    mlp.add(tf.keras.Input(shape=(inputs_shape,)))
    mlp.add(tf.keras.layers.Dense(h_state, activation='relu'))
    mlp.add(tf.keras.layers.Dense(h_state, activation='relu'))
    mlp.add(tf.keras.layers.Dense(output_shape))
    return mlp


def mlp_sigmoid3(inputs_shape, h_state, output_shape,):
    mlp = tf.keras.models.Sequential()
        
    mlp.add(tf.keras.Input(shape=(inputs_shape,)))
    mlp.add(tf.keras.layers.Dense(int(h_state), activation='sigmoid'))
    # mlp.add(tf.keras.layers.Dense(int(h_state * 3), activation='relu'))
    mlp.add(tf.keras.layers.Dense(output_shape))
    return mlp

