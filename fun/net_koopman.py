# -*-coding:utf-8 -*-
import tensorflow as tf
from tensorflow import keras as ks
import abc
import numpy as np

"""
# Project    : fun
# File       : net_koopman.py
# Time       ：2024/10/17 13:10
# Author     ：pang
# version    ： 2.0

"""


def mish(x):
    """
    define mish activation function. In tensorflow == 2.10, there has not the mish
    activation function, so we define mish here.
    """
    softplus = tf.math.log(tf.math.exp(x) + tf.ones(shape=x.shape))
    return x * tf.keras.activations.tanh(softplus)


def activation_fun(act_fun):
    """
    """
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
        function = mish
    elif act_fun == "tanh":
        function = tf.keras.activations.tanh
    else:
        print('the activate fun is wrong')
        print('act_fun')
        function = 0
    return function


class Mlp(ks.Model):
    """
    Generate an MLP
    """

    def __init__(self, mlp_list, act_fun='relu', name='mlp', w_init=tf.random_normal_initializer,
                 b_init=tf.zeros_initializer()):
        super(Mlp, self).__init__()
        self.mlp_list = mlp_list
        self.net_len = len(mlp_list)
        # self.act_fun = act_fun
        self.name_ = name
        self.w_dict = dict()
        self.b_dict = dict()

        # Select activation function based on the string act_fun
        self.act_fun = activation_fun(act_fun)

        for i in range(self.net_len - 1):
            self.w_dict[self.name_ + '_w_%d' % (i + 1)] = tf.Variable(
                initial_value=w_init(seed=i)(shape=(self.mlp_list[i], self.mlp_list[i + 1]),
                                             dtype='float32', ),
                trainable=True,
                name=self.name_ + '_w_%d' % (i + 1))
            self.b_dict[self.name_ + '_b_%d' % (i + 1)] = tf.Variable(
                initial_value=b_init(shape=[self.mlp_list[i + 1], ],
                                     dtype='float32'),
                trainable=True,
                name=self.name_ + '_b_%d' % (i + 1))
        print(name, mlp_list)

    def call(self, inputs):
        x = inputs

        for i in range(self.net_len - 2):  # Mind the last layer doesn't require relu
            x = self.act_fun(
                tf.matmul(x, self.w_dict[self.name_ + '_w_%d' % (i + 1)]) + self.b_dict[self.name_ + '_b_%d' % (i + 1)])

        y = tf.matmul(x, self.w_dict[self.name_ + '_w_%d' % (self.net_len - 1)]) + self.b_dict[
            self.name_ + '_b_%d' % (self.net_len - 1)]
        return y


# something about the Koopman
def k_matrix(shape, name='k_matrix'):
    """
    Initialize a square matrix with (shape,shape) as the Koopman operator
    """
    k_init = tf.random_normal_initializer(stddev=0.05)
    return tf.Variable(initial_value=k_init(shape=(shape, shape), dtype='float32'), trainable=True, name=name)


def koopman_next(data, mat):
    """
    Calculate the state at the next moment in Koopman space
    """
    return tf.matmul(data, mat)


class Koopman(ks.Model, abc.ABC):
    """
    Define a Koopman model and note that only one Koopman operator is generated here
    """

    def __init__(self, shape, name='koopman_matrix', ):
        super(Koopman, self).__init__()

        self.shape = shape
        self.k_matrix = k_matrix(self.shape, name=name)

    def call(self, inputs):
        return koopman_next(inputs, self.k_matrix)


"""
# 定义数据后缀 x 代表非线性系统在现实世界（未升维）的状态
#            x_recon 代表解码器解码之后的现实世界的状态
#            z 代表x在高维空间的状态
#            one_shift 代表 在高维空间演化一个时间步
#            start  代表 起始点, 然后迭代到轨迹最后 一般有z_start 
#            mult_shift 代表 在高维空间演化多个时间步 
# 需要注意的是 对于 mult_shift 相当于一个点生成一个轨迹（一个k时刻的点演化n个时间步，形成一个轨迹）
#            对于 one_shift 相当于一个轨迹上的点都 都演化一个时间步，得到一个新的轨迹，
#            两者在高维空间演化没有误差时应该是等价的，
"""
"""
x: wave elevation
z: the state in the Koopman space
_recon: the estimated x obtained by the decoder, 
_one_shift: donate the predicted at sp=1
_mult_shift: donate the predicted at sp, in this work, sp=15 in training process.
----------
It should be noted as when there is no error in the evolution of high-dimensional space, 
the _one_shift and _mult_shift should be equivalent.
"""


def koopman_model(x, encoder, decoder, koopman_op, sp):
    """
    input
    x: wave elevation data with shape = (N, T, N_X), where N is the size of the set or bath_size,
       T is the number of the time step, N_X is the number of the point in D_AB. In this work, N_X = 360
    encoder: the MLP that transform the wave elevation x (y in paper) to linear state z
    decoder: the MLP that transform the w linear state z to ave elevation x (y in paper)
    koopman_op: the Koopman operator, is a trainable matrix in tensorflow
    sp:, step, calculate the z or x predicted by the model of future sp steps
    output
    pre_dict:, a dictionary with the predicted state z, x and so on.
        x_recon: the reconstruction wave elevation by decoder and linear state z, having the same shape with x
        x_recon_one_shift: the predicted wave elevation at sp=1, having the same shape with x
        x_recon_mult_shift:  the predicted wave elevation at sp (sp!=1) by x[N, 0, N_X], shape = [N, sp, N_X]
        z: koopman linear state obtained by decoder, more specifically, z=h(x), where h is decoder
           ,the shape of z is [N, T, z_dim], z_dim is the Dimension of K
        z_one_shift: obtained by Koopman operator * z,where the shape of z_one_shift is [N, T, z_dim]
        z_mult_shift: obtained by Koopman operator **sp * z[N, 0, z_dim], where the shape of z_mult_shift is
        [N, sp, z_dim]
    """
    pre_dict = {}

    z = encoder(x)
    x_recon = decoder(z)

    z_one_shift = koopman_op(z)
    x_recon_one_shift = decoder(z_one_shift)

    z_mult_shift_list = []
    z_start = z[:, 0, :]
    for j in range(sp):
        z_start = koopman_op(z_start)
        z_mult_shift_list.append(z_start)
    z_mult_shift = tf.stack(z_mult_shift_list, axis=1)
    x_recon_mult_shift = decoder(z_mult_shift)

    pre_dict['z'] = z
    pre_dict['x_recon'] = x_recon
    pre_dict['z_one_shift'] = z_one_shift
    pre_dict['x_recon_one_shift'] = x_recon_one_shift
    pre_dict['z_mult_shift'] = z_mult_shift
    pre_dict['x_recon_mult_shift'] = x_recon_mult_shift
    pre_dict['z_mult_shift_list'] = z_mult_shift_list

    return pre_dict


def loss_fun_model(x, output_dict, sp, L_RECON=0.1, L_PRED=0.1, L_LIN=0.1, L2_lam=10 ** (-15), Linf_lam=10 ** (-7)):
    """
    The loss function mainly consists of three parts: L_recon, l_lin, L_pre

    :param x: the wave elevation with shape = [N, T, N_X]
    :param output_dict: a dictionary with the predicted state z, x and so on.
          see the details in function koopman_model()
    :param sp: step, calculate the z or x predicted by the model of future sp steps
    :param L_RECON:  weight about the reconstruction loss
    :param L_LIN: weight about the linearity loss
    :param L_PRED: weight about the prediction loss
    :param Linf_lam: weight about the Linf
    :param L2_lam: weight about the L2

    :return: the sum of the loss

    """
    loss_recon = tf.reduce_mean(
        tf.square(x[:, 0, :] - output_dict['x_recon'][:, 0, :]))

    # It should be noted as when there is no error in the evolution of high-dimensional space,
    # the _one_shift and _mult_shift should be equivalent. So, the simplified loss can accelerate the training process
    loss_pre = tf.zeros([1, 1], dtype=np.float32)
    for i in range(0, sp):
        # standard pre loss
        # loss_pre = loss_pre + tf.reduce_mean(tf.square(x[:, i + 1, :] - output_dict['x_recon_mult_shift'][:, i, :]))
        # Simplified pre loss
        loss_pre = loss_pre + tf.reduce_mean(tf.square(x[:, i + 1, :] - output_dict['x_recon_one_shift'][:, i, :]))
    loss_pre = loss_pre / sp

    # lin loss
    loss_lin = tf.zeros([1, 1], dtype=np.float32)

    for i in range(0, sp):
        # standard lin loss
        loss_lin = loss_lin + \
                   tf.reduce_mean(tf.square(
                       output_dict['z'][:, i + 1, :] - output_dict['z_mult_shift_list'][i]))
        # Simplified version of lin loss
        # loss_lin = loss_lin + tf.reduce_mean(
        #     tf.square(output_dict['z'][:, i + 1, :] - output_dict['z_one_shift'][:, i, :]))

    loss_lin = loss_lin / sp

    # Punishment function of norm
    Linf1_den = tf.cast(1.0, dtype=tf.float32)
    Linf2_den = tf.cast(1.0, dtype=tf.float32)

    Linf1_penalty = tf.truediv(
        tf.norm(tf.norm(output_dict['x_recon'][:, 0, :] - tf.squeeze(x[:, 0, :]), axis=1, ord=np.inf), ord=np.inf),
        Linf1_den)
    Linf2_penalty = tf.truediv(
        tf.norm(tf.norm(output_dict['x_recon'][:, 1, :] - tf.squeeze(x[:, 1, :]), axis=1, ord=np.inf), ord=np.inf),
        Linf2_den)
    loss_inf = Linf_lam * (Linf1_penalty + Linf2_penalty)

    # Sum of loss functions
    loss_all = L_RECON * loss_recon + L_PRED * loss_pre + loss_inf + L_LIN * loss_lin

    loss_all = tf.squeeze(loss_all)
    loss_recon = tf.squeeze(loss_recon)
    loss_pre = tf.squeeze(loss_pre)
    loss_lin = tf.squeeze(loss_lin)
    loss_inf = tf.squeeze(loss_inf)
    loss_dict = {'loss_all': loss_all, 'loss_recon': loss_recon, 'loss_pre': loss_pre, 'loss_lin': loss_lin,
                 'loss_inf': loss_inf}

    return loss_dict


# ------ Other auxiliary functions
def tensor_board_summary_writer(summary_writer, loss_dict, epoch):
    """
    define the tensor board written function
    time: 24/8/2
    """
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss_dict['loss_all'], step=epoch)
        tf.summary.scalar('loss_recon', loss_dict['loss_recon'], step=epoch)
        tf.summary.scalar('loss_lin', loss_dict['loss_lin'], step=epoch)
        tf.summary.scalar('loss_pre', loss_dict['loss_pre'], step=epoch)

