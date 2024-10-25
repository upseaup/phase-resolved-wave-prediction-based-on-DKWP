# -*-coding:utf-8 -*-
import tensorflow as tf
from tensorflow import keras as ks
import abc
import numpy as np

"""
# Project    : fun
# File       : net_koopman.py
# Time       ：2023/10/12 13:10
# Author     ：pang
# version    ： 1.0
# Description：实现非受力系统的基于深度学习的koopman算符的基础工作。比如定义编码器 解码器 koopman空间 以及相关损失函数等。
"""
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
时间 23年8月5日
# 更新 更新w的初始化方式，使得变量初始化的时候可以接受接受seed变量 
#      并改后缀为_Copy2

时间 23年10月12日
1.在之前net_Copy2 版本上 整理得到 net_koopman
2.MLP中w权重的初始化方式默认是均匀分布改为是GU, 但是koopman矩阵的分布方式默认是均匀分布。
"""


def mish(x):
    """
    定义mish 激活函数
    """
    softplus = tf.math.log(tf.math.exp(x) + tf.ones(shape=x.shape))
    return x * tf.keras.activations.tanh(softplus)


def activation_fun(act_fun):
    """
    通过字符串 实现常用激活函数的选择。
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
        # 因为这个版本没有 mish 自己实现一下
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
    生成一个MLP 可以单独使用，也可以做为大模型的一部分。
    """

    # 时间 2023年8月5日
    # 更新 更新w的初始化方式，接受seed变量
    def __init__(self, mlp_list, act_fun='relu', name='mlp', w_init=tf.keras.initializers.GlorotUniform,
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

        for i in range(self.net_len - 2):  # 处理最后一层不用relu
            x = self.act_fun(
                tf.matmul(x, self.w_dict[self.name_ + '_w_%d' % (i + 1)]) + self.b_dict[self.name_ + '_b_%d' % (i + 1)])

        y = tf.matmul(x, self.w_dict[self.name_ + '_w_%d' % (self.net_len - 1)]) + self.b_dict[
            self.name_ + '_b_%d' % (self.net_len - 1)]
        return y


# 下面这些都是 koopman相关的
def k_matrix(shape, name='k_matrix'):
    """
    初始化一个shape=shape的方阵
    """
    k_init = tf.random_normal_initializer(stddev=0.05)
    return tf.Variable(initial_value=k_init(shape=(shape, shape), dtype='float32'), trainable=True, name=name)


def koopman_next(data, mat):
    """
    计算在koopman空间下一时刻下的状态
    """
    return tf.matmul(data, mat)


class Koopman(ks.Model, abc.ABC):
    """
    定义一个koopman模型 注意这里只生成一个koopman矩阵
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
# 定义数据变量 u_mat_b 代表 控制量乘过矩阵b 后的结果 (如果有)
#            mult_shift 代表 在高维空间演化多个时间步 
# 需要注意的是 对于 mult_shift 相当于一个点生成一个轨迹（一个k时刻的点演化n个时间步，形成一个轨迹）
#            对于 one_shift 相当于一个轨迹上的点都 都演化一个时间步，得到一个新的轨迹，
#            两者在高维空间演化没有误差时 是等价的，
"""


def koopman_model(x, encoder, decoder, koopman_op, sp):
    """
    这里是把整个流程封装到一个函数里，那么需要把其他用到的model 编码器解码器这种的也传递进来
    输入   数据 三个模型 模型想要考虑的时间步长
    返回值 是 整个模型对于输入x 的输出y
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
    损失函数 就是通过损失函数才梯度下降改变权值的
    时间：7月24日
    更改内容：将输出进行打包
            本函数和 loss_fun 函数 区别，将输入利用字典打包 同时将输出也利用字典打包
            todo :主函数对应的还没改 主函数用的还是调bug的版本
    :param x: 数据本来的状态
    :param output_dict: 模型输出的合集 具体如下
           x_recon: 数据只通过编解码器后的状态
           x_recon_one_shift: 是 一条轨迹移动一个时间步后的结果
           x_recon_mult_shift:  上一个变量经过解码器得到 对0时刻起预测几步后的结果
           z: koopman 线性空间的状态
           z_one_shift: 一条轨迹在koopman空间移动一个时间步后的结果
           z_mult_shift: y 空间 各时间步下预测（对0时刻预测几步）的结果
    :param sp: 求loss 会用到的时间步
    :param L_RECON: 复现误差系数
    :param L_LIN: 线性化误差系数
    :param L_PRED: 预报误差系数
    :param Linf_lam: 正则化误差系数
    :param L2_lam: L2误差系数，这里没有用上

    :return: 各种损失的和
    需要注意的是： 本loss_fun 因为对中间 z_mult_shift 进行了 stack 操作 所以 求loss的时候有些量是不一样的
                 以及这个网络精度 是 float32的
    """
    # 这里的shape 和 nature 中不一样 (None, 51, 2)
    loss_recon = tf.reduce_mean(
        tf.square(x[:, 0, :] - output_dict['x_recon'][:, 0, :]))  # 这里的shape 和 nature 中不一样 (None, 51, 2)

    # todo 考虑后续开多进程处理损失函数。loos_pre 是在gpu 还是在cpu上运行的
    # pre 和 lin 的简化版误差有助 收敛 加快训练速度
    loss_pre = tf.zeros([1, 1], dtype=np.float32)
    for i in range(0, sp):
        # 两种不同的的 预测误差 一个是3dim 一个是 2dim
        # loss_pre = loss_pre + tf.reduce_mean(tf.square(x[:, i + 1, :] - output_dict['x_recon_mult_shift'][:, i, :]))
        # loss_pre = loss_pre + tf.reduce_mean(tf.square(x[:, i + 1, :] - output_dict['x_recon_mult_shift_list'][i]))
        # 简化版 pre 误差
        loss_pre = loss_pre + tf.reduce_mean(tf.square(x[:, i + 1, :] - output_dict['x_recon_one_shift'][:, i, :]))
    loss_pre = loss_pre / sp

    # lin loss
    loss_lin = tf.zeros([1, 1], dtype=np.float32)

    for i in range(0, sp):
        # 正常的 lin loss
        loss_lin = loss_lin + \
                   tf.reduce_mean(tf.square(
                       output_dict['z'][:, i + 1, :] - output_dict['z_mult_shift_list'][i]))
        # 简化版线性化误差
        # loss_lin = loss_lin + tf.reduce_mean(
        #     tf.square(output_dict['z'][:, i + 1, :] - output_dict['z_one_shift'][:, i, :]))

    loss_lin = loss_lin / sp

    # 范数的惩罚函数
    Linf1_den = tf.cast(1.0, dtype=tf.float32)
    Linf2_den = tf.cast(1.0, dtype=tf.float32)

    Linf1_penalty = tf.truediv(
        tf.norm(tf.norm(output_dict['x_recon'][:, 0, :] - tf.squeeze(x[:, 0, :]), axis=1, ord=np.inf), ord=np.inf),
        Linf1_den)
    Linf2_penalty = tf.truediv(
        tf.norm(tf.norm(output_dict['x_recon'][:, 1, :] - tf.squeeze(x[:, 1, :]), axis=1, ord=np.inf), ord=np.inf),
        Linf2_den)
    loss_inf = Linf_lam * (Linf1_penalty + Linf2_penalty)

    # 损失函数求和
    loss_all = L_RECON * loss_recon + L_PRED * loss_pre + loss_inf + L_LIN * loss_lin

    loss_all = tf.squeeze(loss_all)
    loss_recon = tf.squeeze(loss_recon)
    loss_pre = tf.squeeze(loss_pre)
    loss_lin = tf.squeeze(loss_lin)
    loss_inf = tf.squeeze(loss_inf)
    loss_dict = {'loss_all': loss_all, 'loss_recon': loss_recon, 'loss_pre': loss_pre, 'loss_lin': loss_lin,
                 'loss_inf': loss_inf}

    return loss_dict


# ------ 其他辅助函数
def tensor_board_summary_writer(summary_writer, loss_dict, epoch):
    """
    定义tensor board 写入函数
    """
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss_dict['loss_all'], step=epoch)
        tf.summary.scalar('loss_recon', loss_dict['loss_recon'], step=epoch)
        tf.summary.scalar('loss_lin', loss_dict['loss_lin'], step=epoch)
        tf.summary.scalar('loss_pre', loss_dict['loss_pre'], step=epoch)


def space_fun(sn, space, space_num_list, test=False):
    """
    这个是一个空间函数 主要看 整个工程中需要遍历几种变量 并打印出来。
    """
    # 定义space_num_list 的长度 用于循环。space_num_list len
    spnll = len(space_num_list)
    # 定义一个list 用于返回数据
    return_list = [space[0][sn%space_num_list[0]]]
    for j in range(spnll-1):
        return_list.append(space[j+1][sn//space_num_list[j]%len(space[j+1])])
    # p_train = space[0][sn%space_num_list[0]]       # 第一个对第一个space_num取余
    # output_shape = space[1][sn//space_num_list[0]%len(space[1])] # 第二个对第一个space_num 向下取整
    # xp = space[2][sn//space_num_list[1]%len(space[2])]           # 第三个变量对第二个space_num 向下取整
    # split = space[3][sn//space_num_list[2]%len(space[3])]
    if test==True:
        print('当前部分超参数和sn', return_list, sn)
    return return_list


# 下面这些是当时想实现对角阵的时候写的，删了怪可惜的。放在最后面吧。
# def diag_matrix(shape=None):
#     # 做判断是否输入正确的数据了 理论上应该是一个int
#     if shape == None or type(shape) != int:
#         assert 1 == 0
#         return 0
#     # 这里要实现随机分布 应该用np 的random 函数实现
#     # 这里应该考虑 把随机种子保存起来
#     np.random.seed(np.random.randint(100))
#     # 初始化一个比较小的值通过除以30 要不要考虑是正数啊
#     element_list = np.random.randn(shape) / 30
#
#     # element_list = np.ones(shape=shape)
#     diag = np.diag(element_list)
#     # 转化为 tf 且可训练
#     diag = tf.Variable(diag, trainable=True, dtype='float32')
#     return diag
#
#
# def diag_row(shape=None):
#     """
#     主要为了解决 diag_matrix 中得到对角阵后 其他位置零元素在训练过程中会被更新的问题
#     自己定义一个list 里面是对应对角阵的特征值mu
#     这里要解决变量 trainable_variables 的问题 初步的想法是 整个类 父类是 layers 这种
#     """
#     if shape == None or type(shape) != int:
#         assert 1 == 0
#         return 0
#     # 得到一个有 shape 个元素的list 元素的是tensor 类型 shape 是 1，1 或没有shape
#     diag_list = []
#     for i in range(shape):
#         mu = np.random.randn(1).reshape(1, 1) / 30
#         diag_list.append(tf.Variable(mu, trainable=True, dtype='float32', name='mu_' + str(i)))
#     return diag_list
#
#
# def diag_matrix_mask(shape=None):
#     """
#     这里要实现一个不可训练的对角阵 shape 决定 方阵的维数 和diag_row 配合使用 实现类似矩阵乘法的作用
#
#     """
#     if shape == None or type(shape) != int:
#         assert 1 == 0
#         return 0
#
#     element_list = np.ones(shape=shape)
#     diag = np.diag(element_list)
#     # 转化为 tf 且可不训练
#     # diag = tf.Variable(diag,  trainable=False, dtype='float32')
#     diag = tf.constant(diag, dtype='float32')
#
#     return diag
#
#
# def koopman_next_v2(data, mat):
#     """
#     这里要配合 diag_row 实现koopman空间中下一步的 data * mat mat为矩阵， 这里是用一个list实现的。
#     这里为了兼容 koopman_next mat 是一个list 第一个元素是 diag_list 第二个元素 是 diag_matrix_mask
#     """
#     diag_list = mat[0]
#     diag_matrix_mask_ = mat[1]
#     diag_matrix_len = len(diag_list)
#     # data shape n, sp, z-dim
#     z_next_list = []
#     for i in range(diag_matrix_len):
#         # 提取对应维度 并降维
#         z_next_part = tf.reduce_max(tf.matmul(data, diag_matrix_mask_[:, i:i + 1]), axis=-1)
#         # 乘特征值 并升维度
#         z_next_part = tf.expand_dims(z_next_part, -1)
#         z_next_part = tf.matmul(z_next_part, diag_list[i])
#         z_next_list.append(z_next_part)
#
#     return tf.concat(z_next_list, axis=-1)
