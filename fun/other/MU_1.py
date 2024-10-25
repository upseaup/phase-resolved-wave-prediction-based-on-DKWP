
# -*-coding:utf-8 -*-

"""
# Project    : PB
# File       : test.py
# Time       ：2023/8/13 17:12
# Author     ：pang
# version    ： v1.1
# Description：这个是最开始的版本 因为没有加mu 所以在（0，0）这个点的数据幅值很高。
               这里mu! = 0 并且去掉其他选择区域的地方。
"""
import numpy as np
import matplotlib.pyplot as plt
import time
# import fun.dft_fortran_v3 as dft
# import fun.lwt as lwt
# from fun.jonswap_matlab import jonswap_matlab as jon
import jonswap_matlab as jon



# ----------------------------
# 定义超参  todo 超参数的形式需要确认一下。这样写肯定不对，最好区分一下，比如哪些是可以改的（放到函数参数里的 hs tp gamma这类的），
#  todo 哪些一般不会改（t 浪高仪个数啊）
# 频谱分割的份数。
M = 300  # todo 300可以吗
G = 9.806

NX = 300
# 考虑几个横坐标的点 x 相当与浪高仪的个数  最少要有两个 A B
dx = 10  # 浪高仪间距0.1米  要记得看其他论文里数据的样式 todo
# 时间间隔为 dt
dt = 0.5   # 这里要看其他论文dt大概是多大有个对比参考吧
t_end = 10 * 60  # 每条轨迹长度 单位s
# 海况
Hs = 1.88
Tp = 8.8
gamma = 3.3
# todo 下面两个的范围需要注意 虽然现在这个谱中 下面两个是满足要求的 但是要是更换H ts 的话
# 可能也是需要更换的 最好在程序里有一个判断标准
w_l = 0.2
w_h = round(3 * 2 * np.pi / Tp, 2)  # 三倍谱峰频率，保留两位小数

N_TEST = 1
N_P_TRAIN = 600
print('海浪共有多少个离散的采样点', t_end/dt)

if __name__ == "__main__":
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # todo 这个地方要考虑一下，如果以后Tp变化了，如何处理 Omega 的范围呢？？ 一般考虑3倍峰值频率
    w = np.arange(0.1, w_h+1, 0.001)
    S = jon.jonswap_matlab(w, Tp=Tp, Hs=Hs, Gamma=gamma, FiguringSpectrum=2)

    s_sum = np.sum(S)

    s_sum_wh = np.sum(S, where=(w < w_h))
    s_sum_wl = np.sum(S, where=(w > w_l))
    # 利用差 把 w_l 到 w_h 的 s_sum 范围选出来
    p = (s_sum_wh - s_sum + s_sum_wl) / s_sum

    # 判断谱是否满足要求
    if p > 0.99:
        print('w的范围不需要重新挑选，当前p为:', p)
    else:
        print('w的范围需要重新挑选，当前p为:', p)
        # assert 1 == 0

    # 使用等频率取法 P224  todo 这里的问题和 上面生成谱的时候的w的范围一样
    dw = (w_h - w_l) / M

    # 这个地方是得到用于求 a 的 wi
    w_range = np.linspace(w_l, w_h, M)
    # 得到 w_i_hot 用的是0-1均匀分布 而不是正态分布
    w_i_hot = w_range + np.random.rand(w_range.shape[0]) * dw

    s_hot = jon.jonswap_matlab(w_i_hot, Tp=Tp, Hs=Hs, Gamma=gamma, FiguringSpectrum=2)
    # # 看生成数据用的功率谱
    plt.plot(w, S)
    plt.plot(w_i_hot, s_hot)
    plt.legend(['S_hot', 'S ground value'])
    plt.show()

    # 求幅值
    a = np.sqrt(2 * dw * s_hot)

    # 求初始相位 mu  这里有个不是问题的问题 就是 用np的时候没有说明数据的类型是32位还是64 不过应该问题不大 todo
    # 这里用的是弧度 等下 求波的时候要注意 也是弧度
    wave_list = []

    # 这里可以设成一个函数 通过赋值需要的演化时间 t_need 来 得到新的
    t = np.arange(0, t_end, dt)
    wt = w_i_hot[:, np.newaxis] * t[:, np.newaxis].T
    k = np.square(w_i_hot) / G
    # 假设浪高仪 间距为 dx = 0.1
    x = np.arange(0, NX * dx, dx)
    # 得到kx
    kx = k[:, np.newaxis] * x[:, np.newaxis].T

    # # 先看一个海况
    for i in range(1):
        mu = np.random.rand(w_range.shape[0]) * 2 * np.pi
        for j in range(1):
            # todo 这里要把维度改下 第二个维度要是时间
            # wave_part = a[:, np.newaxis] * np.cos(wt - kx[:, j:(j + 1)] + mu[:, np.newaxis])
            wave_part = a[:, np.newaxis] * np.cos(kx[:, j:(j + 1)] - wt + mu[:, np.newaxis])
            wave = np.sum(wave_part, axis=0)
        plt.plot(np.arange(0, wave.shape[0], 1), wave)
        plt.title('wave ')
        plt.show()

    # # 看生成的数据反求的功率谱和生成的对比
    # w_dft, p = dft.wave_spectrum_fft(wave, dt)
    # p_filter = dft.mean_filter(p, kernel_size=30)
    #
    # plt.plot(w_dft[:-1], p_filter[0:])
    # plt.plot(w, S)
    # plt.legend(['Specturm by fft', 'Ground Truth'])
    # plt.show()

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 生成一个范围的海浪情况，时空区间的 时间长一点 因为可能用得到 在预报的时候
    t = np.arange(0, t_end, dt)
    wt = w_i_hot[:, np.newaxis] * t[:, np.newaxis].T
    k = np.square(w_i_hot) / G
    # 假设浪高仪 间距为 dx = 0.1
    x = np.arange(0, NX * dx, dx)
    # 得到kx
    kx = k[:, np.newaxis] * x[:, np.newaxis].T

    wave_data = np.zeros(shape=[N_TEST, int(t_end / dt), NX], dtype=np.float32)

    # todo 这里要把维度改下 第二个维度要是时间
    for i in range(N_TEST):
        mu = np.random.rand(w_range.shape[0]) * 2 * np.pi
        mu = mu[:, np.newaxis]
        # mu = np.zeros(w_range.shape[0])
        # mu = mu[:, np.newaxis]
        # mu = np.tile(mu, [1, wt.shape[1]])
        # NX = 500 dx = 10
        for j in range(NX):
            # wave_part = a[:, np.newaxis] * np.cos(wt - kx[:, j:(j + 1)] + mu[:, np.newaxis])
            test = kx[:, j:(j + 1)]
            wave_part = a[:, np.newaxis] * np.cos(kx[:, j:(j + 1)] - wt + mu)
            wave = np.sum(wave_part, axis=0)
            wave_data[i, :, j] = wave
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 先用for 循环的方法计算
    t = np.arange(0, t_end, dt)
    cl = G / (2 * w_h)
    ch = G / (2 * w_l)
    loss = np.zeros(shape=wave_data.shape)
    wave_predict = np.zeros(shape=wave_data.shape)
    wave_predict_list = []
    loss_list = []

    # 先对a 复制一下
    a_ele = a[:]
    print(a_ele.shape, x.shape, t.shape)
    for i in x:
        for j in t:
            a_ele = a[:]
            cone = i / (j + 0.001)
            ctwo = i / (j + 0.001 - N_P_TRAIN * dt)
            elevation = np.sum(a_ele * np.cos(k * i - w_i_hot * j + mu))
            wave_predict_list.append(elevation)
            loss_list.append(np.square(elevation - wave_data[:, int(j / dt), int(i / dx)]))
            # print(elevation, elevation.shape, type(elevation))
            wave_predict[:, int(j / dt), int(i / dx)] = np.ones(shape=[1, 1])*elevation
            loss[0, int(j / dt), int(i / dx)] = np.square(elevation - wave_data[0, int(j / dt), int(i / dx)])
            # pass.
    plt.show()
    plt.contourf(loss[0, :, :].T, levels=[0, 0.01, 0.1, 0.2, 0.3, 0.5, 1], cmap="viridis_r", alpha=0.8)
    plt.colorbar(shrink=0.8, label='NDRMSE')
    plt.show()
    pass