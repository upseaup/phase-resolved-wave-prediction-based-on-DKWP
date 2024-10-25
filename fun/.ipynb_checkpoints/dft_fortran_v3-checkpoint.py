# -*-coding:utf-8 -*-

"""
# Project    : dft-fortran-v2.py
# File       : dft_fortran_v3.py
# Time       ：2023/4/4 17:02
# Author     ：pang
# version    ： 
# Description：
"""
# DFT-main90.f90 改写为python版本
# 时间 2023年3月29
# 作者 冯忠英
# 简介 后续需要补上滤波和转移到tf中 需要边写codee边写论文  归一化因子这里有点小问题
#      fortran 程序中 因为计算过程中中进行了归一化，所以要乘N
#      但是numpy 中 fft 包中默认是没有归一化的，这一点在用tensorflow中也要注意。
#      归一化因子中的 dt 是来自于公式（在随机波浪及其应用公式6.1.8大概）
#      多出来的2可能是用于补偿的。之前11.py中求得补偿因子是0.67. 2的倒数是0.5 相对来说比较接近.
#      这个2可能也是因为计算a的时候有 a = 2 * num /N
#      或者是考虑负频率部分的谱（在2.1小节 有将单侧谱和双侧谱）
#      rfft 和 fft 变化区别在哪里 也需要确认一下。
#      参考网上和官网资料应该是用rfft就行，但是需要注意 rfft的输出长度（n为偶数）是 （n/2）+ 1  fft的输出是 n
# 版本 没有大意外的话，这就是最后的版本（v2）了，或者考虑滤波算法。不过不太需要，或者需要考虑在tf中效率高的滤波算法。
# 后续 后续可能需要考虑最短有多少数据能够进行fft,并保证一定精度。
#      增加滑动滤波算法 使用numpy自带的卷积操作。
# 计算  30001的谱 并考虑缩放因子的影响。果然有点小问题。得重新算。
# 缩尺比 对数据先放大和最后对功率谱放大结果是一样的。
# 时间 2023年5月28
# 更改 jonswap_matlab 的引用 因为放到一个文件夹fun下 所以要增加一个 fun.
#      不用这样，直接把 if __name__ == “__main__” 下面引用的屏蔽掉就行了
#       在把引用屏蔽掉
import numpy as np
import matplotlib.pyplot as plt
# import jonswap_matlab as jon


# 中值平均法
def mean_filter(data, kernel_size):
    # 构建卷积核
    kernel = np.ones(kernel_size) / np.prod(kernel_size)
    # 使用卷积函数进行滤波
    filtered_data = np.convolve(data, kernel, mode='same')
    return filtered_data


# 加一下d 修饰因子
def d_fun(N, dt):
    T = N * dt
    t = np.arange(N) * dt
    a = np.square(0.5 * (1 - np.cos(10 * np.pi * t[0:int(T / 10 + 1)] / T)))
    b = 1 * len(t[int(T / 10 + 1):int(9 * T / 10 + 1)])
    c = np.square(0.5 * (1 + np.cos(10 * np.pi * (t[int(9 * T / 10 + 1):] - 9 * T / 10))))
    d = (np.sum(a) + np.sum(b) + np.sum(c))/N
    print('修饰因子d:', d)
    return d


# 需要注意的是这里默认前向fft是没有归一化的，需要后续处理，具体可以看np官网。
def wave_spectrum_fft(z, dt, ):
    N = len(z)
    sp = np.fft.rfft(z)
    sp = sp[0:-1]
    # 求功率谱 平方  这里是按书上进行缩放的 后续可能需要÷因子？？
    p = (np.abs(sp)**2 * dt * 2) / (2 * np.pi * N)
    # print('归一化因子：', dt/(2 * np.pi * N))
    f = np.fft.rfftfreq(N, dt)
    w_np = f * 2 * np.pi

    return w_np, p


if __name__ == "__main__":
    # 自己的数据  自己旧数据应该用main_copy中的方法进行验证。
    # path = 'D:\\code_project\\jupyter-project\\jaswon_2\\TEST_H1-88_T8-8_gama3-3_w0-35_2-5_NX2_dx500_dt0-5_t_end3600.npy'
    # data = np.load(path)
    # dt = 0.5
    # z = data[0, :, 0]
    # t = np.arange(0, z.shape[-1]) * dt

    # 鹏哥（txt）的数据
    # 对于probes_ 系列数据 第一列是时间， 第二列及以后是数据。这里先用第二列没问题
    # 应该考虑缩尺比的问题 对于自己的数据
    # λ = 50 H* = H/λ
    lam = 50
    path = 'data/probes_M200.txt'
    # path = 'D:\\code_project\\DFT\\DFT\\DFT\\DFT\\old-\\波面.txt'
    data = np.loadtxt(path)
    # 除去前面的启动，波慢慢摇起来的过程
    t = data[1002:, 0]
    z = data[1002:, 2]
    dt = t[1] - t[0]
    plt.plot(t, z)
    plt.show()

    w, p = wave_spectrum_fft(z, dt)
    p_filter = mean_filter(p, kernel_size=30)

# 因为引用关系 只调用函数，这部分就屏蔽掉了 时间 2023年5月28日
#     omegam = np.arange(0.1, 2, 0.001)
#     S = jon.jonswap_matlab(omegam, Hs=1.88, Tp=8.8, Gamma=3.3, FiguringSpectrum=2)

#     plt.plot(w[:len(p)]/np.sqrt(lam), p_filter*lam**2*np.sqrt(lam))
#     plt.plot(omegam, S)
#     plt.legend(['Specturm by fft', 'Ground Truth'])
#     plt.show()
    pass
