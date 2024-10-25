# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
"""
# Project    : main.py
# File       : jonswap_matlab.py
# Time       ：2023/3/31 12:50
# Author     ：pang
# version    ： 
# Description：matlab官网程序改，输出的是S(w)
              （https://ww2.mathworks.cn/matlabcentral/fileexchange/70082-wavemodeling）
              [1]. S. M. Vazirizade, “An Intelligent Integrated Method for Reliability Estimation of Offshore Structures
                   Wave Loading Applied in Time Domain,” The University of Arizona., 2019.
               
# 需要注意的是 ：1.mind 周期 is Tz or Tm（Tp,matlab中参考文献给的也是Tp） ,一个是过零周期，一个是峰值周期。    # Tz 过零周期 Tm波峰周期
#                Tm = Tz * (0.327 * np.exp(-0.315 * Gamma) + 1.17)
#                不同程序表现出来的峰值频率不同可能就是T选取不同造成的。
#                matlab 中有把输入Tz 转化为Tm 的经验公式。main_copy中是没有的。所以两者频率是有所不同的。
#                而，hos-nwt1.2(linux)下源码中输入是Tp.合理推断,Tp=Tm,且输出文件中明确说明（Peak period，probes.dat文件）
#              1.1 fp = 1/Tp wp = 2*pi*fp
#              2. S(w) 和 S(f)差一个常数 2*pi ,这也是为什么main_copy中谱峰频率偏高的原因。 自己做过验证
#               . main_copy是 s(f), 使用方法是随机波浪及其工程应用中公式5.4.7, 只要给Tp Hs gamma 即可。
#               . hos-nwt1.2源码(linux)中 initial_condition 中 生成波用的方法（下面记为ampli）和matlab程序，是s(w)
#              2.1 谱实现细节
#                 但是ampli 的  alphabar 是 1 不知道为什么，下称为系数。可能是后面还需要成别的系数，后面hos-nwt的程序再一起处理了，没有确认
#               . matlab 使用方法是文献[1]中，2.3.1中的公式（公式2.5 和公式2.9 应该有点小错误，但是不影响，可以马上看出）
#                 （matlab 和 main_copy 使用的公式是不一样的）所以，matlab 和 之前自己写的main_copy(由S(f)换算成S(w)，公式5.4.7),有些差距也是正常的。
#               . ampli 的 系数借鉴 matlabd alphabar后，ampli和matlab生成的谱(显然此时是保证两个程序的Tp Hs gamma是一样的)是完全一样的。
#               . 三者各自除各自最大值进行归一化处理(S(f)谱不除2*pi, 因为最大值中是包含2*pi的)，三者完全重合。说明三者不同的地方只是系数不同。
#              2.2 验证
#               . 需要注意的是，虽然目前三个求谱程序输入的都是Hs，Tp, gamma. 但是不同的Hs 在归一化操作后是相同的。而Tp会影响wm（归一化中可以看出来）
#                 和幅值（归一化中看不出来）。gamma 因为会影响谱的形状也有影响。这个主要是说，hos-nwt源码中系数是不准确的可能和这个原因有关。
#                 不能用归一化后的谱进行分析（比如是否生成的数据反演的谱和真值谱是否能对的上），如用归一化分析，需要额外对Hs的影响进行分析。
#                 这个结论也可以通过（https://www.desmos.com/calculator/k5pqn0kslr?lang=zh-CN）的可视化作图进行分析，结论也是一致的。
#                 分析谱生成公式无论是文献[1]中的公式还是随机波浪及其工程应用中的例子，都说明Hs只影响系数。
#               . 那有一个问题 既然谱对于Hs的体现不是太明显，那用波陡评价一个波还准确吗。或者说是不能归一化处理的。这样谱中就有Hs的体现了。                       
# 所以          
# 1. 当对 matlab anpli 进行除他们各自最大值的归一化的时候，他们俩是一样的。目前anpli 没有考虑sigmab=0.09  （不在需要anpli）
# 2. 目前可以用 jonswap_matlab 函数生成谱。输入是Omega Hs Tp Gamma 
时间 8月2日
更改 增加两个本来在main中的函数，用于生成w s a, 便于后续保存。
以及用 w x t 生成 wt kx 
时间 8月16日
更改 增加 get_lwt_range 用于用理论上的a w mu 求出理论上的预报范围 用于指导 在ann设计中 dx的选取 
"""
# 海况4 和其他一些超参数。
Hs = 1.88
Tp = 8.8
gamma = 3.3
M = 300
w_l = 0.2
w_h = round(3 * 2 * np.pi / Tp, 2)

def _jonswap_matlab_old(Omega=0, Hs=1.88, Tz=8.8, Type=1, Tend=1, Cap=1, FiguringSpectrum=1):
    """
    需要注意输入的周期是Tp 还是Tz  为了和HOS—nwt一样，应该直接使用Tp 不需要Tz转换为Tp
    old版本为输入是Tz 新版（没有old）输入是Tp
    :param Omega:
    :param Hs:
    :param Tz:
    :param Type:
    :param Tend:
    :param Cap:
    :param FiguringSpectrum:
    :return:
    """
    g = 9.806
    Gamma = 3.3

    # Tz 过零周期 Tm波峰周期
    # Tz = 8.8
    Tm = Tz * (0.327 * np.exp(-0.315 * Gamma) + 1.17)
    # 先大家都给定peak周期
    # Tm = Tz
    print('Tz,Tm', Tz, Tm)
    # % Tm = Tz * ((11 + gamma). / (5 + Gamma)). ^ .5

    Beta = 5/4
    SigmaA = 0.07
    SigmaB = 0.09

    # OmegaGap = Omega[1] -Omega[0]

    Omegam = 2 * np.pi / Tm
    # Omegam = 0.7
    sigma = np.where(Omega < Omegam, np.ones(shape=Omega.shape)*SigmaA, np.ones(shape=Omega.shape)*SigmaB)
    A = np.exp(-((Omega/Omegam - 1)/(sigma*np.sqrt(2)))**2)
    # todo 这个log 和matlab  是一个吗
    alphabar = 5.058*(1-0.287*np.log(Gamma))*(Hs/Tm**2)**2
    alpha =0.0081
    S = alphabar*g**2 * Omega**(-5)*np.exp(-(Beta*(Omega/Omegam)**(-4)))*Gamma**A
    if FiguringSpectrum == 1:
        plt.plot(Omega, S)
        plt.title('wave Spectrum')
        plt.show()
    pass
    return S


def jonswap_matlab(Omega=0, Hs=1.88, Tp=8.8, Gamma = 3.3, FiguringSpectrum=2):
    """
    需要注意输入的周期是Tp 还是Tz  为了和HOS—nwt一样，应该直接使用Tp 不需要Tz转换为Tp

    :param Omega:
    :param Hs:
    :param Tp:
    :param Gamma:
    :param FiguringSpectrum:
    :return: S(w)
    """
    g = 9.806

    # Tz 过零周期 Tm波峰周期
    # Tz = 8.8
    # Tm = Tz * (0.327 * np.exp(-0.315 * Gamma) + 1.17)
    # 先大家都给定peak周期
    # Tm = Tz
    # print('Tz,Tm', Tz, Tm)
    # % Tm = Tz * ((11 + gamma). / (5 + Gamma)). ^ .5

    Beta = 5/4
    SigmaA = 0.07
    SigmaB = 0.09

    Omegam = 2 * np.pi / Tp

    sigma = np.where(Omega < Omegam, np.ones(shape=Omega.shape)*SigmaA, np.ones(shape=Omega.shape)*SigmaB)
    A = np.exp(-((Omega/Omegam - 1)/(sigma*np.sqrt(2)))**2)
    # todo 这个log 和matlab  是一个吗
    alphabar = 5.058*(1-0.287*np.log(Gamma))*(Hs/Tp**2)**2
    alpha =0.0081
    S = alphabar*g**2 * Omega**(-5)*np.exp(-(Beta*(Omega/Omegam)**(-4)))*Gamma**A
    if FiguringSpectrum == 1:
        plt.plot(Omega, S)
        plt.title('wave Spectrum')
        plt.show()
    pass
    return S


def get_w_a(Tp=Tp, Hs=Hs, Gamma=gamma, M=M, w_l=w_l, w_h=w_h):
    # 看下这个函数的输入输出是什么
    # tp hs gamma M w_l w_h 
    # 返回应该是 w_i_hot 和 a 把s 也返回了吧 
    w = np.arange(0.1, w_h+1, 0.001)
    S = jonswap_matlab(w, Tp=Tp, Hs=Hs, Gamma=gamma,FiguringSpectrum=2)

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

    s_hot = jonswap_matlab(w_i_hot, Tp=Tp, Hs=Hs, Gamma=gamma, FiguringSpectrum=2)
    # # 看生成数据用的功率谱
    plt.plot(w, S)
    plt.plot(w_i_hot, s_hot)
    plt.legend(['S_hot', 'S ground value'])
    plt.show()

    # 求幅值
    a = np.sqrt(2 * dw * s_hot)
    return w_i_hot, s_hot, a


# pre_x_t wave 这两个函数组合 用来计算海浪  
# 这里可以设成一个函数 通过赋值需要的演化时间 t_need 来 得到新的
# 这里的变量就不赋值了 因为赋值还要在这里保存一份。
def pre_x_t(t_end, dt, w_i_hot, dx, NX=2):
    # 想一下这里的输入都是什么
    # t_end 海况有多长
    # NX 默认有2个浪高仪 
    # dx 浪高仪间距 
    # 返回值
    
    G = 9.806
    t = np.arange(0, t_end, dt)
    wt = w_i_hot[:, np.newaxis] * t[:, np.newaxis].T
    k = np.square(w_i_hot) / G
    # 假设浪高仪 间距为 dx = 0.1
    x = np.arange(0, NX * dx, dx)
    # 得到kx
    kx = k[:, np.newaxis] * x[:, np.newaxis].T
    return wt, kx

def wave(a, kx, wt, mu_array, N_DATA, NX): 
    # N_DATA 一共要生成多少海况，这个值要小于mu_array 的第一个shape
    # NX 浪高仪个数。
    # 返回一个 三维的 海况个数 海况 浪高仪 的数据
    wave_data = np.zeros(shape=[N_DATA, wt.shape[1], NX], dtype=np.float32)
    for i in range(N_DATA):
            mu = mu_array[i, :]
            # mu = np.tile(mu, [1, wt.shape[1]])
            for j in range(NX):
                # wave_part = a[:, np.newaxis] * np.cos(wt - kx[:, j:(j + 1)] + mu[:, np.newaxis])
                # test = kx[:, j:(j + 1)]
                wave_part = a[:, np.newaxis] * np.cos(kx[:, j:(j + 1)] - wt + mu[:, np.newaxis])
                wave = np.sum(wave_part, axis=0)
                wave_data[i, :, j] = wave
    return wave_data 

def get_lwt_range(N_P_TRAIN, x, w, a, mu, t_end, dt):
    """
    时间 23年 8月16日
    作者 冯胖
    作用 利用理论上的 a w mu 以及lwt理论 给出 数据的lwt可预报范围
         放在 PB程序的数据生成部分后面就行。
    N_P_TRAIN 用于预报的点的个数，
    x    A B 两点距离
    w    光滑的jonswap谱的频率
    a    光滑的jonswap谱的幅值
    mu   随机相位 应该是np.array shape 是 M,1 其中M是w个数
    t_end 从0时刻开始，一共要预报多远的结果。
    dt   采样时间
    返回值 预报的波浪情况
    # 这个小块程序是用线性波浪理论的方法计算可预报范围
    # 因为PB系列其实一般只有一个Nx 所以这里 x = dx 即可，如果有需要再改
    """
    G = 9.8
    c = G/(2*w)
    k = w*w/G
    t = np.arange(0, t_end, dt)
    a_ele = a.copy()
    # mu = train_mu_array[0, :]
    wave_predict_list = []
    loss_list = []
    print(a.max(), a.min())
    print('c max c min:', c.max(), c.min())

    # 这里假设 给定 x 和一个范围的时间t
    for time in t:

        a_ele = a.copy()
        for i in range(w.shape[0]):
            x_r_min = c[i] * (time - N_P_TRAIN*dt)
            if x_r_min < 0:
                x_r_min = 0 
            x_r_max = c[i] * time
            # print('x_r max, x_r min', x_r.max(), x_r.min())
            # print(x.shape, x_r_min.shape, )
            # print(x, x_r_max, x_r_min)
            if x>=x_r_min and x<=x_r_max:
                a_ele[i] = a_ele[i]*1
                #print(test)
            else:
                # print('a i = 0', i)
                a_ele[i] = a_ele[i]*0
            # print('a_ele[i], a[i]', a_ele[i], a[i])
        # print(a_ele.max(), a_ele.min())

    # 计算最终的结果
        elevation = np.sum(a_ele * np.cos(k * x - w * time+ np.squeeze(mu)))
        # if elevation ==  0:
        #     print('elvevation ==0',time, np.sum(a_ele))
        wave_predict_list.append(elevation)
    return wave_predict_list 


if __name__ == "__main__":
    Omega = np.arange(0.1, 2, 0.001)
    S = jonswap_matlab(Omega, Tp=8.8, FiguringSpectrum=2)
# —————————— main_copy start ———————————————————————————
    import main_copy

    w = np.arange(0.1, 2, 0.001)
    f = w / (2 * np.pi)
    s = main_copy.jonswap_fun(f, H=1.88, TP=8.8, gama=3.3)
# --------------end----------------
    # filter_p = mean_filter(p, kernel_size=50)
    # plt.plot(w_np[:len(filter_p)] / np.sqrt(lam), filter_p)
    # plt.plot(w, s)
    # # plt.legend('dft-fortran S')
    # plt.show()
# —————————— main_copy end ————————————————————————————
# —————————— hos-nwt-ampli-start ——————————————————————
    f_base = 0.001
    freq = np.arange(0.01, 2, f_base)
    g = 9.8
    Tz = 8.8
    Hs = 1.88
    gamma = 3.3
    # 假定 给的都是peak周期 即，tp=8.8
    # Tp = Tz * (0.327 * np.exp(-0.315 * gamma) + 1.17)
    Tp = Tz

    sigma = 0.07
    # sigma = np.where(Omega < 2*np.pi/Tp, np.ones(shape=Omega.shape) * 0.07, np.ones(shape=Omega.shape) * 0.09)
    # plt.plot(Omega, sigma)
    # plt.show()
    alphabar = 5.058 * (1 - 0.287 * np.log(gamma)) * (Hs / Tp ** 2) ** 2
    ampli_t = 1 / (2 * np.pi) ** 4 * g ** 2 / freq ** 5 * np.exp(-5.0 / 4.0 * (freq * Tp) ** (-4)) \
            * gamma ** (np.exp(-(freq - 1.0 / Tp) ** 2 / (2.0 * (sigma / Tp) ** 2)))
    ampli = ampli_t * alphabar
# ——————————— hos-nwt-ampli-end ——————————————————————————
#     plt.plot(freq * 2 * np.pi, ampli/(2*np.pi))
    plt.plot(w, s/(2*np.pi),)
    plt.plot(Omega, S)
    plt.legend(['main_copy', 'matlab'])
    plt.show()
    pass