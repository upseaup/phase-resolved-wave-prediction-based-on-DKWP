# 时间 23年7月28日
# 文件名字 lwt
# 概述     利用线性波浪理论对某个点的海浪数据数据进行预报
# 作者     冯胖
# 时间 23年7月31日
# 增加     增加选取总功率谱的 85% 部分
import numpy as np
import matplotlib.pyplot as plt
# 想办法把他给封装起来
# 输入是一段数据。numpy 利用shape 得到N, 另一个输入应该是dt 
def lwt_recon(data, dt, w_d=0.5, w_u=1, p_range=0.85, recon=False, picture=False):
    """
    描述： 把一段数据利用fft分解，并重构 需要注意的是 如果没有到85%的能量，对频率上限额外调整。
    输入:  data, numpy 一维数据，需要利用shape[0] 得到N
           dt,   采样时间
           w_d, w_u, 选取频率的上限和下限，需要保证在整体功率谱的85%左右，down, up
           p_range, 默认 .85, 是选取能量谱对应频率的范围 
           recon, 默认False,设置w_lo wr 为85%能量, 如果为真,可用来调试，则设置 w_lo w_r 为 0, -1 并打印
                  重构的数据和真实的数据是否一致。
           picture, 默认为假，不打印重构后的和原来图像，如果为真，则打印重构和原来对比，以及频率选取信息
                  主要用于调试，
    return:
            w_np, 选取哪些w 做为lwp的基本频率，
            a,b   原数据fft的实部和虚部
            k,    波数， 需要注意的是 w a b k 都是经过 w_lo w_r 限制过的。
            N,    输入的数据的点的个数 
    问题： 对于不同的海浪数据，他的85%能量带是不一样的，可能需要写个程序进行调整。
           现在的调整方法是，固定频率下限，调整上限，可能不是太合理，但是先这样吧。
    时间 23年 7月31
    更新 把在复原中的打印放到了了picture下面，
    """
    
    N = data.shape[0]
    T = N * dt 
    # 局部的常变量
    G = 9.8    # 地球重力加速度
    W_LO = w_d # 限制频率下限，得到85%的功率谱能量
    W_R  = w_u   # 限制频率上限，
    P_RANGE = p_range
    
    # 因为数据是实数，所以用rfft  好像直接就是一半 不用取一半了
    sp = np.fft.rfft(data)
    f = np.fft.rfftfreq(N, dt)


    w_np = f * 2 * np.pi
    
    if recon == False:
        w_lo = np.argwhere(w_np>W_LO)
        w_r = np.argwhere(w_np<W_R)
        
        # print(w_lo.shape, w_l[0, 0], w_r[-1, 0])
        w_lo = w_lo[0, 0]
        w_r = w_r[-1, 0]
        
        # 调整 功率谱对应的 w_l w_r 范围  因为只有预报是需要 这个范围的，所以复原
        # 的部分不需要这个程序
        p = (np.abs(sp)**2 * dt * 2) / (2 * np.pi * N)
        p_sum = np.sum(p)
        p_part = np.sum(p[w_lo:w_r])
        p_part_old = 0 
        if p_part/p_sum > P_RANGE:
            while p_part/p_sum > P_RANGE:
                # 保留一步的旧值
                p_part_old = p_part
                w_r_old = w_r
                # w_lo = w_lo + 1
                w_r = w_r - 1

                p_part = np.sum(p[w_lo:w_r])
                # print('1')
                # print('选取的功率在整体功率中的比例：', np.sum(p[w_lo:w_r])/np.sum(p))

        elif p_part/p_sum < P_RANGE:
            while p_part/p_sum < P_RANGE:
                p_part_old = p_part
                w_r_old = w_r
                # w_lo = w_lo - 1
                w_r = w_r + 1
                p_part = np.sum(p[w_lo:w_r])
                # print('选取的功率在整体功率中的比例：', np.sum(p[w_lo:w_r])/np.sum(p))

        elif p_part/p_sum == P_RANGE:
            pass
        else:
            print("something error!!")
            assert 1==0
        # 处理新选的值比旧的值差的情况
        if np.abs(p_part_old/p_sum - P_RANGE) < np.abs(p_part/p_sum -P_RANGE):
            w_r = w_r_old
            # print('旧值比新值好，使用旧值')
    elif recon == True:
        w_lo = 0
        w_r = -1
        # print('复原原来数据，用于调试')
    else :
        print('得到错误参数！终止程序')
        assert 1==0
    
    # 这里fft 之后得到的sp 应该表示成功率谱的形式，然后判断一下 选取的范围是否满足
    # 85% 的功率谱。
    p = (np.abs(sp)**2 * dt * 2) / (2 * np.pi * N)
    # print('选取的功率在整体功率中的比例：', np.sum(p[w_lo:w_r])/np.sum(p))
    # w_np = w_np[w_lo:w_r]
    # 得到 波数k
    k = w_np[w_lo:w_r] **2 /G

    # 需要计算 计算实部和虚部
    a = np.real(sp[w_lo:w_r]) * 2

    a = a / N

    b = np.imag(sp[w_lo:w_r]) * -2
    b = b / N
    
    if picture :
        print('选取的功率在整体功率中的比例：', np.sum(p[w_lo:w_r])/np.sum(p))
        print('N', N, 'T', N*dt)
        print('选取的频率上下限为', w_lo, w_r, w_np[w_lo], w_np[w_r])
        # angle=np.angle(sp)
        t = np.arange(N,)
        t_np = np.arange(N,) * dt # + 100
        t_np = t_np.tolist()
        wave_hot = []
        print(w_np.shape, len(t_np), a.shape, )
        # 下面的2000应该改成N
        for i in range(N):
            # print(np.cos( w_np * t_np[i]).shape)
            wave_hot.append(np.sum(a * np.cos( w_np[w_lo:w_r] * t_np[i])) + np.sum(b * np.sin( w_np[w_lo:w_r] * t_np[i])))
        # tt = a * np.cos(-w_np * t_np[i])
        plt.plot(t[:N], data[:N,])
        plt.plot(t[:N], wave_hot)
        plt.show()
    # 需要注意的是 只有返回的w_np 需要范围，其他在计算的过程中已经考虑了，
    return w_np[w_lo:w_r], a, b, k, N


def lwt_predicton(w_np, a, b, k , x, N, dt):
    """
    简述： 利用 lwt_recon 得到的 w, a, b, k 对 x位置的波浪进行预报
    公式： Bayesian machine learning(Applied Energy, 利用贝叶斯网络做海浪预报的文章)的公式17
           目前这个程序中的公式直观上来说和王战老师给的代码是一致的。（predciton）
    输入： w_np, 85%功率谱的频率，
          a, b, 分别为 w_np 对应的原始数据fft后的实部和虚部，
          k,   w_np 对应的波数，
          x,   希望预报的位置，应为正数，
          N,   希望预报的点的个数, 注意是从0开始的。
          dt,  采样时间
          
    输出： 希望预报多长的时间序列呢？这个长度如何确定呢？
          wave_hot  一个list 
    """
    # 预报
    t_np = np.arange(int(N),) * dt # + 100
    t_np = t_np.tolist()
    wave_hot = []

    for i in range(N):
        # print(sp.shape, k.shape, w_np.shape, angle.shape)
        # 这个就是用 公式17的写法， 下面这个是利用三角公式展开的，两个是完全一样的，只是写法不同
        # wave_hot.append(np.sum(np.sqrt(a**2 + b**2)*np.cos(k *x - w_np* t_np[i] -angle[w_lo:w_r] )))
        wave_hot.append(np.sum(a * np.cos(-k *x + w_np * t_np[i])) + np.sum(b * np.sin(-k * x + w_np * t_np[i])))
        # 求angle 和求 a b 其实是等价的, angle = arctan(b/a)=np.angle(sp) 可能需要注意角度在哪个象限
        # 然后复现的时候也是一样的, 具体见上面公式， 
    return wave_hot


def lwt_predicton_v2(w_np, a, angle, k , x, N, dt):
    """
    简述： 利用 lwt_recon 得到的 w, a, b, k 对 x位置的波浪进行预报
    公式： Bayesian machine learning(Applied Energy, 利用贝叶斯网络做海浪预报的文章)的公式17
           目前这个程序中的公式直观上来说和王战老师给的代码是一致的。（predciton）
    输入： w_np, 85%功率谱的频率，
          a,     w_np 对应的幅值
          angle, 对应的相位
          k,   w_np 对应的波数，
          x,   希望预报的位置，应为正数，
          N,   希望预报的点的个数, 注意是从0开始的。
          dt,  采样时间
          
    输出： 希望预报多长的时间序列呢？这个长度如何确定呢？
          wave_hot  一个list 
    需要注意的是，这里的 a w 都是没有滤过波的。
    """
    # 预报
    t_np = np.arange(int(N),) * dt # + 100
    t_np = t_np.tolist()
    wave_hot = []

    for i in range(N):
        # print(sp.shape, k.shape, w_np.shape, angle.shape)
        # 这个就是用 公式17的写法， 下面这个是利用三角公式展开的，两个是完全一样的，只是写法不同
        wave_hot.append(np.sum(a*np.cos(k *x - w_np* t_np[i] -angle )))
        # wave_hot.append(np.sum(a * np.cos(-k *x + w_np * t_np[i])) + np.sum(b * np.sin(-k * x + w_np * t_np[i])))
        # 求angle 和求 a b 其实是等价的, angle = arctan(b/a)=np.angle(sp) 可能需要注意角度在哪个象限
        # 然后复现的时候也是一样的, 具体见上面公式， 
    return wave_hot