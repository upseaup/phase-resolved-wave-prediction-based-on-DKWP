import numpy as np
import matplotlib.pyplot as plt


# 把各个算法封装为函数 便于使用
# 多项式算法 全局算法
def polynomial_filter(data, data_name, z_ture=3, p_num=7):
    """

    :param data_name: 数据名字
    :param data: 需要滤波的数据
    :param z_ture: 野值判断标准
    :param p_num: 多项式阶次
    :return:
    """
    # 野值剔除 利用 3 sigma 法则 用野值周围的6个数据的平均值剔除野值
    # Z-score方法 得到数据加速度的分布情况 先求 均值mu和方差sigma
    angles_np = np.array(data)
    angles_a = angles_np[1:] - angles_np[0:-1]

    mu = np.sum(angles_a) / len(angles_a)
    sigma = np.sqrt(np.sum(np.square(angles_a - mu)) / (len(angles_a) - 1))

    z_a = (angles_a - mu) / sigma
    plt.plot(np.arange(len(z_a)), z_a)
    plt.plot(np.arange(len(z_a)), np.ones(shape=z_a.shape,) * 3)
    plt.plot(np.arange(len(z_a)), np.ones(shape=z_a.shape,) * (-3))
    plt.title('Standardization residual')
    plt.show()
    # # 绘制密度图
    # s.plot()
    n, bins, patches = plt.hist(z_a, 50)
    plt.title('residual distribution')
    plt.show()
    # 剔除数据中  |z - z_ture|>0 的 z ||为绝对值符号  z_ture通常设置为2.5、3.0,3.5
    Z_TRUE = z_ture
    ye = np.argwhere(z_a > Z_TRUE)
    ye_ = np.argwhere(z_a < -Z_TRUE)
    ye_all = np.concatenate((ye, ye_), axis=0)
    # 索引排序
    ye_all_sort = np.sort(np.squeeze(ye_all))

    angle_ye_out = np.array(data)

    N_sum = 10
    angle_sum = 0
    for i in ye_all_sort:

        for j in range(5):
            if (i - j) in ye_all_sort:
                N_sum = N_sum - 1
            else:
                angle_sum = angle_sum + data[i + 1 - j]

            if (i + j) in ye_all_sort:
                N_sum = N_sum - 1
            else:
                angle_sum = angle_sum + data[i + 1 + j]

        angle_ye_out[i + 1] = angle_sum / N_sum
        angle_sum = 0
        N_sum = 10

    # 利用numpy 库函数 进行多项式滤波
    x_angles = np.arange(len(angle_ye_out))
    angles_yonp = np.array(angle_ye_out)

    z_angles = np.polyfit(x_angles, angles_yonp, p_num)
    p_angles = np.poly1d(z_angles)

    p_v_angles = p_angles(x_angles)

    # 原始数据 剔除野值后 多项式拟合后
    plt.plot(x_angles, data)
    # plt.plot(x_angles, angles_yonp)
    plt.plot(x_angles, p_v_angles)
    plt.legend(labels=[data_name, "polynomial filtering"], )
    plt.show()
    return p_v_angles


def class_polynomial_filter(data, data_name, z_ture=3, p_num=7):
    """
    # 这个是上课老师的方法 先拟合 然后剔除残差中3sigma
    :param data_name: 数据名字
    :param data: 需要滤波的数据
    :param z_ture: 野值判断标准
    :param p_num: 多项式阶次
    :return:
    """
    x_angles = np.arange(len(data))
    angles_np = np.array(data)

    z_angles = np.polyfit(x_angles, angles_np, 7)
    p_angles = np.poly1d(z_angles)

    p_v_angles = p_angles(x_angles)

    res = angles_np - p_v_angles
    # 原始数据 剔除野值后 多项式拟合后
    # plt.plot(x_angles, data)
    # # plt.plot(x_angles, angles_yonp)
    # plt.plot(x_angles, p_v_angles)
    # plt.legend(labels=[data_name, "polynomial filtering"], )
    # plt.show()
    plt.plot(x_angles, res)
    plt.legend(labels=['residual error'], )
    plt.show()
    # angles_np = np.array(data)
    # angles_a = angles_np[1:] - angles_np[0:-1]
    mu = np.sum(res) / len(res)
    sigma = np.sqrt(np.sum(np.square(res - mu)) / (len(res) - 1))

    z_a = (res - mu) / sigma
    plt.plot(np.arange(len(z_a)), z_a)
    plt.plot(np.arange(len(z_a)), np.ones(shape=z_a.shape,) * 3)
    plt.plot(np.arange(len(z_a)), np.ones(shape=z_a.shape,) * (-3))
    plt.title('Standardization residual')
    plt.show()
    # # 绘制密度图
    # s.plot()
    n, bins, patches = plt.hist(z_a, 50)
    plt.title('residual distribution')
    plt.show()
    # 剔除数据中  |z - z_ture|>0 的 z ||为绝对值符号  z_ture通常设置为2.5、3.0,3.5
    Z_TRUE = z_ture
    ye = np.argwhere(z_a > Z_TRUE)
    ye_ = np.argwhere(z_a < -Z_TRUE)
    ye_all = np.concatenate((ye, ye_), axis=0)
    # 索引排序
    ye_all_sort = np.sort(np.squeeze(ye_all))

    angle_ye_out = np.array(data)
    # print('课堂的滤波方法 得到的野值索引', ye_all_sort)
    # 剔除野值 平均数替代
    N_sum = 10
    angle_sum = 0
    for i in ye_all_sort:

        for j in range(5):
            if (i - j) in ye_all_sort:
                N_sum = N_sum - 1
            else:
                angle_sum = angle_sum + data[i - j]

            if (i + 1 + j) in ye_all_sort:
                N_sum = N_sum - 1
            else:
                angle_sum = angle_sum + data[i + 1 + j]
        # print(i, j, N_sum)
        angle_ye_out[i + 1] = angle_sum / N_sum
        angle_sum = 0
        N_sum = 10

    # 利用numpy 库函数 进行多项式滤波
    x_angles = np.arange(len(angle_ye_out))
    angles_yonp = np.array(angle_ye_out)

    z_angles = np.polyfit(x_angles, angles_yonp, p_num)
    p_angles = np.poly1d(z_angles)

    p_v_angles = p_angles(x_angles)

    # 原始数据 剔除野值后 多项式拟合后
    plt.plot(x_angles, data)
    # plt.plot(x_angles, angles_yonp)
    plt.plot(x_angles, p_v_angles)
    plt.legend(labels=[data_name, "polynomial filtering"], )
    plt.show()
    return p_v_angles


def sliding_filter(data, data_name, a=.6, N=10):
    """
    滑动滤波 两端用一阶低通滤波
    :param data:  需要注意的是 data 应该是一个list 而不是ndarray 因为下面的算法是list的而不是numpy的
    :param data_name:
    :param a: 低通滤波系数
    :param N: 滑动滤波系数
    :return: 滤波后数据
    """
    # 对数据前一部分做低通滤波
    angles_wa = [data[0]]
    for i, data_one in enumerate(data[1:int(N/2)]):
        angles_wa.append((1 - a) * angles_wa[-1] + a * data_one)
    # 中值滤波
    angles_m = []
    for i in range(len(data)):
        angles_m.append(np.sum(data[i - int(N/2): i + int(N / 2)]) / N)
    angles_m[:int(N / 2)] = angles_wa[:]
    angles_wa = [angles_m[-5]]
    for i, data_one in enumerate(data[-4:]):
        angles_wa.append((1 - a) * angles_wa[-1] + a * data_one)
    angles_m[-5:] = angles_wa[:]
    assert len(angles_m) == len(data)
#     print('angle_wa len, ', len(angles_wa))
    t = np.arange(len(data))
    plt.plot(t, data)
    plt.plot(t, angles_m)
    plt.legend(labels=[data_name, "sliding filtering"], )
    plt.show()
    return angles_m


def low_pass_filter(data, a=0.6, data_name=''):
    # 先考虑一阶低通滤波 属于用部分节拍信息
    # First order low-pass
    angles_folp = []
    # a = 0.2  # 低通系数
    # 添加第一个元素
    angles_folp.append(data[0])
    for i in data[1:]:
        angles_folp.append((1 - a) * angles_folp[-1] + a * i)

    t = np.arange(len(data))
    plt.plot(t, data)
    plt.plot(t, angles_folp)
    plt.legend(labels=[data_name, "low pass filtering"], )
    plt.show()
    return angles_folp


if __name__ == '__main__':
    b = [i for i in range(100)]
    # sliding filter 是全局滤波 局部滤波使用一阶低通滤波算法
    # 等下回来继续写 todo
    a = sliding_filter(b, data_name='test')
    pass
