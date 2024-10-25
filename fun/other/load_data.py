import numpy as np 
"""
时间 23年3月5日
内容 定义读取数据生成测试集 训练集文件
     后续用Koopman 可能需要更复杂的数据集结构
作者 冯胖子 
"""

# 定义一个函数读取数据并根据上面设计 定义数据集 
def get_data_set(data_path, pt, bs, be, N=1):
    """
    data_path 数据路径 这里应该是 DATA_TRAIN_NAME 这类的
    pt        P_train
    bs        b_start
    be        b_end
    N         b 点数据对应第几个浪高仪 默认数据只有两个浪高仪
              分别对应A 点数据 和B 点数据  0对应A点数据
              后期可能因为非线性波数据形式不一样，考虑其他方法
              比如直接在数据名字中读取出N值。后期用非线性波的时候
              可能数据格式也不一样了 
    return    x_data, y_data 
              分别对应数据 和 标签 
    其他      需要注意的是pm谱的数据格式是 N, 浪高仪， 时间
              jonswap谱的格式是 N, 时间，浪高仪
    """
    # 把 x_data 对应的0 改为 int(N-1)
    data = np.load(data_path)
    x_data = data[:, :pt, int(N-1)]       # 前p_train步作为训练数据 浪高仪0
    y_data = data[:, bs:be, N]
    return x_data, y_data 