# -*- coding:utf-8 -*-
class Parameters(object):
    """模型参数"""
    num_epochs = 60  # 迭代次数
    START_INDEX = 2  # 标志句子开始字符的索引
    END_INDEX = 1  # 标志句子结束字符的索引
    hidden_size = 128  # 隐藏层的个数
    batch_size = 32  # 每批进入模型句数，CPU对2的指数不友好，GPU用2的指数较好
    learning_rate = 0.001  # 学习率
    decay_step = 100000  # 学习率衰减步数
    min_learning_rate = 1e-6  # 最小学习率
    max_decode_step = 80  # 最大回答长度
    max_gradient_norm = 4.0  # 梯度剪裁，控制参数调节的大小，防止梯度爆炸
    keep_prob = 0.8  # 向量防止过拟合正则化（丢失）程度
    beam_width = 200  # BeamSearch时的宽度

    word_vec_path = r'./Data/vec.npy'  # 向量
    word2id_path = r'./Data/word2id.json'  # 字ID
    data_path = r'./Data/test_data.txt'  # 用于训练的数据

    graph = r'./graph'  # tensorflow计算图保存地址
    save_path = r'./model_path'  # 模型保存地址
