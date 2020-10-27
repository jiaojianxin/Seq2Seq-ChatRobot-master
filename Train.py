# encoding: utf-8
import os
import time

import tensorflow as tf
from SequenceToSequence import Seq2Seq
# from tqdm import tqdm
import numpy as np

from Parameters import Parameters as pm
from DataProcessing import DataUnits


# 是否在原有模型的基础上继续训练
continue_train = False


def train():
    """训练模型"""
    save_path = os.path.join(pm.save_path, 'best_validation')
    # 获取数据
    du = DataUnits(pm.data_path, pm.word2id_path, 'train')

    # 创建session的时候设置显存根据需要动态申请
    tf.reset_default_graph()
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = False

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            # 定义模型
            model = Seq2Seq(batch_size=pm.batch_size, mode='train')

            init = tf.global_variables_initializer()
            writer=tf.summary.FileWriter(pm.graph, sess.graph)
            sess.run(init)
            if continue_train:
                model.load(sess, save_path=pm.save_path)

            for epoch in range(1, pm.num_epochs + 1):
                print(f'epoch: {epoch}')
                start_time = time.time()
                costs = []
                for x, xl, y, yl in du.next_batch(batch_size=pm.batch_size):
                    # x为question,xl为question实际长度；y为answer,yl为answer实际长度
                    max_len = np.max(yl)  # 实际最长句子的长度
                    y = y[:, 0:max_len]  # 表示所有行的第0:max_len列
                    cost, lr, global_step = model.train(sess, x, xl, y, yl)
                    costs.append(cost)
                    if global_step % 30 == 0:
                        model.save(sess, global_step, save_path=save_path)
                        print('ModelSave...')
                        print(f'global_step: {global_step}, cost: {round(cost, 6)}, lr: {round(lr, 8)}')
                    # bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(epoch, np.mean(costs), lr))
                print(f'--------------------------------\nmean_loss: {np.mean(costs)}\n--------------------------------')
                end_time = time.time()
                print(f'epoch: {epoch}-->时间：{round(end_time-start_time, 2)}秒')


if __name__ == '__main__':
    train()
