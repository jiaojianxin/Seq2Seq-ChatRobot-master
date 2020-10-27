# encoding: utf-8
import numpy as np
import tensorflow as tf

from DataProcessing import DataUnits
from SequenceToSequence import Seq2Seq
from Parameters import Parameters as pm


def predict():
    """预测模块"""
    du = DataUnits(pm.data_path, pm.word2id_path, 'decode')
    batch_size = 1
    model = Seq2Seq(batch_size=batch_size, mode='decode')
    # 加载模型
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint(pm.save_path)  # latest_checkpoint() 方法查找最新保存的模型
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    while True:
        predict_contents = input('请输入对话, "E" 结束对话： ')
        predict_contents = predict_contents.replace(' ', '')
        if predict_contents == 'E':
            print('对话结束！')
            break
        elif not predict_contents:
            continue
        contents2id = du.sequence2id(predict_contents)
        x = np.asarray(contents2id).reshape((1, -1))
        xl = np.asarray(len(contents2id)).reshape((1,))
        pred = model.predict(session, np.array(x), np.array(xl))[0]

        res = du.id2sequence(pred)  # 将id转化为句子
        print(f'机器人回答：{res}\n')


if __name__ == '__main__':
    predict()
