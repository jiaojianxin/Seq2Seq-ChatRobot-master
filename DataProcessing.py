import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr


def read_json(read_path):
    with open(read_path, 'r', encoding='utf-8') as f:
        load_dict = json.load(f)
    return load_dict


def get_bert_vec(filename):
    """读取bert向量"""
    vec_np_list = np.load(filename)
    vec_np_list = vec_np_list.astype('float32')
    vec_tf = tf.constant(vec_np_list)

    return vec_tf


class DataUnits(object):
    def __init__(self, data_path, word2id_path, mode='train'):
        assert mode in ['train', 'decode'], 'ValueError: mode必须为train或为decode'
        self.word2id = read_json(word2id_path)

        if mode == 'train':
            self.question_content, self.answer_content = self.read_data(data_path)
            self.data_len = len(self.question_content)
        else:
            self.id2word = {v: k for k, v in self.word2id.items()}

    def read_data(self, filename):
        """读取数据 --> [吃了吗？， 早上好...] [吃了， 好...]"""
        question_content, answer_content = [], []
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read()
        # data = self._regular_(data)

        data_list = data.split('\n')
        data_list = [x.split('\t') for x in data_list]

        _ = [[question_content.append(list(x[0])), answer_content.append(list(x[1]))] for x in data_list \
             if len(x) == 2 and len(x[0]) > 0 and len(x[1]) > 0 and x[1] != '=。=']

        answer_content = [x + ['<E>'] for x in answer_content]

        return question_content, answer_content

    def sequence2id(self, contents):
        """
        将句子转化为数字代替
        :param contents: ['你', '好', '吗']
        :param word2id: {'你': 23, '好': 45, '吗': 87, '交': 12...}
        :return: [23, 45, 87]
        """
        contents2id = []
        unk_index = self.word2id['<UNK>']
        for word in contents:
            index = self.word2id.get(word, unk_index)
            contents2id.append(index)
        return contents2id

    def id2sequence(self, ids):
        """将id转化为句子"""
        res = []
        for id in ids:
            if id not in [0, 1, 2, 3]:
                res.append(self.id2word.get(id, '<UNK>'))
        return ''.join(res)

    def next_batch(self, batch_size=64):
        """将字转换为id后返回padding后的句子及句子的真实长度"""
        question2ids = [self.sequence2id(contents) for contents in self.question_content]
        answer2ids = [self.sequence2id(contents) for contents in self.answer_content]

        x = np.asarray(question2ids)
        y = np.asarray(answer2ids)
        num_batch = int((self.data_len - 1) / batch_size) + 1  # 计算一个epoch,需要多少次batch

        indices = np.random.permutation(self.data_len)  # 生成随机数列
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = batch_size * i
            end_id = batch_size * (i + 1)
            if end_id > self.data_len:
                return
            # end_id = min(batch_size * (i + 1), self.data_len)
            x_batch, y_batch = x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

            x_pad, x_seq_len = self.process_seq(x_batch)
            y_pad, y_seq_len = self.process_seq(y_batch)

            yield x_pad, x_seq_len, y_pad, y_seq_len

    def process_seq(self, batch):
        """
        计算一个batch里面最长句子，对其他做填充操作并保存真实长度
        :param batch: [[2,3], [4,5,6,7]]
        :return: [[2,3,0,0], [4,5,6,7]]
        """
        seq_len = []
        max_len = max(map(lambda x: len(x), batch))  # 计算一个batch中最长长度
        for i in range(len(batch)):
            seq_len.append(len(batch[i]))
        # 对短句进行填充, 默认value=0
        x_pad = kr.preprocessing.sequence.pad_sequences(batch, max_len, padding='post', truncating='post')

        return x_pad, np.asarray(seq_len)

    # def __len__(self):
    #     """返回处理后的语料库中问答对的数量"""
    #     return self.data_len

