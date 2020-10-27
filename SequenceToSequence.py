# encoding: utf-8

"""
    SequenceToSequence模型
    定义了模型编码器、解码器、优化器、训练、预测
"""

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple, DropoutWrapper
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper, \
    TrainingHelper, BasicDecoder, BeamSearchDecoder
from tensorflow import layers
from tensorflow.python.ops import array_ops

from DataProcessing import get_bert_vec
from Parameters import Parameters as pm


class Seq2Seq(object):

    def __init__(self, batch_size, mode):
        """初始化函数"""
        self.batch_size = batch_size
        self.mode = mode
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = get_bert_vec(pm.word_vec_path)

        self.encoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, shape=[self.batch_size, ], name='encoder_inputs_length')
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

        if self.mode == 'train':
            self.decoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='decoder_inputs')
            self.decoder_inputs_length = tf.placeholder(tf.int32, shape=[self.batch_size, ], name='decoder_inputs_length')
            self.decoder_start_token = tf.ones(shape=(self.batch_size, 1), dtype=tf.int32) * pm.START_INDEX
            self.decoder_inputs_train = tf.concat([self.decoder_start_token, self.decoder_inputs], axis=1)

        self.build_model()

    def build_model(self):
        """
        构建模型
        1、编码器
        2、解码器
        3、优化器
        4、模型保存
        """
        encoder_outputs, encoder_state = self.build_encoder()
        self.build_decoder(encoder_outputs, encoder_state)
        if self.mode == 'train':
            self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_encoder(self):
        """构建编码器，返回网络输出及隐藏层状态"""
        encoder_embedding = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
        with tf.name_scope('encoder'):
            # 定义双向LSTM网络
            cell_fw = tf.contrib.rnn.LSTMCell(pm.hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(pm.hidden_size)
            if self.mode == 'train':
                cell_fw = DropoutWrapper(cell_fw, dtype=tf.float32, output_keep_prob=pm.keep_prob)
                cell_bw = DropoutWrapper(cell_bw, dtype=tf.float32, output_keep_prob=pm.keep_prob)
            else:
                cell_fw = DropoutWrapper(cell_fw, dtype=tf.float32, output_keep_prob=1.0)
                cell_bw = DropoutWrapper(cell_bw, dtype=tf.float32, output_keep_prob=1.0)

            # 创建双向递归神经网络的动态版本
            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_state, encoder_bw_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=encoder_embedding,
                                                sequence_length=self.encoder_inputs_length, dtype=tf.float32, swap_memory=True)
            # 将双向神经网络拼接
            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            encoder_state_c = tf.concat((encoder_fw_state.c, encoder_bw_state.c), 1)
            encoder_state_h = tf.concat((encoder_fw_state.h, encoder_bw_state.h), 1)
            encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        return encoder_outputs, encoder_state

    def build_decoder_cell(self, encoder_outputs, encoder_state):
        """构建解码器所有层"""
        sequence_length = self.encoder_inputs_length
        if self.mode == 'decode':
            # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份
            encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=pm.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=pm.beam_width)
            sequence_length = tf.contrib.seq2seq.tile_batch(sequence_length, multiplier=pm.beam_width)

        cell = tf.contrib.rnn.LSTMCell(pm.hidden_size * 2)

        if self.mode == 'train':
            cell = DropoutWrapper(cell, dtype=tf.float32, output_keep_prob=pm.keep_prob)
        else:
            cell = DropoutWrapper(cell, dtype=tf.float32, output_keep_prob=1.0)

        # 使用attention机制
        self.attention_mechanism = BahdanauAttention(num_units=pm.hidden_size, memory=encoder_outputs,
                                                     memory_sequence_length=sequence_length)

        def cell_input_fn(inputs, attention):
            attn_projection = layers.Dense(pm.hidden_size * 2, dtype=tf.float32, use_bias=False,
                                           name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))

        cell = AttentionWrapper(
            cell=cell,  # rnn cell实例，可以是单个cell，也可以是多个cell stack后的mutli layer rnn
            attention_mechanism=self.attention_mechanism,  # attention mechanism的实例，此处为BahdanauAttention
            attention_layer_size=pm.hidden_size,
            # 用来控制我们最后生成的attention是怎么得来;如果不是None，则在调用_compute_attention方法时，得到的加权和向量还会与output进行concat，然后再经过一个线性映射，变成维度为attention_layer_size的向量
            cell_input_fn=cell_input_fn,  # input送入decoder cell的方式，默认是会将input和上一步计算得到的attention拼接起来送入decoder cell,
            name='Attention_Wrapper')

        if self.mode == 'decode':
            decoder_initial_state = cell.zero_state(batch_size=self.batch_size * pm.beam_width,
                                                    dtype=tf.float32).clone(cell_state=encoder_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(cell_state=encoder_state)

        return cell, decoder_initial_state

    def build_decoder(self, encoder_outputs, encoder_state):
        """构建完整解码器"""
        with tf.variable_scope("decode"):
            decoder_cell, decoder_initial_state = self.build_decoder_cell(encoder_outputs, encoder_state)
            # 输出层投影
            decoder_output_projection = layers.Dense(self.embedding.shape[0], dtype=tf.float32,
                                                     use_bias=False,
                                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                                     name='decoder_output_projection')
            if self.mode == 'train':
                decoder_inputs_embdedded = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs_train)
                training_helper = TrainingHelper(inputs=decoder_inputs_embdedded,
                                                 sequence_length=self.decoder_inputs_length, name='training_helper')
                training_decoder = BasicDecoder(decoder_cell, training_helper, decoder_initial_state,
                                                decoder_output_projection)
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length)
                training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                  maximum_iterations=max_decoder_length)
                self.masks = tf.sequence_mask(self.decoder_inputs_length, maxlen=max_decoder_length,
                                              dtype=tf.float32, name='masks')
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=training_decoder_output.rnn_output,
                                                             targets=self.decoder_inputs,
                                                             weights=self.masks,  # mask，滤去padding的loss计算，使loss计算更准确。
                                                             average_across_timesteps=True,
                                                             average_across_batch=True
                                                             )
            else:
                # 预测模式
                start_token = [pm.START_INDEX] * self.batch_size
                end_token = pm.END_INDEX
                inference_decoder = BeamSearchDecoder(  # 广度优先，且减少内存消耗，搜索最优结果
                    cell=decoder_cell,
                    embedding=lambda x: tf.nn.embedding_lookup(self.embedding, x),
                    start_tokens=start_token,
                    end_token=end_token,
                    initial_state=decoder_initial_state,
                    beam_width=pm.beam_width,
                    output_layer=decoder_output_projection
                )
                inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                                   maximum_iterations=pm.max_decode_step)
                self.decoder_pred_decode = inference_decoder_output.predicted_ids
                self.decoder_pred_decode = tf.transpose(self.decoder_pred_decode, perm=[0, 2, 1])

    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, keep_prob, decode):
        """检查输入,返回输入字典"""
        input_batch_size = encoder_inputs.shape[0]
        assert input_batch_size == encoder_inputs_length.shape[0], 'encoder_inputs 和 encoder_inputs_length的第一个维度必须一致'
        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            assert target_batch_size == input_batch_size, 'encoder_inputs 和 decoder_inputs的第一个维度必须一致'
            assert target_batch_size == decoder_inputs_length.shape[0], 'decoder_inputs 和 decoder_inputs_length的第一个维度必须一致'

        input_feed = {self.encoder_inputs.name: encoder_inputs, self.encoder_inputs_length.name: encoder_inputs_length}
        input_feed[self.keep_prob.name] = keep_prob

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length
        return input_feed

    def build_optimizer(self):
        """构建优化器"""
        learning_rate = tf.train.polynomial_decay(pm.learning_rate, self.global_step, pm.decay_step, pm.min_learning_rate, power=0.5)
        self.current_learning_rate = learning_rate
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        # 优化器
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # 梯度裁剪
        clip_gradients, _ = tf.clip_by_global_norm(gradients, pm.max_gradient_norm)
        # 更新梯度
        self.update = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length):
        """训练模型"""
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, pm.keep_prob, False)
        output_feed = [self.update, self.loss, self.current_learning_rate, self.global_step]
        _, cost, lr, global_step = sess.run(output_feed, input_feed)
        return cost, lr, global_step

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        """预测"""
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      None, None, 1.0, True)
        pred = sess.run(self.decoder_pred_decode, input_feed)
        return pred[0]

    def save(self, sess, global_step, save_path=pm.save_path):
        """保存模型"""
        self.saver.save(sess, save_path=save_path, global_step=global_step)

    def load(self, sess, save_path=pm.save_path):
        """加载模型"""
        save_path = tf.train.latest_checkpoint(save_path)
        self.saver.restore(sess, save_path=save_path)
