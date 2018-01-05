#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

class NMTModel(object):
    def __init__(self, sess, scope, FLAGS,
                 src_vocab_size, tgt_vocab_size,
                 src_padding_id=0, tgt_padding_id=0,
                 src_go_id=2, tgt_go_id=2,
                 src_eos_id=3, tgt_eos_id=3,
                 forward_only=False):
        self.sess = sess
        self.scope = scope
        self.FLAGS = FLAGS

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_padding_id = src_padding_id
        self.tgt_padding_id = tgt_padding_id
        self.src_go_id = src_go_id
        self.tgt_go_id = tgt_go_id
        self.src_eos_id = src_eos_id
        self.tgt_eos_id = tgt_eos_id

        self.forward_only = forward_only

        self.max_token_len = FLAGS.max_token_len
        self.enc_cell_size = FLAGS.enc_cell_size
        self.dec_cell_size = FLAGS.dec_cell_size

        with tf.name_scope("io"):
            self.input_tokens = tf.placeholder(dtype=tf.int32, shape=(None, self.max_token_len), name="source")
            self.input_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="source_lens")

            if not self.forward_only:
                self.output_tokens = tf.placeholder(dtype=tf.int32, shape=(None, None), name="target")
                self.output_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="target_lens")

            self.learning_rate = FLAGS.learning_rate

            self.batch_size = array_ops.shape(self.input_tokens)[0]

        with tf.variable_scope("wordEmbedding"):
            src_embedding = tf.get_variable("src_embedding", [self.src_vocab_size, FLAGS.emb_size], dtype=tf.float32)
            # mask <pad> embedding
            src_embedding_mask = tf.constant([0 if i == self.src_padding_id else 1 for i in range(self.src_vocab_size)],
                                             dtype=tf.float32, shape=[self.src_vocab_size, 1])
            src_embedding = src_embedding * src_embedding_mask

            tgt_embedding = tf.get_variable("tgt_embedding", [self.tgt_vocab_size, FLAGS.emb_size], dtype=tf.float32)
            # mask <pad> embedding
            tgt_embedding_mask = tf.constant([0 if i == self.tgt_padding_id else 1 for i in range(self.tgt_vocab_size)],
                                             dtype=tf.float32, shape=[self.tgt_vocab_size, 1])
            tgt_embedding = tgt_embedding * tgt_embedding_mask

            input_embedding = tf.nn.embedding_lookup(src_embedding, self.input_tokens)
            output_embedding = tf.nn.embedding_lookup(tgt_embedding, self.output_tokens)

        with tf.variable_scope("encoder"):
            # bi-gru-rnn
            fwd_sent_cell = self.get_rnncell("gru", self.enc_cell_size, keep_prob=1.0, num_layer=1)
            bwd_sent_cell = self.get_rnncell("gru", self.enc_cell_size, keep_prob=1.0, num_layer=1)

            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                fwd_sent_cell, bwd_sent_cell,
                input_embedding, sequence_length=self.input_lens,
                dtype=tf.float32)

        with tf.variable_scope("decoder"):
            if self.forward_only:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    tgt_embedding, tf.fill([self.batch_size], self.tgt_go_id), self.tgt_eos_id)
            else:
                dec_input_embedding = output_embedding[:, 0:-1, :]
                dec_seq_lens = self.output_lens - 1
                helper = tf.contrib.seq2seq.TrainingHelper(dec_input_embedding, dec_seq_lens)

            attention_states = encoder_outputs
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.dec_cell_size, attention_states, memory_sequence_length=self.input_lens)
            dec_cell = self.get_rnncell("gru", self.dec_cell_size, keep_prob=1.0, num_layer=1)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                dec_cell, attention_mechanism,
                output_attention=False
            )
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, self.tgt_vocab_size)

            # dec_init_state must be a instance of AttentionWrapperState Class
            dec_init_state = out_cell.zero_state(self.batch_size, dtype=tf.float32).clone(
                cell_state=encoder_state)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper, initial_state=dec_init_state)

            # Dynamic decoding
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=self.max_token_len)

            dec_outs = outputs.rnn_output
            self.dec_out_words = outputs.sample_id

        if not self.forward_only:
            with tf.variable_scope("loss"):
                labels = self.output_tokens[:, 1:]
                dec_output_embedding = output_embedding[:, 1:, :]
                label_mask = tf.to_float(tf.sign(tf.reduce_max(tf.abs(dec_output_embedding), reduction_indices=2)),reduction_indices=1)

                rc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs, labels=labels)
                rc_loss = tf.reduce_sum(rc_loss * label_mask, reduction_indices=1)
                self.avg_rc_loss = tf.reduce_mean(rc_loss)

            self.optimize(self.avg_rc_loss)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    def optimize(self, loss):
        # optimization
        if self.scope is None:
            tvars = tf.trainable_variables()
        else:
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        grads = tf.gradients(loss, tvars)
        if self.FLAGS.grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, tf.constant(self.FLAGS.grad_clip))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.train_ops = optimizer.apply_gradients(zip(grads, tvars))

    def get_rnncell(self, cell_type, cell_size, keep_prob, num_layer):
        if cell_type == "gru":
            cell = tf.nn.rnn_cell.GRUCell(cell_size)
        else:
            cell = tf.nn.rnn_cell.LSTMCell(cell_size, use_peepholes=False, forget_bias=1.0)

        if keep_prob < 1.0:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

        if num_layer > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layer, state_is_tuple=True)

        return cell

    def train(self, train_feed):
        train_loss = None
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break

            src_tokens, src_token_lens, tgt_tokens, tgt_token_lens = batch
            feed_dict = {self.input_tokens: src_tokens, self.input_lens: src_token_lens,
                     self.output_tokens: tgt_tokens, self.output_lens: tgt_token_lens}

            _, avg_rc_loss = self.sess.run(
                [self.train_ops, self.avg_rc_loss],
                feed_dict)
            train_loss = avg_rc_loss

        return train_loss

    def valid(self, valid_feed):
        rc_losses = []

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break

            src_tokens, src_token_lens, tgt_tokens, tgt_token_lens = batch
            feed_dict = {self.input_tokens: src_tokens, self.input_lens: src_token_lens,
                         self.output_tokens: tgt_tokens, self.output_lens: tgt_token_lens}

            avg_rc_loss, = self.sess.run([self.avg_rc_loss], feed_dict)
            rc_losses.append(avg_rc_loss)

        return np.mean(rc_losses)

    def infer(self, test_feed, tgt_vocab, dest=sys.stdout):
        while True:
            batch = test_feed.next_batch()
            if batch is None:
                break

            src_tokens, src_token_lens = batch
            feed_dict = {self.input_tokens: src_tokens, self.input_lens: src_token_lens}

            word_outs, = self.sess.run([self.dec_out_words], feed_dict)

            for b_id in range(test_feed.batch_size):
                gen_str = " ".join([tgt_vocab[e] for e in word_outs[b_id].tolist() if e not in [self.tgt_padding_id, self.tgt_eos_id, self.tgt_go_id]])
                dest.write("%s\n" % gen_str)