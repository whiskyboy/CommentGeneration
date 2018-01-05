#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import os
import sys
import argparse

import numpy as np
import tensorflow as tf

from corpus import TrainCorpus, InferCorpus
from model import NMTModel as CGModel

def argParser():
    parser =argparse.ArgumentParser()

    # network
    parser.add_argument("--enc_cell_size", dest="enc_cell_szie", type=int, default=200)
    parser.add_argument("--dec_cell_size", dest="dec_cell_size", type=int, default=200)
    parser.add_argument("--emb_size", dest="emb_size", type=int, default=200)

    # data
    parser.add_argument("--out_dir", dest="out_dir", type=str, required=True)
    parser.add_argument("--train_corpus", dest="train_corpus_path", type=str, default=None)
    parser.add_argument("--dev_corpus", dest="dev_corpus_path", type=str, default=None)
    parser.add_argument("--test_corpus", dest="test_corpus_path", type=str, default=None)
    parser.add_argument("--infer_input", dest="infer_input_path", type=str, default=None)
    parser.add_argument("--infer_output", dest="infer_output_path", type=str, default=None)

    # data-config
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=4000)
    parser.add_argument("--max_token_len", dest="max_token_len", type=int, default=40)

    # others
    parser.add_argument("--max_pass", dest="max_pass", type=int, default=20)
    parser.add_argument("--init_w", dest="init_w", type=float, default=0.08)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_clip", dest="grad_clip", type=float, default=None)

    FLAGS, _ = parser.parse_known_args()

    return FLAGS


def run(FLAGS):
    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)

    if FLAGS.infer_input_path is None:
        # train mode
        corpus = TrainCorpus(FLAGS.train_corpus_path, FLAGS.dev_corpus_path, FLAGS.test_corpus_path,
                             FLAGS.out_dir, FLAGS.vocab_size, FLAGS.max_token_len)
        with tf.Session() as sess:
            initializer = tf.random_uniform_initializer(-1.0 * FLAGS.init_w, FLAGS.init_w)
            scope = "model"
            with tf.variable_scope(scope, reuse=None, initializer=initializer):
                model = CGModel(sess, scope, FLAGS,
                                len(corpus.src_vocab), len(corpus.tgt_vocab),
                                src_padding_id=corpus.src_rev_vocab["<pad>"], tgt_padding_id=corpus.tgt_rev_vocab["<pad>"],
                                src_go_id=corpus.src_rev_vocab["<s>"], tgt_go_id=corpus.tgt_rev_vocab["<s>"],
                                src_eos_id=corpus.src_rev_vocab["</s>"], tgt_eos_id=corpus.tgt_rev_vocab["</s>"],
                                forward_only=False)
            with tf.variable_scope(scope, reuse=True, initializer=initializer):
                infer_model = CGModel(sess, scope, FLAGS,
                                      len(corpus.src_vocab), len(corpus.tgt_vocab),
                                      src_padding_id=corpus.src_rev_vocab["<pad>"],
                                      tgt_padding_id=corpus.tgt_rev_vocab["<pad>"],
                                      src_go_id=corpus.src_rev_vocab["<s>"], tgt_go_id=corpus.tgt_rev_vocab["<s>"],
                                      src_eos_id=corpus.src_rev_vocab["</s>"], tgt_eos_id=corpus.tgt_rev_vocab["</s>"],
                                      forward_only=True)
            sess.run(tf.global_variables_initializer())

            dm_checkpoint_path = os.path.join(FLAGS.out_dir, model.__class__.__name__+ ".ckpt")
            best_dev_loss = np.inf
            for _pass in range(FLAGS.max_pass):
                print(">> Pass %d" % _pass)
                corpus.train_feed.initialize(batch_size=FLAGS.batch_size, shuffle=True)
                train_loss = model.train(corpus.train_feed)
                print("Training Loss: %.4f" % train_loss)

                corpus.dev_feed.initialize(batch_size=FLAGS.batch_size, shuffle=True)
                valid_loss = model.valid(corpus.dev_feed)
                print("Valid Loss: %.4f" % valid_loss)

                corpus.test_feed.initialize(batch_size=FLAGS.batch_size, shuffle=False)
                infer_model.infer(corpus.test_feed, corpus.tgt_vocab)

                if valid_loss < best_dev_loss:
                    model.saver.save(sess, dm_checkpoint_path, global_step=_pass)
                    best_dev_loss = valid_loss

            print("Best validation loss %f" % best_dev_loss)
            print("Done training")
    else:
        # infer mode
        corpus = InferCorpus(FLAGS.infer_input_path, FLAGS.out_dir, FLAGS.max_token_len)
        with tf.Session() as sess:
            scope = "model"
            with tf.variable_scope(scope, reuse=None):
                infer_model = CGModel(sess, scope, FLAGS,
                                      len(corpus.src_vocab), len(corpus.tgt_vocab),
                                      src_padding_id=corpus.src_rev_vocab["<pad>"],
                                      tgt_padding_id=corpus.tgt_rev_vocab["<pad>"],
                                      src_go_id=corpus.src_rev_vocab["<s>"], tgt_go_id=corpus.tgt_rev_vocab["<s>"],
                                      src_eos_id=corpus.src_rev_vocab["</s>"], tgt_eos_id=corpus.tgt_rev_vocab["</s>"],
                                      forward_only=True)

            ckpt = tf.train.get_checkpoint_state(FLAGS.out_dir)
            infer_model.saver.restore(sess, ckpt.model_checkpoint_path)

            corpus.infer_feed.initialize(batch_size=FLAGS.batch_size, shuffle=False)
            infer_model.infer(corpus.infer_feed, corpus.tgt_vocab, dest=open(FLAGS.infer_output_path, 'w'))

            print("Done inference")

if __name__ == "__main__":
    FLAGS = argParser()
    run(FLAGS)
