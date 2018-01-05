from collections import Counter
import numpy as np
import os
import tensorflow as tf
import abc

SRC_VOCAB_FILENAME = "vocab.src"
TGT_VOCAB_FILENAME = "vocab.tgt"

class CorpusFeed(object):
    def __init__(self, corpus, max_token_len, src_rev_vocab, tgt_rev_vocab):
        self.max_token_len = max_token_len
        self.src_rev_vocab = src_rev_vocab
        self.tgt_rev_vocab = tgt_rev_vocab
        self.corpus = self._get_id_corpus(corpus)
        self.corpus_size = len(self.corpus)
        self.indexes = np.arange(0, self.corpus_size)

        self.is_init = False

    def _pad_to(self, tokens, pad_id=None, do_pad=True):
        if len(tokens) >= self.max_token_len:
            return tokens[0:self.max_token_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [pad_id] * (self.max_token_len-len(tokens))
        else:
            return tokens

    @abc.abstractmethod
    def _get_id_corpus(self, corpus):
        pass

    @abc.abstractmethod
    def _get_batch(self, batch_corpus):
        pass

    def initialize(self, batch_size=16, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ptr = 0
        self.num_batch = self.corpus_size // self.batch_size

        if shuffle:
            np.random.shuffle(self.indexes)

        self.is_init = True

    def next_batch(self):
        if not self.is_init:
            raise RuntimeError("Must call initialize method before calling next_batch method")

        if self.ptr < self.num_batch:
            cur_indexes = self.indexes[self.ptr * self.batch_size:(self.ptr + 1) * self.batch_size]
            cur_corpus = [self.corpus[idx] for idx in cur_indexes]
            cur_batch = self._get_batch(cur_corpus)
            self.ptr += 1
            return cur_batch
        else:
            return None

class TrainCorpusFeed(CorpusFeed):
    def _get_id_corpus(self, corpus):
        id_corpus = []
        src_unk_id = self.src_rev_vocab["<unk>"]
        src_pad_id = self.src_rev_vocab["<pad>"]
        tgt_unk_id = self.tgt_rev_vocab["<unk>"]
        tgt_pad_id = self.tgt_rev_vocab["<pad>"]
        for src, tgt in corpus:
            src = ["<s>"] + src + ["</s>"]
            src_ids = [self.src_rev_vocab.get(w, src_unk_id) for w in src]
            src_ids = self._pad_to(src_ids, pad_id=src_pad_id)

            tgt = ["<s>"] + tgt + ["</s>"]
            tgt_ids = [self.tgt_rev_vocab.get(w, tgt_unk_id) for w in tgt]
            tgt_ids = self._pad_to(tgt_ids, pad_id=tgt_pad_id)
            id_corpus.append((src_ids, tgt_ids))
        return id_corpus

    def _get_batch(self, batch_corpus):
        src_tokens, src_lens, tgt_tokens, tgt_lens = [], [], [], []
        for src_ids, tgt_ids in batch_corpus:
            src_tokens.append(src_ids)
            # since src_ids have padding, we need to remove the pad_id before calculating the src length
            src_ids_without_padding = filter(lambda x: x != self.src_rev_vocab["<pad>"], src_ids)
            src_lens.append(len(src_ids_without_padding))

            tgt_tokens.append(tgt_ids)
            # TODO: this should be padding length or no-padding length?
            #tgt_ids_without_padding = filter(lambda x: x != self.tgt_rev_vocab["<pad>"], tgt_ids)
            #tgt_lens.append(len(tgt_ids_without_padding))
            tgt_lens.append(len(tgt_ids))

        vec_src_lens = np.array(src_lens)
        vec_tgt_lens = np.array(tgt_lens)
        vec_src_tokens = np.zeros((self.batch_size, self.max_token_len), dtype=np.int32)
        vec_tgt_tokens = np.zeros((self.batch_size, self.max_token_len), dtype=np.int32)

        for b_id in range(self.batch_size):
            vec_src_tokens[b_id, :] = np.array(src_tokens[b_id])
            vec_tgt_tokens[b_id, :] = np.array(tgt_tokens[b_id])

        return vec_src_tokens, vec_src_lens, vec_tgt_tokens, vec_tgt_lens

class InferCorpusFeed(CorpusFeed):
    def _get_id_corpus(self, corpus):
        id_corpus = []
        src_unk_id = self.src_rev_vocab["<unk>"]
        src_pad_id = self.src_rev_vocab["<pad>"]
        for src in corpus:
            src = ["<s>"] + src + ["</s>"]
            src_ids = [self.src_rev_vocab.get(w, src_unk_id) for w in src]
            src_ids = self._pad_to(src_ids, pad_id=src_pad_id)
            id_corpus.append(src_ids)
        return id_corpus

    def _get_batch(self, batch_corpus):
        src_tokens, src_lens = [], []
        for src_ids in batch_corpus:
            src_tokens.append(src_ids)
            # since src_ids have padding, we need to remove the pad_id before calculating the src length
            src_ids_without_padding = filter(lambda x: x != self.src_rev_vocab["<pad>"], src_ids)
            src_lens.append(len(src_ids_without_padding))

        vec_src_lens = np.array(src_lens)
        vec_src_tokens = np.zeros((self.batch_size, self.max_token_len), dtype=np.int32)
        for b_id in range(self.batch_size):
            vec_src_tokens[b_id, :] = np.array(src_tokens[b_id])

        return vec_src_tokens, vec_src_lens

class TrainCorpus(object):
    def __init__(self, train_corpus_path, dev_corpus_path, test_corpus_path,
                 vocab_path=None, max_vocab_cnt=4000, max_token_len=40):
        train_corpus = self.__load_corpus(train_corpus_path)
        dev_corpus = self.__load_corpus(dev_corpus_path)
        test_corpus = self.__load_corpus(test_corpus_path)
        test_corpus = [src for src, tgt in test_corpus]
        self.__build_vocab(corpus=train_corpus, max_vocab_cnt=max_vocab_cnt, vocab_path=vocab_path)
        if vocab_path is not None:
            if not os.path.exists(vocab_path):
                os.mkdir(vocab_path)

            self.__dump_vocab(self.src_vocab, os.path.join(vocab_path, SRC_VOCAB_FILENAME))
            self.__dump_vocab(self.tgt_vocab, os.path.join(vocab_path, TGT_VOCAB_FILENAME))

        self.train_feed = TrainCorpusFeed(train_corpus, max_token_len, self.src_rev_vocab, self.tgt_rev_vocab)
        self.dev_feed = TrainCorpusFeed(dev_corpus, max_token_len, self.src_rev_vocab, self.tgt_rev_vocab)
        self.test_feed = InferCorpusFeed(test_corpus, max_token_len, self.src_rev_vocab, self.tgt_rev_vocab)

    def __load_corpus(self, corpus_path):
        corpus = []
        for line in open(corpus_path, 'r'):
            fields = line.strip().split("\t")
            if len(fields) != 2:
                continue
            src, tgt = fields
            corpus.append((src.split(" "), tgt.split(" ")))
        return corpus

    def __build_vocab(self, corpus, max_vocab_cnt, vocab_path=None):
        src_vocabs = []
        tgt_vocabs = []
        for src_text, tgt_text in corpus:
            src_vocabs.extend(src_text)
            tgt_vocabs.extend(tgt_text)

        def _cutoff_vocab(vocab):
            vocab_count = Counter(vocab).most_common() # word frequence
            vocab_count = vocab_count[0:max_vocab_cnt]
            vocab = ["<pad>", "<unk>", "<s>", "</s>"] + [t for t, cnt in vocab_count]
            rev_vocab = {t:idx for idx, t in enumerate(vocab)}

            return vocab, rev_vocab

        print("Building vocabulary")
        self.src_vocab, self.src_rev_vocab = _cutoff_vocab(src_vocabs)
        self.tgt_vocab, self.tgt_rev_vocab = _cutoff_vocab(tgt_vocabs)

    def __dump_vocab(self, vocab, filename):
        with open(filename, 'w') as fout:
            for w in vocab:
                fout.write(w + "\n")

class InferCorpus(object):
    def __init__(self, infer_input_path, vocab_path, max_token_len=40):
        infer_corpus = self.__load_corpus(infer_input_path)
        self.src_vocab, self.src_rev_vocab = self.__load_vocab(os.path.join(vocab_path, SRC_VOCAB_FILENAME))
        self.tgt_vocab, self.tgt_rev_vocab = self.__load_vocab(os.path.join(vocab_path, TGT_VOCAB_FILENAME))
        self.infer_feed = InferCorpusFeed(infer_corpus, max_token_len, self.src_rev_vocab, self.tgt_rev_vocab)

    def __load_corpus(self, corpus_path):
        corpus = []
        for line in open(corpus_path, 'r'):
            src = line.strip()
            corpus.append(src.split(" "))
        return corpus

    def __load_vocab(self, filename):
        vocab = [w.strip() for w in open(filename, 'r')]
        rev_vocab = {t:idx for idx, t in enumerate(vocab)}
        return vocab, rev_vocab

