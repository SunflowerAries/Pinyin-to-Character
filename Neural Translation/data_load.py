
'''
Data loading.
Note:
Nine key pinyin keyboard layout sample:

`      ABC   DEF
GHI    JKL   MNO
POQRS  TUV   WXYZ

'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import codecs
import numpy as np
import re

def load_vocab():
    import pickle
    if hp.isqwerty:
        return pickle.load(open('data/vocab.qwerty.pkl', 'rb'))
    else:
        return pickle.load(open('data/vocab.nine.pkl', 'rb'))

def load_vocab_json():
    import json
    if hp.isqwerty:
        return json.load(open('data/vocab.qwerty.json', 'r'))
    else:
        return json.load(open('data/vocab.nine.json', 'r'))


def load_train_data():
    '''Loads vectorized input training data'''
    # pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab()
    pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab_json()

    print("pnyn vocabulary size is", len(pnyn2idx))
    print("hanzi vocabulary size is", len(hanzi2idx))

    xs, ys = [], []
    for line in codecs.open('data/train.tsv', 'r', 'utf-8'):
        try:
            _, pnyn_sent, hanzi_sent = line.strip().split("\t")
        except ValueError:
            continue

        assert len(pnyn_sent)==len(hanzi_sent)
        x = [pnyn2idx.get(pnyn, 1) for pnyn in pnyn_sent] # 1: OOV
        y = [hanzi2idx.get(hanzi, 1) for hanzi in hanzi_sent] # 1: OOV
        xs.append(np.array(x, np.int32).tostring())
        ys.append(np.array(y, np.int32).tostring())

    return xs, ys

def load_test_data():
    '''Embeds and vectorize words in input corpus'''
    try:
        lines = [line for line in codecs.open('data/valid.tsv', 'r', 'utf-8').read().splitlines()[1:]]
    except IOError:
        raise IOError("Write the sentences you want to test line by line in `data/input.csv` file.")

    # pnyn2idx, _, hanzi2idx, _ = load_vocab()
    pnyn2idx, _, hanzi2idx, _ = load_vocab_json()

    xs, ys = [], [] # ys: ground truth (list of string)
    for line in lines:
        i, pnyn_sent, hanzi_sent = line.strip().split("\t")
        assert len(pnyn_sent) == len(hanzi_sent)
        x = [pnyn2idx.get(pnyn, 1) for pnyn in pnyn_sent]  # 1: OOV
        if len(x) > hp.maxlen:
            print(i)
            break
        x += [0] * (hp.maxlen - len(x))
        xs.append(x)
        ys.append(hanzi_sent)
    X = np.array(xs, np.int32)
    return X, ys

def load_test_string(pnyn2idx, test_string):
    '''Embeds and vectorize words in user input string'''
    pnyn_sent= test_string
    xs = []
    x = [pnyn2idx.get(pnyn, 1) for pnyn in pnyn_sent]
    x += [0] * (hp.maxlen - len(x))
    xs.append(x)
    X = np.array(xs, np.int32)
    return X


def get_batch():
    '''Makes batch queues from the training data.
    Returns:
      A Tuple of x (Tensor), y (Tensor).
      x and y have the shape [batch_size, maxlen].
    '''
    import tensorflow as tf

    # Load data
    X, Y = load_train_data()

    # Create Queues
    x, y = tf.train.slice_input_producer([tf.convert_to_tensor(X),
                                          tf.convert_to_tensor(Y)])

    x = tf.decode_raw(x, tf.int32)
    y = tf.decode_raw(y, tf.int32)

    x, y = tf.train.batch([x, y],
                          shapes=[(None,), (None,)],
                          num_threads=8,
                          batch_size=hp.batch_size,
                          capacity=hp.batch_size * 64,
                          allow_smaller_final_batch=False,
                          dynamic_pad=True)
    num_batch = len(X) // hp.batch_size

    return x, y, num_batch  # (N, None) int32, (N, None) int32, ()