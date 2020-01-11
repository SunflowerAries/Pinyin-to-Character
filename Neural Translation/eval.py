from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
from prepro import *
# from data_load import load_vocab, load_test_data, load_test_string
from data_load import load_vocab_json, load_test_data, load_test_string
from train import Graph
import codecs
import distance
import os



#Evaluate on testing batches
def main_batches():  
    g = Graph(is_training=False)
    
    # Load data
    xs, ys = load_test_data()
    # pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab()
    pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab_json()
    
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")

            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            
            with codecs.open('data/{}_{}.csv'.format(mname, "qwerty" if hp.isqwerty else "nine"), 'w', 'utf-8') as fout:

                total_edit_distance, num_chars = 0, 0
                trupred = 0
                for step in range(len(xs)//hp.batch_size):
                    x = xs[step*hp.batch_size:(step+1)*hp.batch_size] # input batch
                    y = ys[step*hp.batch_size:(step+1)*hp.batch_size] # batch of ground truth strings
                    
                    preds = sess.run(g.preds, {g.x: x})
                    for xx, pred, expected in zip(x, preds, y): # sentence-wise
                        #got = "".join(idx2hanzi[idx] for idx in pred)[:np.count_nonzero(xx)].replace("_", "")
                        got = "".join(idx2hanzi[str(idx)] for idx in pred)[:np.count_nonzero(xx)].replace("_", "")
                        expected = expected.replace("_", "")
                        # print(expected, got)
                        edit_distance = distance.levenshtein(expected, got)
                        total_edit_distance += edit_distance
                        num_chars += len(expected)
                        fout.write(u"{},{},{},{}\n".format(expected, got, len(expected), edit_distance))
                        # if edit_distance / len(expected) < 0.1:
                        if edit_distance == 0:
                            trupred += 1
                print(trupred / len(xs))
                fout.write(u"Total CER: {}/{}={},,,,\n".format(total_edit_distance,
                                                        num_chars, 
                                                        round(float(total_edit_distance)/num_chars, 2)))
                                

#For user input test                
def main():  
    g = Graph(is_training=False, have_input=True)
    
    # Load vocab
    # pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab()
    pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab_json()
    
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")

            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            while True:
                line = input("请输入测试拼音：")
                if len(line) > hp.maxlen:
                    print('最长拼音不能超过50')
                    continue
                x = load_test_string(pnyn2idx, line)
                #print(x)
                # preds = sess.run(g.preds, {g.x: x})
                # preds = sess.run(g.recommend, {g.x: x})
                # print(preds)
                # for i in range(5):
                #     if i < len(preds):
                #         gots.append("".join(idx2hanzi[str(idx)] for idx in preds[i])[:np.count_nonzero(x[0])].replace("_", ""))
                # print(gots)
                recommend = sess.run(g.recommend, {g.x: x})
                # got = "".join(idx2hanzi[str(idx)] for idx in preds[0])[:np.count_nonzero(x[0])].replace("_", "")
                for i in range(hp.recommend):
                    got = "".join(idx2hanzi[str(idx)] for idx in recommend[i][0])[:np.count_nonzero(x[0])].replace("_", "")
                    print(got)
                #got = "".join(idx2hanzi[str(idx)] for idx in preds[0])[:np.count_nonzero(x[0])].replace("_", "")

                                                                                                   
if __name__ == '__main__':
    main_batches()
    # main()
    print("Done")

