# coding: utf-8

import json, codecs, regex, util
from xpinyin import Pinyin

SENTENCE_FILE     = 'data/train.tsv'
WORD_FILE         = 'word.txt'
HANZI2PINYIN_FILE = 'hanzipinyin.txt'
PINYIN2HANZI_FILE = 'pinyinhanzi.txt'
BASE_START      = 'data/base_start.json'
BASE_EMISSION   = 'data/base_emission.json'
BASE_TRANSITION = 'data/base_transition.json'

def writejson2file(data, filename):
    json.dump(data, open(filename, 'w'))

def process_hanzipinyin(emission):
    ## ./hanzipinyin.txt
    print('read from hanzipinyin.txt')
    with codecs.open(HANZI2PINYIN_FILE, 'r', 'utf-8') as fin:
        with codecs.open("data/dictionary.txt", 'w', 'utf-8') as fout:
            while True:
                line = fin.readline()
                if not line: break
                line = line.strip()
                if '=' not in line:
                    continue
                hanzi, pinyins = line.split('=')
                pinyins = pinyins.split(',')
                pinyins = [util.simplify_pinyin(py) for py in pinyins]
                pnyn = ""
                for i in range(len(pinyins)):
                    if i != len(pinyins) - 1:
                        pnyn += pinyins[i] + ","
                    else:
                        pnyn += pinyins[i]
                fout.write(u"{}\t{}\n".format(hanzi, pnyn))
                for pinyin in pinyins:
                    emission.setdefault(hanzi, {})
                    emission[hanzi].setdefault(pinyin, 0)
                    emission[hanzi][pinyin] += 1

def read_from_sentence_txt(start, emission, transition):
    ## ./result/sentence.txt
    print('read from sentence.txt')
    with codecs.open(SENTENCE_FILE, 'r', 'utf-8') as fin:
        while True:
            line = fin.readline()
            if not line: break
            line = regex.sub(u"[_《》“”]", r"", line.strip().split('\t')[2])
            if line[-1] in ['，', '：', '？', '！', '。']:
                line = line[:-1]
            if len(line) < 2:
                continue
            ## for start
            start.setdefault(line[0], 0)
            start[line[0]] += 1

            ## for emission
            pinyin = Pinyin()
            pnyns = pinyin.get_pinyin(line, " ").split()
            hanzis = [c for c in line]
            # print(pnyns, hanzis)

            for hanzi, pinyin in zip(hanzis, pnyns):
                emission.setdefault(hanzi, {})
                emission[hanzi].setdefault(pinyin, 0)
                emission[hanzi][pinyin] += 1

            ## for transition
            for f, t in zip(line[:-1], line[1:]):
                transition.setdefault(f, {})
                transition[f].setdefault(t, 0)
                transition[f][t] += 1

def read_from_word_txt(start, emission, transition):
    ## ! 基于word.txt的优化
    print('read from word.txt')
    _base = 1000.
    _min_value = 2.
    with codecs.open(WORD_FILE, 'r', 'utf-8') as fin:
        while True:
            line = fin.readline()
            if not line: break
            if '=' not in line:
                continue
            if len(line) < 3:
                continue
            ls = line.split('=')
            if len(ls) != 2:
                continue
            word, num = ls
            word = word.strip()
            num  = num.strip()
            if len(num) == 0:
                continue
            num = float(num)
            num = max(_min_value, num/_base)

            ## for start
            start.setdefault(word[0], 0)
            start[word[0]] += num

            ## for emission
            pinyin = Pinyin()
            pnyns = pinyin.get_pinyin(word, " ").split()
            words = [c for c in word]
            for hanzi, pinyin in zip(words, pnyns):
                emission.setdefault(hanzi, {})
                emission[hanzi].setdefault(pinyin, 0)
                emission[hanzi][pinyin] += num

            ## for transition
            for f, t in zip(word[:-1], word[1:]):
                transition.setdefault(f, {})
                transition[f].setdefault(t, 0)
                transition[f][t] += num


def gen_base():
    """ 先执行gen_middle()函数 """
    start      = {}     # {'你':2, '号':1}
    emission   = {}     # 应该是 {'泥': {'ni':1.0}, '了':{'liao':0.5, 'le':0.5}}  而不是 {'ni': {'泥': 2, '你':10}, 'hao': {...} } × 
    transition = {}     # {'你': {'好':10, '们':2}, '我': {}}

    process_hanzipinyin(emission)
    
    read_from_sentence_txt(start, emission, transition)
    read_from_word_txt(start, emission, transition)

    ## write to file
    writejson2file(start, BASE_START)
    writejson2file(emission, BASE_EMISSION)
    writejson2file(transition, BASE_TRANSITION)

if __name__ == '__main__':
    gen_base()
