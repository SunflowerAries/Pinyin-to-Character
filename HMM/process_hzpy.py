# coding: utf-8

import codecs, util

SOURCE_FILE           = 'hanzipinyin.txt'

ALL_STATES_FILE       = 'data/all_states.txt'         # 汉字（隐藏状态）
ALL_OBSERVATIONS_FILE = 'data/all_observations.txt'   # 拼音（观测值）
PINYIN2HANZI_FILE     = 'data/pinyin2hanzi.txt'

states = set()
observations = set()
py2hz = {}

with codecs.open(SOURCE_FILE, 'r', 'utf-8') as fin:
    while True:
        line = fin.readline().strip()
        if not line: break
        hanzi, pinyin_list = line.split('=')
        pinyin_list = [util.simplify_pinyin(item.strip()) for item in pinyin_list.split(',')]

        states.add(hanzi)

        for pinyin in pinyin_list:
            observations.add(pinyin)
            py2hz.setdefault(pinyin, set())
            py2hz[pinyin].add(hanzi)
            # 声母
            shengmu = util.get_shengmu(pinyin)
            if shengmu is not None:
                py2hz.setdefault(shengmu, set())
                py2hz[shengmu].add(hanzi)

with codecs.open(ALL_STATES_FILE, 'w', 'utf-8') as fout:
    s = '\n'.join(states)
    fout.write(s)

with codecs.open(ALL_OBSERVATIONS_FILE, 'w', 'utf-8') as fout:
    s = '\n'.join(observations)
    fout.write(s)

with codecs.open(PINYIN2HANZI_FILE, 'w', 'utf-8') as fout:
    s = ''
    for k in py2hz:
        s = s + k + '=' + ''.join(py2hz[k]) + '\n'
    fout.write(s)

print('end')