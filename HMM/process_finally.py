# coding: utf-8

import json, codecs

BASE_START_FILE      = 'data/base_start.json'
BASE_EMISSION_FILE   = 'data/base_emission.json'
BASE_TRANSITION_FILE = 'data/base_transition.json'

ALL_STATES_FILE       = 'data/all_states.txt'   # 所有的字
ALL_OBSERVATIONS_FILE = 'data/all_observations.txt' # 所有的拼音
PY2HZ_FILE            = 'data/pinyin2hanzi.txt'

HZ2PY_FILE            = 'hanzipinyin.txt'

FIN_PY2HZ_FILE      = 'data/hmm_py2hz.json'
FIN_START_FILE      = 'data/hmm_start.json'
FIN_EMISSION_FILE   = 'data/hmm_emission.json'
FIN_TRANSITION_FILE = 'data/hmm_transition.json'


PINYIN_NUM = 411.
HANZI_NUM  = 20903.

def writejson2file(obj, filename):
    json.dump(obj, open(filename, 'w'), sort_keys=True)

def readdatafromfile(filename):
    return json.load(open(filename, 'r'))

def gen_py2hz():
    data = {}
    with codecs.open(PY2HZ_FILE, 'r', 'utf-8') as fin:
        while True:
            line = fin.readline()
            if not line: break
            line = line.strip()
            ls = line.split('=')
            if len(ls) != 2:
                raise Exception('invalid format')
            py, chars = ls
            py = py.strip()
            chars = chars.strip()
            if len(py)>0 and len(chars)>0:
                data[py] = chars

    writejson2file(data, FIN_PY2HZ_FILE)

def gen_start():
    data = {'default': 1, 'data': None}
    start = readdatafromfile(BASE_START_FILE)
    count = HANZI_NUM
    for hanzi in start:
        count += start[hanzi]

    for hanzi in start:
        start[hanzi] = start[hanzi] / count

    data['default'] = 1.0 / count
    data['data'] = start

    writejson2file(data, FIN_START_FILE)

def gen_emission():
    """
    base_emission   = {} #>   {'泥': {'ni':1.0}, '了':{'liao':0.5, 'le':0.5}}
    """
    data = {'default': 1.e-200, 'data': None}
    emission = readdatafromfile(BASE_EMISSION_FILE)

    for hanzi in emission:
        num_sum = 0.
        for pinyin in emission[hanzi]:
            num_sum += emission[hanzi][pinyin]
        for pinyin in emission[hanzi]:
            emission[hanzi][pinyin] = emission[hanzi][pinyin] / num_sum

    data['data'] = emission
    writejson2file(data, FIN_EMISSION_FILE)

def gen_tramsition():
    """  
    {'你': {'好':10, '们':2}, '我': {}}
    """
    data = {'default': 1./HANZI_NUM, 'data': None}
    transition = readdatafromfile(BASE_TRANSITION_FILE)
    for c1 in transition:
        num_sum = HANZI_NUM # 默认每个字都有机会
        for c2 in transition[c1]:
            num_sum += transition[c1][c2]

        for c2 in transition[c1]:
            transition[c1][c2] = float(transition[c1][c2]+1) / num_sum
        transition[c1]['default'] = 1./num_sum

    data['data'] = transition
    writejson2file(data, FIN_TRANSITION_FILE)

def main():
    gen_py2hz()
    gen_start()
    gen_emission()
    gen_tramsition()

if __name__ == '__main__':
    main()