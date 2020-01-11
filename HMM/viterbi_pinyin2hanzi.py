# coding: utf-8

from implement import DefaultHmmParams
from viterbi import viterbi
from PnynSplit import SplitPinyin
import distance, codecs, regex, _codecs

hmmparams = DefaultHmmParams()
split = SplitPinyin()

try:
    lines = [line for line in codecs.open('data/valid.tsv', 'r', 'utf-8').read().splitlines()]
except IOError:
    raise IOError("Write the sentences you want to test line by line in `data/input.csv` file.")

xs, ys = [], []
total_edit_distance, num_chars, trupred = 0, 0, 0
with codecs.open('data/viterbi.csv', 'w', 'utf-8') as fout:
    for line in lines:
        if "〡" in line:
            continue
        i, pnyn_sent, hanzi_sent = line.strip().split('\t')
        if len(pnyn_sent) > 33:
            break
        hanzi_sent = regex.sub(u"[_《》“”]", r"", hanzi_sent)
        pnyn_sent = regex.sub(u"[_《》“”]", r"", pnyn_sent)
        if hanzi_sent[-1] in ['，', '：', '？', '！', '。']:
            hanzi_sent = hanzi_sent[:-1]
        if pnyn_sent[-1] in ['，', '：', '？', '！', '。']:
            pnyn_sent = pnyn_sent[:-1]
        pnyn, _, _ = split.split_pinyin(pnyn_sent)
        print(pnyn)
        result = viterbi(hmm_params=hmmparams, observations=pnyn, path_num = 1, log = True)
        predict = ''.join(result[0].path)
        print(predict)
        edit_distance = distance.levenshtein(hanzi_sent, predict)
        total_edit_distance += edit_distance
        num_chars += len(hanzi_sent)
        fout.write(u"{},{},{},{}\n".format(hanzi_sent, predict, len(hanzi_sent), edit_distance))
        if edit_distance == 0:
            trupred += 1
    print(trupred / len(xs))
    fout.write(u"Total CER: {}/{}={},,,,\n".format(total_edit_distance,
                                                   num_chars,
                                                   round(float(total_edit_distance) / num_chars, 2)))
