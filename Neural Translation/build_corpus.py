
"""
Before running this code, make sure that you've downloaded Leipzig Chinese Corpus 
(http://corpora2.informatik.uni-leipzig.de/downloads/zho_news_2007-2009_1M-text.tar.gz)
Extract and copy the `zho_news_2007-2009_1M-sentences.txt` to `data/` folder.

This code should generate a file which looks like this:
2[Tab]zhegeyemianxianzaiyijingzuofei...。[Tab]这__个_页_面___现___在__已_经___作__废__...。

In each line, the id, pinyin, and a chinese sentence are separated by a tab.
Note that _ menist@gmail.com
"""
from __future__ import print_function
import codecs
import os
import regex # pip install regex
from xpinyin import Pinyin # pip install xpinyin
import matplotlib.pyplot as plt
from contextlib import ExitStack

length, number = [], []
maxlen = 200
minlen = 0
py, hz = [], []

def align(sent):
    '''
    Args:
      sent: A string. A sentence.
    
    Returns:
      A tuple of pinyin and chinese sentence.
    '''
    pys, hzs = [], []
    sents = regex.sub(u"(?<=([。，！？：]))", r"|", sent).split("|")
    for sent in sents:
        if len(sent) == 0:
            continue
        pinyin = Pinyin()
        pnyns = pinyin.get_pinyin(sent, " ").split()
        hanzis = []
        for char, p in zip(sent.replace(" ", ""), pnyns):
            hanzis.extend([char] + ["_"] * (len(p) - 1))
            
        pnyns = "".join(pnyns)
        hanzis = "".join(hanzis)
        assert len(pnyns) == len(hanzis), "The hanzis and the pinyins must be the same in length."
        pys.append(pnyns)
        hzs.append(hanzis)
    return pys, hzs, sents

def clean(text):
    text = regex.findall("_!_([\p{Han}。，！？：“”《》]+)_!_", text)[0]
    if regex.search("[A-Za-z0-9]", text) is not None: # For simplicity, roman alphanumeric characters are removed.
        return ""
    return text
    
def build_corpus():
    global py, hz, maxlen, minlen, number
    for i in range(maxlen):
        number.append(0)
    with codecs.open("data/zh.tsv", 'w', 'utf-8') as fout:
        with codecs.open("data/toutiao.txt", 'r', 'utf-8') as fin:
            i = 1
            while True:
                line = fin.readline()
                if not line: break
                try:
                    tmp = clean(line)
                    if len(tmp) > 0:
                        pnyns, hanzis, sents = align(tmp)
                        py += pnyns
                        hz += hanzis
                        i += 1
                        if i % 10000 == 0:
                            print(i)
                        for sent in sents:
                            if len(sent) > 0:
                                number[len(sent)] += 1
                except:
                    continue  # it's okay as we have a pretty big corpus!
            py2hz = list(zip(py, hz))
            # py2hz = sorted(py2hz, key=lambda x: len(x[0]))
            i = 1
            for py, hz in py2hz:
                fout.write(u"{}\t{}\t{}\n".format(i, py, hz))
                i += 1
            with codecs.open("data/number.tsv", 'w', 'utf-8') as f:
                for j in range(len(number)):
                    f.write(u"{}\t".format(number[j]))
            for j in range(maxlen):
                if (number[maxlen - 1 - j] > 63):
                    maxlen = maxlen - j
                    break
            for j in range(maxlen):
                if (number[j] > 63):
                    minlen = j
                    break
            print(sum(number), i - 1)
            number = number[minlen:maxlen]
            for j in range(maxlen - minlen):
                length.append(j + 1 + minlen)
            plt.bar(range(len(number)), number, tick_label=length)
            for a, b in zip(range(len(number)), number):
                plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=7)
            print(sum(number))
            plt.show()

if __name__ == "__main__":
    build_corpus(); print("Done")
