import codecs
trainset, validset = [], []
with codecs.open("data/zh_shuf.tsv", 'r', 'utf-8') as tsv:
    i = 0
    while True:
        line = tsv.readline()
        if not line: break
        _, pnyn_sent, hanzi_sent = line.strip().split("\t")
        tmp = hanzi_sent.replace("_", "")
        if len(tmp) < 2 or len(tmp) > 33:
            continue
        i += 1
        if i <= 400000:
            trainset.append((pnyn_sent, hanzi_sent))
        else:
            validset.append((pnyn_sent, hanzi_sent))
trainset = sorted(trainset, key=lambda x: len(x[0]))
validset = sorted(validset, key=lambda x: len(x[0]))
with codecs.open("data/train.tsv", 'w', 'utf-8') as train:
    i = 1
    for py, hz in trainset:
        train.write(u"{}\t{}\t{}\n".format(i, py, hz))
        i += 1
with codecs.open("data/valid.tsv", 'w', 'utf-8') as valid:
    i = 1
    for py, hz in validset:
        valid.write(u"{}\t{}\t{}\n".format(i, py, hz))
        i += 1
