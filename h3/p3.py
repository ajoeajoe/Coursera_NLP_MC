# -*- coding:utf-8 -*-
# Filename: p3.py
# Author：hankcs
# Date: 2017-01-31 PM4:40
from collections import defaultdict
from p12 import IBM1, IBM2, IBM
import math


class Phrase:
    def __init__(self):
        self.all_f = defaultdict(list)  # all_f[sent_id] = list((e, f))
        self.all_e = defaultdict(list)  # all_e[sent_id] = list((e, f))
        self.insec = defaultdict(list)  # insec[sent_id] = list(intersection of all_f and all_e)
        self.union = defaultdict(list)  # union[sent_id] = list(union of all_f and all_e)
        self.sym_dif = defaultdict(list)
        self.grow = defaultdict(list)

    def read_alignments(self):
        a1 = file("dev.ibm1.out", "r")
        a2 = file("dev.ibm2.out", "r")

        for line in a1:
            line = line.split(" ")
            line[-1] = line[-1].strip()
            self.all_f[int(line[0])].append((int(line[1]), int(line[2])))

        for line in a2:
            line = line.split(" ")
            line[-1] = line[-1].strip()
            self.all_e[int(line[0])].append((int(line[1]), int(line[2])))

    def intersection(self):
        for key in self.all_f:
            s1 = set(self.all_f[key]).intersection(set(self.all_e[key]))
            self.insec[key] = list(s1)

    def uni(self):
        for key in self.all_f:
            s1 = set(self.all_f[key]).union(set(self.all_e[key]))
            self.union[key] = list(s1)

    def symmetric_dif(self):
        for key in self.all_f:
            s1 = set(self.all_f[key]).symmetric_difference(set(self.all_e[key]))
            self.sym_dif[key] = list(s1)

    def distance(self, pair, done):
        """
        treat pairs as points, return distance to closest neighbour of pair in done
        :param pair:
        :param done:
        :return:
        """
        d_v = []
        for ef in done:
            d = math.pow(pair[0] - ef[0], 2) + math.pow(pair[1] - ef[1], 2)
            d_v.append(d)

        return min(d_v)

    def growing(self):
        self.uni()
        self.intersection()
        self.grow = self.insec.copy()

        for sent in self.grow:
            u = self.union[sent]
            for ef in u:
                fdone = set([k[1] for k in self.grow[sent]])
                edone = set([k[0] for k in self.grow[sent]])
                if ef[1] not in fdone or ef[0] not in edone:
                    if self.distance(ef, self.grow[sent]) <= 9:
                        self.grow[sent].append(ef)

            d1 = [k[0] for k in self.all_e[sent]]
            d2 = [k[1] for k in self.all_f[sent]]
            d = math.fabs(max(d1) - max(d2)) + 3

            self.grow[sent] = [v for v in self.grow[sent]
                               if math.fabs(v[0] - v[1]) < (d + v[1] / 4)]


def output_dev_file(alignments, output_file):
    output = file(output_file, "w")
    for key in alignments:
        val = alignments[key]
        for v in val:
            output.write("%s %s %s\n" % (key, v[0], v[1]))


if __name__ == "__main__":
    model = IBM1()
    mt = IBM(model)
    mt.read_file("corpus.es", "corpus.en")
    mt.EM()

    model = IBM2()
    mt = IBM(model)
    mt.read_file("corpus.es", "corpus.en")
    mt.EM()
    mt.read_dev_file("dev.es", "dev.en", "dev.ibm2r.out", invert=True)

    p = Phrase()
    p.read_alignments()
    p.growing()
    output_dev_file(p.grow, "dev.phase.out")
