from collections import defaultdict
import math, sys


class IBM1:
    def __init__(self):
        self.t = defaultdict(float)  # t[(fword, eword)] := t(f|e)

    def initialize(self, corpus):
        words = defaultdict(float)
        pairs = defaultdict(float)
        for e, f in corpus:
            for eword in e:
                for fword in f:
                    if (fword, eword) not in pairs:  # plus one smooth
                        words[eword] += 1.0
                        pairs[(fword, eword)] = 1.0

        for (fword, eword) in pairs:
            self.t[(fword, eword)] = pairs[(fword, eword)] / words[eword]

    def prob_list(self, fword, e, i, l, m):
        a = defaultdict(float)
        total = 0.0
        for j, eword in enumerate(e):
            val = self.t[fword, eword]
            total += val
            a[j] = val

        return total, a

    def set_t(self, tvals):
        for pair in tvals:
            self.t[pair] = tvals[pair]

    def set_q(self, qvals):
        pass

    def finalize(self):
        output = file("ibm1.t.txt", "w")
        for pair in self.t:
            output.write("%s %s %s\n" % (str(pair[0]), str(pair[1]), str(self.t[pair])))


class IBM2:
    def __init__(self):
        self.t = defaultdict(float)  # t[(fword, eword)] := t(f|e)
        self.q = defaultdict(float)  # q[(j, i, l, m)] := q(j|i,l,m)

    def initialize(self, corpus):
        for e, f in corpus:
            l = len(e)
            m = len(f)

            for i in range(m):
                for j in range(l):
                    self.q[(j, i, l, m)] = 1.0 / float(1 + l)

        for k in open("ibm1.t.txt", "r"):
            k = [l.strip() for l in k.split(" ")]
            self.t[(k[0], k[1])] = float(k[2])

    def prob_list(self, fword, e, i, l, m):
        a = {}
        total = 0.0
        for j, eword in enumerate(e):
            val = self.t[fword, eword] * self.q[j, i, l, m]
            total += val
            a[j] = val

        return total, a

    def set_t(self, tvals):
        self.t = tvals.copy()

    def set_q(self, qvals):
        self.q = qvals.copy()

    def finalize(self):
        output = file("ibm2.t.txt", "w")
        for pair in self.t:
            output.write("%s %s %s\n" % (str(pair[0]), str(pair[1]), str(self.t[pair])))

        output.close()
        output = file("ibm2.q.txt", "w")
        for quad in self.q:
            s = ""
            for v in quad:
                s += str(v) + " "
            s += str(self.q[quad])
            output.write(s + "\n")

        output.close()


class IBM:
    def __init__(self, IBMmodel):
        self.text = []
        self.model = IBMmodel

    def read_file(self, en_file, es_file):
        en = file(en_file, "r")
        es = file(es_file, "r")

        for e, f in zip(en, es):
            e = ["*"] + e.split(" ")
            f = f.split(" ")
            e[-1] = e[-1].strip()
            f[-1] = f[-1].strip()
            self.text.append((e, f))

        self.model.initialize(self.text)

    def EM(self, iterations=5):
        c_fe = defaultdict(int)
        c_e = defaultdict(int)
        c_jilm = defaultdict(int)
        c_ilm = defaultdict(int)

        t = defaultdict(float)
        q = defaultdict(float)

        for s in range(iterations):

            c_fe.clear()
            c_e.clear()
            c_jilm.clear()
            c_ilm.clear()
            t.clear()
            q.clear()

            for fe, (e, f) in enumerate(self.text):

                l = len(e)
                m = len(f)
                for i, fword in enumerate(f):
                    total, vals = self.model.prob_list(fword, e, i, l, m)
                    for j, eword in enumerate(e):
                        delta = vals[j] / total
                        c_fe[fword, eword] += delta
                        c_e[eword] += delta
                        c_jilm[j, i, l, m] += delta
                        c_ilm[i, l, m] += delta

            for fe in c_fe:
                t[fe] = c_fe[fe] / c_e[fe[1]]

            for j, i, l, m in c_jilm:
                q[j, i, l, m] = c_jilm[j, i, l, m] / c_ilm[i, l, m]

            self.model.set_t(t)
            self.model.set_q(q)

            print s

        self.model.finalize()

    def read_dev_file(self, en_file, es_file, out_file, invert=False):

        output = file(out_file, "w")
        en = file(en_file, "r")
        es = file(es_file, "r")

        for s, (e, f) in enumerate(zip(en, es)):
            e = [k.strip() for k in ["*"] + e.split(" ")]
            f = [k.strip() for k in f.split(" ")]

            l = len(e)
            m = len(f)

            for i, fword in enumerate(f):
                total, vals = self.model.prob_list(fword, e, i, l, m)
                d1 = max(vals, key=(lambda x: vals[x]))
                print d1
                if invert:
                    output.write("%s %s %s\n" % (s + 1, i + 1, d1))
                else:
                    output.write("%s %s %s\n" % (s + 1, d1, i + 1))

            output.flush()


if __name__ == "__main__":
    model = IBM1()
    mt = IBM(model)
    mt.read_file("corpus.en", "corpus.es")
    mt.EM()
    mt.read_dev_file("dev.en", "dev.es", "dev.ibm1.out")

    model = IBM2()
    mt = IBM(model)
    mt.read_file("corpus.en", "corpus.es")
    mt.EM()
    mt.read_dev_file("dev.en", "dev.es", "dev.ibm2.out")
