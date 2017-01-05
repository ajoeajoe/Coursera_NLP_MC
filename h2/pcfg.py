#! /usr/bin/python
# coding=utf-8
import json
from count_cfg_freq import Counts
from collections import defaultdict
from decimal import Decimal

__author__ = 'ping.zou'
__date__ = '30 Mar 2013'

DEBUG = True
RARE_TAG = '_RARE_'
RARE_WORD_THRESHOLD = 5
ROOT = 'SBARQ'


def rare_words_rule_p1(word):
    return RARE_TAG


def process_rare_words(input_file, output_file, rare_words, processer):
    """
    替换低频词，并输出到文件
    :param input_file:
    :param output_file:
    :param rare_words:
    :param processer:
    """
    for line in input_file:
        tree = json.loads(line)
        replace(tree, rare_words, processer)
        output = json.dumps(tree)
        output_file.write(output)
        output_file.write('\n')


def replace(tree, rare_words, processer):
    """
    替换一棵树中的低频词
    :param tree:
    :param rare_words:
    :param processer:
    :return:
    """
    if isinstance(tree, basestring):
        return

    if len(tree) == 3:
        # Recursively count the children.
        replace(tree[1], rare_words, processer)
        replace(tree[2], rare_words, processer)
    elif len(tree) == 2:
        if tree[1] in rare_words:
            tree[1] = processer(tree[1])


def log(msg):
    if DEBUG:
        print msg


class PCFG(Counts):
    """
    Store counts, and model params
    """

    def __init__(self):
        super(PCFG, self).__init__()

        self.word = defaultdict(int)
        self.rare_words = []
        self.q_x_y1y2 = defaultdict(float)
        self.q_x_w = defaultdict(float)

    def count_word(self):
        '''
        count emitted words and find rare words
        '''
        # count emitted word
        for (sym, word), count in self.unary.iteritems():
            self.word[word] += count
        # find rare word
        for word, count in self.word.iteritems():
            if count < RARE_WORD_THRESHOLD:
                self.rare_words.append(word)

    def cal_rule_params(self):
        """
        统计uni和bin rule的频率
        """
        # q(X->Y1Y2) = Count(X->Y1Y2) / Count(X)
        for (x, y1, y2), count in self.binary.iteritems():
            key = (x, y1, y2)
            self.q_x_y1y2[key] = float(count) / float(self.nonterm[x])
        # q(X->w) = Count(X->w) / Count(X)
        for (x, w), count in self.unary.iteritems():
            key = (x, w)
            self.q_x_w[key] = float(count) / float(self.nonterm[x])

    def write(self, output):
        for sym, count in self.nonterm.iteritems():
            output.write('{count} NONTERMINAL {sym}\n'.format(count=count, sym=sym))

        for (sym, word), count in self.unary.iteritems():
            output.write('{count} UNARYRULE {sym} {word}\n'.format(count=count, sym=sym, word=word))

        for (sym, y1, y2), count in self.binary.iteritems():
            output.write('{count} BINARYRULE {sym} {y1} {y2}\n'.format(count=count, sym=sym, y1=y1, y2=y2))

    def read(self, input):
        '''
        Read model
        '''
        self.unary = {}
        self.binary = {}
        self.nonterm = {}
        self.word = defaultdict(int)
        self.rare_words = []
        self.q_x_y1y2 = defaultdict(float)
        self.q_x_w = defaultdict(float)

        for line in input:
            parts = line.strip().split(' ')
            count = float(parts[0])
            if parts[1] == 'NONTERMINAL':
                sym = parts[2]
                self.nonterm.setdefault(sym, 0)
                self.nonterm[sym] = count
            elif parts[1] == 'UNARYRULE':
                sym = parts[2]
                word = parts[3]
                self.unary.setdefault((sym, word), 0)
                self.unary[(sym, word)] = count
            elif parts[1] == 'BINARYRULE':
                sym = parts[2]
                y1 = parts[3]
                y2 = parts[4]
                key = (sym, y1, y2)
                self.binary.setdefault(key, 0)
                self.binary[key] = count
        self.count_word()
        self.cal_rule_params()

    def write_params(self, output_file):
        for (x, y1, y2), param in self.q_x_y1y2.iteritems():
            output_file.write('{param} BINARYRULE {x} {y1} {y2}\n'.format(param=param, x=x, y1=y1, y2=y2))

        for (x, w), param in self.q_x_w.iteritems():
            output_file.write('{param} UNARYRULE {x} {w}\n'.format(param=param, x=x, w=w))


class CKYTagger(PCFG):
    def __init__(self):
        super(CKYTagger, self).__init__()
        self.rare_words_rule = rare_words_rule_p1

    def tag(self, input, output):
        for line in input:
            result = self.CKY(line)
            output.write(json.dumps(result))
            output.write('\n')
            pass

    def CKY(self, sentence):
        pi = defaultdict()
        bp = defaultdict()
        N = self.nonterm.keys()

        words = sentence.strip().split(' ')
        n = len(words)

        # process rare word,测试文件中的未登录词按照相同的规则预处理
        for i in xrange(0, n):
            if words[i] not in self.word.keys():
                words[i] = self.rare_words_rule(words[i])

        log('Sentence to process: {sent}'.format(sent=' '.join(words)))
        log('n = {n}, len(N) = {ln}'.format(n=n, ln=len(N)))

        # reduce X, Y and Z searching space,剪枝策略，过滤掉那些不合法的rule
        SET_X = defaultdict()
        for (X, Y, Z) in self.binary.keys():
            if X in SET_X:
                SET_X[X].append((Y, Z))
            else:
                SET_X[X] = []

        # init, unary rule
        for i in xrange(1, n + 1):
            w = words[i - 1]
            for X in N:
                if (X, w) in self.unary.keys():
                    pi[(i, i, X)] = Decimal(self.q_x_w[(X, w)])
                else:
                    pi[(i, i, X)] = Decimal(0.0)

        # dp
        for l in xrange(1, n):
            for i in xrange(1, n - l + 1):
                j = i + l

                for X, YZPairs in SET_X.iteritems():
                    cur_pi, max_pi = 0.0, -1.0
                    for (Y, Z) in YZPairs:
                        for s in xrange(i, j):
                            # 由于我们用SET_X做了剪枝，所以需要检查是否属于被过滤掉的非法rule
                            if (i, s, Y) not in pi or (s + 1, j, Z) not in pi:
                                continue
                            cur_pi = Decimal(self.q_x_y1y2[(X, Y, Z)]) \
                                  * pi[(i, s, Y)] \
                                  * pi[(s + 1, j, Z)]
                            if cur_pi > max_pi:
                                max_pi = cur_pi
                                max_Y, max_Z, max_s = Y, Z, s
                                pi[(i, j, X)] = max_pi
                                bp[(i, j, X)] = (max_Y, max_Z, max_s)

        if (1, n, ROOT) not in bp:
            max_pi = 0.0
            max_X = ''
            for X, YZPairs in SET_X.iteritems():
                if (1, n, X) in pi and pi[(1, n, X)] > max_pi:
                    max_pi = pi[(1, n, X)]
                    max_X = X
        else:
            max_X = ROOT
        result = self.traceback(pi, bp, sentence, 1, n, max_X)
        return result

    def traceback(self, pi, bp, sentence, i, j, X):
        words = sentence.strip().split(' ')
        tree = []
        tree.append(X)
        if i == j:
            tree.append(words[i - 1])
        else:
            Y1, Y2, s = bp[(i, j, X)]
            # print Y1, Y2, s
            tree.append(self.traceback(pi, bp, sentence, i, s, Y1))
            tree.append(self.traceback(pi, bp, sentence, s + 1, j, Y2))
        return tree
