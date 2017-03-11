# -*- coding:utf-8 -*-
# Filename: p1.py.py
# Author：hankcs
# Date: 2017-03-04 PM10:42
# Reference: https://github.com/leungwk/nlangp
import sys
from collections import defaultdict
from itertools import product


def read_sentences(doc_fpath):
    """where the document separates each token (including newlines) into its own line"""
    acc = []
    with open(doc_fpath, 'r') as pile:
        sentence = []
        for line in pile:
            word = line.strip().split(' ')[0]  # should handle "word" and "word tag" files
            if not word:  # end of sentence
                acc.append(sentence) if sentence else None  # ignore newline only sentences
                sentence = []
            else:
                sentence.append(word)
    return acc


def read_sentence_tags(doc_fpath):
    """where the document separates each token (including newlines) into its own line"""
    sentences = []
    tag_lines = []
    with open(doc_fpath, 'r') as pile:
        sentence = []
        tags = []
        for line in pile:
            tup = line.strip().split(' ')
            if len(tup) == 1:  # end of sentence
                sentences.append(sentence) if sentence else None  # ignore newline only sentences
                tag_lines.append(tags) if tags else None
                sentence = []
                tags = []
            else:
                word, tag = tup
                sentence.append(word)
                tags.append(tag)
    return sentences, tag_lines


def read_tag_model(file_path):
    model = {}
    with open(file_path, 'r') as pile:
        for line in pile:
            feature, weight = line.strip().split(' ')
            model[feature] = float(weight)
        return model


def trigram_feature(a, b, c):
    """
    trigram feature generator
    :param a: first word
    :param b: second word
    :param c: third word
    :return: string representation
    """
    return 'TRIGRAM:{}:{}:{}'.format(a, b, c)


def tag_feature(word, tag):
    """
    tag feature generator
    :param word:
    :param tag:
    :return:
    """
    return 'TAG:{}:{}'.format(word, tag)


def suff_feature(word, tag):
    """
    suffix feature function
    :param word:
    :param tag:
    :return:
    """
    suff_keys = []
    for idx in range(1, 3 + 1):
        suff = word[-idx:]
        if len(suff) != idx:  # ie. not enough letters remaining
            continue
        suff_keys.append('SUFF:{}:{}:{}'.format(suff, idx, tag))
    return suff_keys


def gene_punc_feature(word, w, u, v):
    if word in '-/()\'.':
        return 'GENE_PUNC:{}:{}:{}:{}'.format(word, w, u, v)


def _len_tag_key(word, tag):  # v
    return 'LEN_TAG:{}:{}'.format(len(word), tag)


def simple_feature(w, u, v, word):
    """
    simple feature extractor
    :param w: -2 tag
    :param u: -1 tag
    :param v: current tag
    :param word: current word
    :return:
    """
    feature = [trigram_feature(w, u, v), tag_feature(word, v)]
    return feature


def more_feature(w, u, v, word):
    """
    simple feature extractor
    :param w: -2 tag
    :param u: -1 tag
    :param v: current tag
    :param word: current word
    :return:
    """
    feature = simple_feature(w, u, v, word)
    feature += suff_feature(word, v)
    return feature


def most_feature(w, u, v, word):
    """
    simple feature extractor
    :param w: -2 tag
    :param u: -1 tag
    :param v: current tag
    :param word: current word
    :return:
    """
    feature = simple_feature(w, u, v, word)
    feature += suff_feature(word, v)
    punc_feature = gene_punc_feature(word, w, u, v)
    if punc_feature:
        feature.append(gene_punc_feature(word, w, u, v))
    return feature


def vg(w, u, v, word, dict_tag_model):
    """
    v dot g for current state
    :param w: -2 tag
    :param u: -1 tag
    :param v: current tag
    :param word: current word
    :param dict_tag_model: tag model
    :return:
    """
    v = most_feature(w, u, v, word)
    # gene_punc_keys = gene_punc_feature(word, w, u, v)

    ## dot product on weights and if it exists
    vg = sum([dict_tag_model.get(k, 0) for k in v])
    return vg


def viterbi(sentence, dict_tag_model):
    """calculate y^* = \arg\max_{t_1,\dots,t_n \in GEN(x)} f(x,y) \cdot v. x is the sentence history"""

    n = len(sentence)

    S = ['I-GENE', 'O']
    pie, bp = {(0, '*', '*'): 0}, {}

    def S_(k):
        return ['*'] if k <= 0 else S

    for k in xrange(1, n + 1):
        word = sentence[k - 1]  # idx-0
        for u, s in product(S_(k - 1), S_(k)):
            ## \max_{w \in S_{k-2}}
            max_val, max_tag = -float('inf'), None
            for t in S_(k - 2):
                vg_ = vg(t, u, s, word, dict_tag_model)
                pie_ = pie[(k - 1, t, u)]
                ## previous most likely sentence so far, then probability of this trigram
                pkus = pie_ + vg_
                if pkus > max_val:
                    max_val, max_tag = pkus, t

            idx = (k, u, s)
            pie[idx], bp[idx] = max_val, max_tag

    ## calculate for the end of sentence
    max_val, max_tag = -float('inf'), None
    for u, s in product(S_(n - 1), S_(n)):
        pkus = pie[(n, u, s)] + vg(u, s, 'STOP', word, dict_tag_model)
        if pkus > max_val:
            max_val, max_tag = pkus, (u, s)

    ## go back in the chain ending with y_{n-1} and y_n
    t = [None] * n
    t[n - 1] = max_tag[1]  # y_n
    t[n - 1 - 1] = max_tag[0]  # y_{n-1}
    for k in xrange(n - 2, 1 - 1, -1):
        t[k - 1] = bp[(k + 2, t[(k + 1) - 1], t[(k + 2) - 1])]

    return t


def perceptron(sentences, tag_lines, n_iter):
    model = defaultdict(int)  # v

    def feature_vector(tags, sentence):
        v = defaultdict(int)
        tags = ['*', '*'] + tags + ['STOP']
        for word, w, u, s in zip(sentence, tags, tags[1:], tags[2:]):
            for f in most_feature(w, u, s, word):
                v[f] += 1

        return dict(v)

    for _ in xrange(n_iter):
        for sentence, gold_tags in zip(sentences, tag_lines):
            best_tags = viterbi(sentence, model)  # best tag sequence (GEN) under the current model
            gold_features = feature_vector(gold_tags, sentence)
            best_features = feature_vector(best_tags, sentence)
            if gold_features != best_features:
                ## update weight vector v
                for key, val in gold_features.iteritems():
                    model[key] += val
                for key, val in best_features.iteritems():
                    model[key] -= val
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('part', help='part to run')  # nargs=1, produces a list, while nothing produes the value itself
    parser.add_argument('--gene-data', help='document of sentences (default: gene.key)', default='gene.key')
    parser.add_argument('--model', help='', default='tag.model')
    args = parser.parse_args()

    gene_fpath = args.gene_data

    if args.part == 'p1':
        sentences, tag_lines = read_sentence_tags(gene_fpath)
        dict_tag_model = read_tag_model(args.model)

        for sentence in sentences:
            tags = viterbi(sentence, dict_tag_model)
            for word, tag in zip(sentence, tags):
                sys.stdout.write('{} {}\n'.format(word, tag))
            sys.stdout.write('\n')

    elif args.part == 'p2a':
        sentences, tag_lines = read_sentence_tags(gene_fpath)
        dict_tag_model = perceptron(sentences, tag_lines, 6)
        for key, val in dict_tag_model.iteritems():
            sys.stdout.write('{} {}\n'.format(key, val))
    elif args.part == 'p2b':
        sentences = read_sentences(gene_fpath)
        dict_tag_model = read_tag_model(args.model)

        xy = []
        for sentence in sentences:
            tags = viterbi(sentence, dict_tag_model)
            xy.append((sentence, tags))
        for sentence, tags in xy:
            for word, tag in zip(sentence, tags):
                sys.stdout.write('{} {}\n'.format(word, tag))
            sys.stdout.write('\n')
    else:
        raise ValueError('unknown part specified')
