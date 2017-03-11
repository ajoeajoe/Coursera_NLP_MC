#!/usr/bin/env bash
python glm.py p2a --gene-data gene.train > suffix_tagger.model
python glm.py p2b --gene-data gene.dev --model suffix_tagger.model > gene_dev.a4.p2.out
python eval_gene_tagger.py gene.key gene_dev.a4.p2.out