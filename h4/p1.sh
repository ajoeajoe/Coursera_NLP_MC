#!/usr/bin/env bash
python glm.py p1  > gene_dev.p1.out
python eval_gene_tagger.py gene.key gene_dev.p1.out