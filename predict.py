#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhou Ze'
__version__ = '2.0'

import sys
import pickle

path_to_svm_model, path_to_cmp_result = sys.argv[1:]
clf = pickle.load(open(path_to_svm_model, 'rb'))

with open(path_to_cmp_result) as f:
    vals = next(f).rstrip().split('\t')

similarity, tau, tau_p, up_med, mid_med, low_med, up_p, low_p, ul_p = [float(i) for i in vals]

vals = [tau, tau_p, up_med, mid_med, low_med, up_p, low_p, ul_p]

ctDNA_positive = clf.predict([vals])[0] == 1

print(f'{path_to_cmp_result} ctDNA:\t{ctDNA_positive}')
