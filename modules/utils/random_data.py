#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from random import *
import numpy as np
import pandas as pd
import os

def select_random_data(data,nb):
    random = np.random.choice(data, nb, replace=False)
    return random


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    project_root = root + os.sep + '..' + os.sep + '..'

    list=['joli','beau','magnifique']

    base_textes=pd.read_csv(project_root + os.sep + 'data' + os.sep + 'base_textes.txt', sep='\t', encoding='utf8')
    dataBefEOA=base_textes['BASE'][base_textes['LOC']=='BefEOA']
    print(select_random_data(dataBefEOA,2))