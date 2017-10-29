# -*- coding: utf-8 -*-
"""
WMalgos: https://github.com/Dosann/WMalgos
author: duxin
email: duxin_be@outlook.com
create time: 2017/10/29 12:31
Just enjoy it and welcome for joining us!
"""

import numpy as np

class Optimizer:

    def __init__(self):
        self.name = 'Optimizer'

class GradientDescentOptimizer(Optimizer):

    def __init__(self, func, gradfunc, lr = 0.1, maxiter = 1000, precthres = 1e-5):
        self.name = 'GradientDescentOptimizer'
        self.func = func
        self.gradfunc = gradfunc
        self.lr = lr
        self.maxiter = maxiter
        self.precthres = precthres

    def optimize(self, X, Y, paras = 0):
        if W0 == 0 and X.shape[1] != 1:
            W0 = np.zeros(1, X.shape[1])
        for i in range(self.maxiter):
            y = func()