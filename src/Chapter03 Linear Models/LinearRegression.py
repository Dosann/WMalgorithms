# -*- coding: utf-8 -*-
"""
WMalgos: https://github.com/Dosann/WMalgos
author: duxin
email: duxin_be@outlook.com
create time: 2017/10/29 12:08
Just enjoy it and welcome for joining us!
"""

import sys
sys.path.append('../Core')
import basemodel
import numpy as np
import pandas as pd


class LinearRegressionModel(basemodel.BaseModel):
    w = 0
    b = 0

    def __init__(self, lr = 0.01, maxiter = 1000, prec = 1e-5):
        basemodel.BaseModel.__init__(self)
        self.name = 'LinearRegressionModel'
        self.lr = lr
        self.maxiter = maxiter
        self.prec = prec

    def fit(self, X, Y):
        X = np.hstack((X, np.ones([X.shape[0], 1])))
        XTX = np.dot(X.T, X)
        # make sure if XTX is singular matrix
        isSing = 0
        try:
            XTXinv = np.linalg.inv(XTX)
        except:
            print('XTX is singular!')
            isSing = 1
        if isSing == 1:
            w = self.__fitByGDopt(X, Y, w0 = np.random.randn(X.shape[1], 1))
        else:
            w = np.dot(XTXinv, np.dot(X.T, Y))
        self.w = w
        # calculate SSR
        Yhat = np.dot(X, self.w)
        return np.mean((Yhat - Y)**2)

    def __fitByGDopt(self, X, Y, w0):
        w = w0
        for iter in range(self.maxiter):
            error = np.dot(X, w) - Y
            if np.mean(error**2) < self.prec:
                return w
            grad = 2 * np.dot(X.T, np.dot(X, w) - Y)
            print(grad)
            w += - grad * self.lr
        self.w = w
        return w


    def predict(self, x):
        x = np.hstack((x, np.array([[1]])))
        return np.dot(x, self.w)


