# -*- coding: utf-8 -*-
"""
WMalgos: https://github.com/Dosann/WMalgos
author: duxin
email: duxin_be@outlook.com
create time: 2017/10/29 19:26
Just enjoy it and welcome for joining us!
"""

import sys
sys.path.append('../../src/Core')
sys.path.append('../../src/Chapter03 Linear Models')
import numpy as np
from LinearRegression import *


lrm = LinearRegressionModel(lr = 0.005);
X = np.array([[1,3],[2,4],[3,7],[6,8]])
Y = np.array([[1],[2],[3],[4]])
SSR = lrm.fit(X, Y)
print("SSR: ", SSR)
pred = lrm.predict(np.array([[5,9]]))
print(pred)