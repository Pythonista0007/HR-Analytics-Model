#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:56:16 2021

@author: Milind Prakash
"""

# =============================================================================
# Problem Statement related to HR analytics:
#     
# This csv file contains hiring statics for a firm such as experience of candidate, his written test score and personal interview score. Based on these 3 factors, 
# HR team will decide the salary. Given this data, you need to build a machine learning model for HR department that can help them decide salaries for future candidates. 
# Using this predict salaries for following candidates,
# 
# **2 yr experience, 9 test score, 6 interview score**
# 
# **12 yr experience, 10 test score, 10 interview score**
# 
# 
# =============================================================================


import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n
pip install word2number

df=pd.read_csv('/Users/apple/Desktop/Codebasics/py-master 2/ML/2_linear_reg_multivariate/Exercise/hiring.csv')
df
mapper={df.columns[0]:'Experience',df.columns[1]:'Test_score',df.columns[2]:'Interview_score',df.columns[3]:'Salary'}
df1=df.rename(columns=mapper)
df1

df1.Experience=df1.Experience.fillna('zero')
df1
#here we made mistake as we filled zero in the test score column as well

## Also convert string column to number column using word 2 number package.

df1.Experience=df1.Experience.apply(w2n.word_to_num)

df1

import math

mean_test_score=math.floor(df1['Test_score'].mean())

mean_test_score

df1['Test_score']=df1['Test_score'].fillna(mean_test_score)

df1

# **** Model Building********

regression= linear_model.LinearRegression()

regression.fit(df1[['Experience','Test_score','Interview_score']],df1['Salary'])

#** Always keep independent variables in 2d braces and target variable in 1d

regression.predict([[2,9,8]])


regression.predict([[2,9,6]])

regression.predict([[12,10,10]])






