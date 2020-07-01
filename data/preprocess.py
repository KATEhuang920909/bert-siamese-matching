# -*- coding: utf-8 -*-
"""
 @Time : 2020/6/23 22:38
 @Author : huangkai
 @File : preprocess.py
 @Software: PyCharm 
 
"""
import  pandas as pd
from sklearn.utils import shuffle
data1=pd.read_csv(r'atec_nlp_sim_train.csv',sep='\t',header=None,index_col=0)
data2=pd.read_csv(r'atec_nlp_sim_train_add.csv',sep='\t',header=None,index_col=0)
print(data1.head())
print(data2.head())
data=pd.concat([data1,data2]).reset_index(drop=True)
data=shuffle(data)
data[:int(4*(len(data))/5)].reset_index(drop=True).to_csv(r'train.csv',sep='\t')
data[int(4*(len(data))/5):].reset_index(drop=True).to_csv(r'dev.csv',sep='\t')
