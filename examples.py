# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:20:01 2022


This file contains two sentence classification problems: 
    -sentiment classification for sentences in a foreign language
    -topic classification for english sentences

@author: chesn
"""

import pandas as pd
import numpy as np
from BTFC import BTF

""" sentiment classification example""" 

# data manipulation
df=pd.read_csv(r'./Roman Urdu DataSet.csv',header=None)

df.head()

# drop NA cols and rows
df[2].isna().sum()/len(df)
df.iloc[np.where(df[2].isna()==False)]

df=df.drop(np.where(df[2].isna()==False)[0])
df=df[[0,1]]

df.columns=['text','label']

df=df.drop(np.where(df['text'].isna()==True)[0])

df.index=range(len(df))

# clean labels
set(df['label'])

np.where(df['label']=='Neative')
df.loc[np.where(df['label']=='Neative')[0],'label']='Negative'

# get training and testing data

train_indc=np.random.choice(range(len(df)),int(.75*len(df)),replace=False)
train_df=df.loc[train_indc]
test_df=df.iloc[~train_indc]

# train
btf=BTF(train_df)
btf.single_term_dists('text','label')

# use Bayes Theorem to compute conditional probabilities 
# P(A|B), where A=sample belongs to class  B=words used
# classify sample as belonging to class with highest conditional prob 

# evaluate
btf.single_term_test(np.array(train_df['text']),np.array(train_df['label']))

#'null' represents the number of samples for which the prob of the occurance for 
# the sequence of terms was so low that Baye's Theorem's denominator was rounded to 0 by Python
# For these samples, I randomly assign a class using the class distribution


""" topic classification example """

df=pd.read_csv(r'.\payment_sentences_task.csv')
df.head()

# split data
train_indc=np.random.choice(range(len(df)),int(.75*len(df)),replace=False)
train_df=df.loc[train_indc]
test_df=df.drop(train_indc)

# train
model=BTF(train_df)
model.single_term_dists('sentence','payment_label')

# evaluate
model.single_term_test(np.array(train_df['sentence']),np.array(train_df['payment_label']))




