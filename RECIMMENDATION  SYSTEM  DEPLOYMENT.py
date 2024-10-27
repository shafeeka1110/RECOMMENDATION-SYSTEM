# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:25:43 2024

@author: Mohammed Arif
"""

import numpy as np
import pandas as pd
import pickle


loaded_model=pickle.load(open('C:/Users/Mohammed Arif/Downloads/deploy machinelearning/recommendation.sav','rb'))
df=pd.read_csv(r"C:/Users/Mohammed Arif/Documents/dataset/ratings_Electronics (1).csv",names=['userid','productid','rating','timestamp'])
#electronics_data=df.sample(n=1048576,ignore_index=True)
data=electronics_data.groupby('productid').filter(lambda x:x['rating'].count()>50)

#!pip install scikit-surprise
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
data2=data.sample(20000)
ratings_matrix=data2.pivot_table(values='rating',index='userid',columns='productid',fill_value=0)

reader=Reader(rating_scale=(1,5))
surprise_data=Dataset.load_from_df(data,reader)

trainset,testset=train_test_split(surprise_data,test_size=0.3,random_state=42)

algo=KNNWithMeans(k=5,sim_options={'name':'pearson_baseline','user_based':False})

algo.fit(trainset)


test_pred=loaded_model.test(testset)

recommend=list(x_ratings_matrix.index[correlation_product_id>0.85])
recommend[:20]

