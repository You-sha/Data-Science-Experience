# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 23:31:54 2023

@author: Yousha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',10)
pd.set_option('display.width', 1000)

df = pd.read_csv('naukri_scraped_data.csv')
df.head()

df.info()

df.columns
df.title.value_counts()
df.salary.value_counts()
df.experience.value_counts()
df.days_posted.value_counts()
df.location.value_counts()[:20]


# experience
df['min_experience'] = df.experience.apply(lambda x: int(x.split()[0].split('-')[0]))
df['max_experience'] = df.experience.apply(lambda x: int(x.split()[0].split('-')[1]))
df['avg_experience'] = (df.min_experience+df.max_experience)/2


# salary
df_sal = df[df.salary != 'Not disclosed'] # too little data
df['salary_mentioned'] = df.salary.apply(lambda x: True if x.lower()!='not disclosed' else False)
df.salary_mentioned.value_counts()


# days
df['days_posted_int'] = df.days_posted.apply(lambda x: x.split()[0].replace('+','').replace('Few','7')\
                                             .replace('Just','0').strip())
df.days_posted_int = df.days_posted_int.astype('int')

#location
def location(column,name):
    df[column] = df.location.apply(lambda x: 1 if name in x.lower() else 0)
    print(f"{column} value counts:\n{df[column].value_counts()}")

# delhi
location('delhi','delhi')

# kolkata
location('kolkata','kolkata')

# bangalore
location('bangalore','bangalore')

# gurgaon
location('gurgaon','gurgaon')

# chennai
location('chennai','chennai')

# hyderabad
location('hyderabad','hyderabad')

# mumbai
location('mumbai','mumbai')

# noida
location('noida','noida')

# remote
location('remote','remote')

# pune
location('pune','pune')


# skills

df['python'] = df.tags.apply(lambda x: 1 if 'python' in x.lower() else 0)
df.python.value_counts()

def skill(column,name):
    df[column] = df.tags.apply(lambda x: 1 if name in x.lower() else 0)
    print(f"{column} value counts:\n{df[column].value_counts()}")

# sql
skill('sql','sql') #keep

# deep learning
skill('deep_learning','deep learning') #keep

# machine learning
df['machine_learning'] = df.tags.apply(lambda x: 1 if 'ml' in x.lower() or 'machine learning' in x.lower() else 0)
df.machine_learning.value_counts() #keep

# big data
skill('big_data','big data') #keep

# r
skill('r', ' r ') #keep

# hadoop
# skill('hadoop','hadoop') #discard

# spark
# skill('spark','spark') #discard

# nlp
df['nlp'] = df.tags.apply(lambda x: 1 if 'nlp' in x.lower() or 'natural language processing' in x.lower() else 0)
df.nlp.value_counts() #keep

# gpt
# skill('gpt','gpt') #discard

# management
# skill('management','management') #discard

# senior
# skill('senior','senior') #discard

# aws
# skill('aws','aws') #discard

# sas
skill('sas','sas') #keep

# automation
skill('automation','automation') #keep

# visual studio
# skill('visual_studio','visual studio') #discard

# excel
# skill('excel','excel') #discard

# research
skill('research','research') #keep

# git
skill('git','git') #keep

# supply chain
# skill('supply_chain','supply chain') #discard

# block chain
# skill('block_chain','blockchain') #discard

# pandas
# skill('pandas','pandas') #discard

# tensorflow
skill('tensorflow','tensorflow') #keep

# pytorch
skill('pytorch','pytorch') #keep

# tableau
skill('tableau','tableau') #keep

# power bi
skill('power_bi','power bi') #keep

# pivot
# skill('pivot','pivot') #discard

# scala
skill('scala','scala') #keep

# health
df['health'] = df.tags.apply(lambda x: 1 if 'health' in x.lower() or 'medicin' in x.lower() else 0)
df.health.value_counts() #keep

# apache
skill('apache','apache') #keep

# pipeline
# skill('pipeline','pipeline') #discard

# ai
df['ai'] = df.tags.apply(lambda x: 1 if 'ai' in x.lower() or 'artificial intelligence' in x.lower() else 0)
df.ai.value_counts() #keep

# NoSQL
# skill('nosql','nosql') #discard

# devops
# skill('devops','devops') #discard

# Java
# skill('java','java') #discard

# c++
skill('c++','c++') #keep

# julia
# skill('julia','julia') #discard

# matlab
# skill('matlab','matlab') #discard

# database management
# skill('database','database') #discard

# data vis
# skill('data_viz','visualization') #discard



















