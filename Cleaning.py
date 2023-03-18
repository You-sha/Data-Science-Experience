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

df = pd.read_csv('data/naukri_scraped_data.csv')
df.head()

df.info()

df.columns
df.title.value_counts()
df.salary.value_counts()
df.experience.value_counts()
df.days_posted.value_counts()
df.location.value_counts()[:20]

df.drop('url',axis=1,inplace=True)

# job simplify
def title_simp(title):
    if 'data scientist' in title.lower() or 'datascientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower() or 'dataengineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower() or 'ml' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'

df['job_simp'] = df.title.apply(title_simp)

# job post
def job_post(title):
    if 'sr' in title.lower() or 'sr.' in title.lower() or 'senior' in title.lower()\
        or 'lead' in title.lower() or  'principal' in title.lower():
            return 'senior'
    
    elif 'jr' in title.lower() or 'jr.' in title.lower() or 'junior' in title.lower()\
        or 'associate' in title.lower():
            return 'junior'
    else:
        return 'na'
    
df['post'] = df.title.apply(job_post)  

# salary
df_sal = df[df.salary != 'Not disclosed'] # too little data
df['salary_mentioned'] = df.salary.apply(lambda x: 1 if x.lower()!='not disclosed' else 0)

# reviews int
df['reviews_int'] = df.reviews.apply(lambda x: int(x.split()[0]))

# experience
df['min_experience'] = df.experience.apply(lambda x: int(x.split()[0].split('-')[0]))
df['max_experience'] = df.experience.apply(lambda x: int(x.split()[0].split('-')[1]))
df['avg_experience'] = (df.min_experience+df.max_experience)/2

# days
df['days_posted_int'] = df.days_posted.apply(lambda x: x.split()[0].replace('+','').replace('Few','7')\
                                             .replace('Just','0').strip())
df.days_posted_int = df.days_posted_int.astype('int')

#location
locations = ['bangalore','delhi','kolkata','mumbai','remote','gurgaon','hyderabad',
             'noida','pune']
for i in locations:
    df[i] = df.location.apply(lambda x: 1 if i in x.lower() else 0)

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

# big data
skill('big_data','big data') #keep

# r
skill('r', ' r ') #keep

# nlp
df['nlp'] = df.tags.apply(lambda x: 1 if 'nlp' in x.lower() or 'natural language processing' in x.lower() else 0)
df.nlp.value_counts() #keep

# sas
skill('sas','sas') #keep

# git
skill('git','git') #keep

# tensorflow
skill('tensorflow','tensorflow') #keep

# pytorch
skill('pytorch','pytorch') #keep

# tableau
skill('tableau','tableau') #keep

# power bi
skill('power_bi','power bi') #keep

# apache
skill('apache','apache') #keep

# c++
skill('c++','c++') #keep


#output
df.to_csv('job_data_prepared.csv',index=None)

    
    
    
    













