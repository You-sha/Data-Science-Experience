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

df.query('location == "Bangalore/Bengaluru"').shape

df['reviews'] = df['reviews'].apply(lambda x: int(x.split()[0]))
pd.pivot_table(df,index='company',values='ratings').sort_values(by='ratings', \
                                                                ascending=False)[:10].plot.bar()

plt.figure(figsize=(10,5))
df.groupby('ratings')['reviews'].sum().sort_index(ascending=False).plot.bar()
df.ratings.plot.box()
df.reviews.plot.box()

df['avg_experience'] = df.experience.apply(lambda x: x.split()[0])
df['avg_experience'] = df['avg_experience'].apply(lambda x: x.split('-'))
df['avg_experience'] = df['avg_experience'].apply(lambda x: (int(x[0])+int(x[1]))/2)

df_sal = df.loc[df.salary != 'Not disclosed'].copy()
df_sal['avg_salary'] = df.salary.apply(lambda x: x.split()[0])
df_sal['avg_salary'] = df_sal['avg_salary'].apply(lambda x: x.replace('50,000','0.5'))
df_sal['avg_salary'] = df_sal['avg_salary'].apply(lambda x: x.split('-'))
df_sal['avg_salary'] = df_sal['avg_salary'].apply(lambda x: (float(x[0])+float(x[1]))/2)






