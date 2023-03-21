# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:35:50 2023

@author: Yousha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dython import nominal

df = pd.read_csv('data/naukri_final_data.csv')

df.columns
df.drop('Unnamed: 0',axis=1,inplace=True)
columns= ['avg_experience','title', 'company', 'ratings', 'reviews', 'experience', 'salary',
         'location', 'days_posted', 'tags', 'job_simp', 'post','top_20_comp',
         'salary_mentioned', 'reviews_int', 'min_experience', 'max_experience',
         'days_posted_int', 'bangalore', 'delhi', 'kolkata',
         'mumbai', 'remote', 'gurgaon', 'hyderabad', 'noida', 'pune', 'python',
         'sql', 'deep_learning', 'big_data', 'r', 'nlp', 'sas', 'git',
         'tensorflow', 'pytorch', 'tableau', 'power_bi', 'apache', 'c++'
         ]

df = df.reindex(columns, axis='columns')

#cont_corr
plt.figure(figsize=(25,25))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.heatmap(df.corr(),annot=True,square=True,linewidth=2,cbar_kws={'shrink':0.823})
plt.savefig('plots/Cont_correlation.png',dpi=800,bbox_inches='tight')

# avg_experience
avg_exp = pd.DataFrame(df.corr()['avg_experience'].sort_values(ascending=False))
plt.figure(figsize=(25,25))
sns.heatmap(avg_exp,square=True,linewidth=2,cbar=True,annot=True,annot_kws={'rotation':90})
plt.xticks(fontsize=12,rotation=90)
plt.yticks(fontsize=12,rotation=130)
plt.savefig('plots/avg_exp_correlation.png',dpi=800,bbox_inches='tight')

features =  ['avg_experience','ratings','salary','job_simp', 'post','top_20_comp',
             'salary_mentioned', 'reviews_int','days_posted_int', 'bangalore', 'delhi', 
             'kolkata','mumbai', 'remote', 'gurgaon', 'hyderabad', 'noida', 'pune', 'python',
             'sql', 'deep_learning', 'big_data', 'r', 'nlp', 'sas', 'git',
             'tensorflow', 'pytorch', 'tableau', 'power_bi', 'apache', 'c++'
             ]

ax = plt.figure()
ax = nominal.associations(df[features],figsize=(20,20),title='Feature Correlations')
plt.show()

X = pd.get_dummies(df[features].drop('avg_experience',axis=1))
y = df.avg_experience.values


# Models

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
scaler = StandardScaler()

X_trans = scaler.fit_transform(X)

# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=1) 

# statsmodel LR
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()


# sklearn LR
from sklearn.linear_model import LinearRegression, Lasso
lr = LinearRegression()

np.mean(cross_val_score(lr, X_train, y_train, scoring='neg_mean_absolute_error',cv=3)) # -2.37
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
pred_real = np.concatenate([y_pred.reshape(-1,1),y_test.reshape(-1,1)],axis=1)
residuals = y_test - y_pred

plt.scatter(np.linspace(0,residuals.max(),105), residuals,c=residuals,cmap='magma', edgecolors='black', linewidths=.1)
plt.colorbar(label="Quality", orientation="vertical")
# plot a horizontal line at y = 0
plt.hlines(y = 0,
xmin = 0, xmax=12.08,
linestyle='--',colors='black')
# set xlim
plt.xlim((0, 12.08))
plt.show()


# Lasso
lm = Lasso()

np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error',cv=3)) # -2.22

alpha = []
error = []

for i in range(1,100):
    lml = Lasso(alpha=i/100)
    alpha.append(i/100)
    x = np.mean(cross_val_score(lml, X_train, y_train, scoring='neg_mean_absolute_error',cv=3))
    error.append(x)

plt.plot(alpha,error)
alph_err = tuple(zip(alpha,error))
err_df = pd.DataFrame(alph_err, columns=['Alpha','Error'])

best_lm = err_df.query('Error == Error.max()')

# Random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error',cv=3)) # -2.21

rf.fit(X_train,y_train)  #performs better on unscaled data
y_pred_rf = rf.predict(X_test)

# GridSearch
from sklearn.model_selection import GridSearchCV
params = {'n_estimators': range(100,1000,100),
          'criterion':('squared_error', 'absolute_error'),
          'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(rf, param_grid=params,scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train,y_train)
gs.best_score_
gs.best_params_


# Tried using a neural netwrok ... I have no clue how to make this work at the moment.
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(100,50), activation='relu', solver='adam', alpha=0.001,max_iter=10000,random_state=1) 

np.mean(cross_val_score(mlp, X_train, y_train, scoring='neg_mean_absolute_error',cv=3))






















