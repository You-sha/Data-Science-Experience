# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:51:02 2023

@author: Yousha
"""
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=DeprecationWarning)

pages = int(input('Enter number of pages to scrape: '))
page = 0
df = pd.DataFrame(columns=['title','company','ratings','reviews','experience',
                           'salary','location','days_posted','tags','url'])

for i in range(1,pages+1):
    if i == 1:
        # insert link to page 1 here
        url = "https://www.naukri.com/data-scientist-jobs-?k=data%20scientist&nignbevent_src=jobsearchDeskGNB"
    else:
        # insert link to 2nd page here, replace number 2 with str(i)
        url = "https://www.naukri.com/data-scientist-jobs-"+str(i)+"?k=data%20scientist&nignbevent_src=jobsearchDeskGNB"
    driver = webdriver.Chrome("C:\\Users\\Shumail\\anaconda3\\Lib\\site-packages\\chromedriver_binary\\chromedriver.exe")
    driver.get(url)
    
    time.sleep(4)
    
    soup = BeautifulSoup(driver.page_source,'html5lib')
    
    driver.close()
    
    results = soup.find(class_="list")
    job_elems = results.find_all('article',class_='jobTuple')
    
    for job_elem in job_elems:

        # Job URL
        url = job_elem.find('a',class_='title ellipsis').get('href')

        # Job Title
        title = job_elem.find('a',class_='title ellipsis')

        # Number of reviews
        review_span = job_elem.find('a',class_='reviewsCount fleft')
        if review_span is None:
            continue
        else:
            reviews = review_span.text

        # Company ratings
        rating_span = job_elem.find('span',class_='starRating fleft')
        if rating_span is None:
            continue
        else:
            ratings = rating_span.text

        # Company name
        company = job_elem.find('a',class_='subTitle ellipsis fleft')

        # Experience required
        Exp = job_elem.find('li',class_='fleft br2 placeHolderLi experience')
        Exp_span = Exp.find('span',class_='ellipsis fleft expwdth')
        if Exp_span is None:
            continue
        else:
            Experience = Exp_span.text

        # Salary
        Sal = job_elem.find('li',class_='fleft br2 placeHolderLi salary')
        Sal_span = Sal.find('span',class_='ellipsis fleft')
        if Sal_span is None:
            continue
        else:
            Salary = Sal_span.text

        # Location
        Loc = job_elem.find('li',class_='fleft br2 placeHolderLi location')
        Loc_exp = Loc.find('span',class_='ellipsis fleft locWdth')
        if Loc_exp is None:
            continue
        else:
            Location = Loc_exp.text

        # Days since job was posted
        hist = job_elem.find('div', class_='jobTupleFooter mt-8')
        hist2 = hist.find('div', class_='tupleTagsContainer')
        hist3 = hist2.find('span', class_='fleft postedDate')
        
        # job Tags
        tags = job_elem.find('ul',class_='tags has-description')
        tag1 = ""
        for tag in tags:
            tag1 += " "+tag.text

        # Adding data to dataframe
        df = df.append({'title':title.text,'company':company.text,'ratings':ratings,
                'reviews':reviews,'experience':Experience,'salary':Salary,
                'location':Location,'days_posted':hist3.text,'tags':tag1,'url':url}, 
                 ignore_index=True)
    
    page += 1
    print(f'Pages done {page}/{pages}')
    
df.duplicated().sum()

df2 = df.drop_duplicates()

#df2.to_csv('naukri_scraped_data.csv', index=None)
    
    
    
    
    
    
    
    
    
    
    
    
    