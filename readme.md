# Data Science Experience Predictor: Project Overview

*   Created a tool that predicts average experience required for a data science position (**MAE: ~2 years**), to help prospective applicants find and apply to relevant and attainable positions.
*   Scraped over 70 pages from Naukri using Python, BeautifulSoup and Selenium.
*   Engineered features from the tags provided by the company on their job postings to quantify the value that companies put on the most popular skills in the field.
*   Optimized Linear, Lasso and Random Forest regressors with GridSearchCV to attain the best performing model.
*   Built a client facing API using Flask.

## Code and resources

**Python version:** 3.11

**Packages:** Pandas, Numpy, Scikit-Learn, Matplotlib, Seaborn, Selenium, BeautifulSoup, Flask, Json, Pickle

**Web Framework Requirements:** ```pip install -r requirements.txt```

**Scraper Article:** https://medium.com/analytics-vidhya/scraping-job-aggregator-site-naukri-com-using-python-and-beautiful-soup-a08a2046639b

**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Web Scraper

I learnt the basics of building a scraper from the article mentioned above and modified it to scrape all 78 pages of data science job postings. The following features were scraped:

*   Job URL
*   Job title
*   Number of reviews
*   Company rating
*   Company name
*   Experience required
*   Salary
*   Location
*   Days since posted
*   Tags

## Data Cleaning

After scraping the data, it required cleaning so that it could provide insights and be usable for the model. I made the following changes and created the following features:

*   Simplified job column.
*   Added a column stating the seniority of the position.
*   Added a column stating whether salary is mentioned or not.
*   Parsed the numeric data out of 'Number of reviews' column.
*   Parsed out min and max experience and created an average experience required column.
*   Created columns for job state.
*   Made new columns for skills mentioned in the tags.
*   Made a column for the top 20 companies with the most job postings.

## Exploratory Data Analysis

I analysed the distributions of the data and the value counts for the categorical columns. And made a WordCloud for the most frequent keywords appearing in the job tags. Below are a few highlights from the EDA:

<p>
<img src="https://user-images.githubusercontent.com/123200960/227010476-2ded9ec3-5dbe-427e-9547-3288c96ba658.png" width="280" height="300">
<img src="https://user-images.githubusercontent.com/123200960/227010094-7f965c6e-c7b5-411a-b9df-bcef6425dab5.jpg" width="300" height="300">
</p>

<p>
<img src="https://user-images.githubusercontent.com/123200960/227010164-1b47e7c8-6f8a-40c8-a22d-d43b17a191cd.jpg" width="200" height="400">
<img src="https://user-images.githubusercontent.com/123200960/227010749-97dca497-1f6f-487a-9ddb-3eeb7d4bf25b.jpg" width="440" height="280">
</p>

## Model Building

I transformed the categorical features to dummy variables. I used 3-fold Cross Validation for performance evaluation.

I built three different models and evaluated them using Mean Absolute Error, as it is relatively easy to interpret, and outliers aren't such an issue here.

The models I tried:

* **Multivariate Linear Regression** - Model baseline.
* **Lasso Regression** - Since Lasso Regression is normalized, it could be useful here as the data is sparse due to the many categorical variables.
* **Random Forest** - This should be good because of the sparsity as well. Plus, RFs tend to have relatively decent performance on most data.

## Model Performance

Lasso Regression performed the best on this data.

* **Lasso Regression MAE:** 2.14
* **Random Forest Regressor MAE:** 2.21
* **Multivariate Linear Regression MAE:** 2.37

## Productionization

I built a flask API endpoint that was hosted on a local server by following the tutorial provided in the reference. The API endpoint takes in a request with a list of values from a job listing and returns an estimated value for average experience.
