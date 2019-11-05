import requests
import json
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
##
import pandas as pd
import json
import sqlite3
import ast
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy import stats
from sklearn import preprocessing 
from statsmodels.multivariate.manova import MANOVA 
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
# Web scraping

def get_response( food , city):
    """ it takes the type of food and the city and return the respond"""
    term = 'food' 
    location = 'city'
    SEARCH_LIMIT = 10

    url = 'https://api.yelp.com/v3/businesses/search'

    headers = {
            'Authorization': 'Bearer {}'.format(key),
        }

    url_params = {
                    'term': term.replace(' ', '+'),
                    'location': location.replace(' ', '+'),
                    'limit': SEARCH_LIMIT
                }
    response = requests.get(url, headers=headers, params=url_params)
    return response

def all_results(url_params, key , food, city ):
    """ it takes the url, key, type of food and city and return
        the webside information"""
    num = response.json()['total']
    print('{} total matches found.'.format(num))
    cur = 0
    dfs = []
    while cur < num and cur < 700:
        url_params['offset'] = cur
        dfs.append(yelp_call(url_params, key)) # change for a df
        time.sleep(1) #Wait a second
        cur += 50
    df = pd.concat(dfs, ignore_index=True)
    return df

term = 'x'
location = 'city'
url_params = {  'term': term.replace(' ', '+'),
                'location': location.replace(' ', '+'),
                'limit' : 50,
                 'radius' : 20000
             }

def store_csv(file):
    """ it takes the info from the yelp web and store in a csv file"""
    return df.to_csv('file')

##
def csv_to_db(a_list):
    '''it takes a list of csv files and converted into a dataframe'''
    for i in a_list:
        with open('/Users/cristinamulas/Desktop/Vegan_vs_meat/cris/csv/{}'.format(i[0]), newline='') as csvfile:
            _data = csv.reader(csvfile)
            for row in _data:
                rest_info = []
                try: 
                    name = row[10] #name
                    city = i[1] #city i[1]
                    price = row[12] #price
                    rating = row[13] #rating
                    review_count = row[14] #review_count
                    rest_type = i[2]
                    pop_in_mil = i[3]
                    rest_info.append([name, city, price, rating, review_count, rest_type, pop_in_mil])
                    c.execute("""INSERT INTO rest_info (name, city, price, rating, review_count, rest_type, pop_in_mil)
                    VALUES (?, ?, ?, ?, ?, ?, ?);""", rest_info[0])
                except: 
                    print('failed')
                    
def Cohen_d(group1, group2):
    diff = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()
    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    return d