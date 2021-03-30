#!/usr/bin/python -tt
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 04:07:23 2020

@author: DaltonGlove with credit to 

"""

#import modules
import bs4 as bs
import pickle
import requests

import matplotlib.pyplot as plt
from matplotlib import style

import datetime as dt
import os
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np

from collections import Counter

from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from sklearn.neural_network import MLPClassifier

#import fix_yahoo_finance as yf
import yfinance as yf
yf.pdr_override()

# read in sp500 tickers from Wikipedia
def save_sp500_tickers():
    
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
                        #headers=headers)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    #table = soup.find('table', {'class':'wikitable sortable'})
    table = soup.find('table', {'id': 'constituents'})
    tickers=[]
    for row in table.findAll('tr')[1:]:
        #ticker = row.findAll('td')[1].text.replace('.', '-')
        ticker = row.find('td').text.replace('.', '-')
        #ticker = ticker[:-1]
        tickers+=[ticker.strip()]
        #print(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    print(tickers)
    
    return tickers








# get financial data for tickers from yahoo finance
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
            
    if not os.path.exists('/stock_dfs'):
        os.makedirs('/stock_dfs')
    start = dt.datetime(2005, 1, 1)
    end = dt.datetime(2020, 7, 3)
    
    for ticker in tickers:
      """adjusted webscraping to using fix_yahoo import so retry and regenerate dfs"""
      try:
        if not os.path.exists('/stocks_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker,start, end)             #pdr.DataReader(ticker,'yahoo',start,end)  #pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index('Date', inplace=True)
            df.to_csv('/stock_dfs/{}.csv'.format(ticker))
            print('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))
      except KeyError: pass

#bring data together and save to csv files
def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    
    main_df = pd.DataFrame()
    print(main_df.head())
    start = dt.datetime(2005, 1, 1)
    end = dt.datetime(2020, 7, 3)
    for count, ticker in enumerate(tickers):
      try:
        #if ticker=='CARR' or ticker=='CTVA' or ticker == 'DOW' or ticker == 'FOXA' or ticker =='FOX' or ticker =='HWM' or ticker == 'IR': continue
        if not os.path.exists('stocks_dfs/{}.csv'.format(ticker.replace('.', '-'))):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker.replace('.', '-')))
            print(ticker)
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker.replace('.', '-'), index_col=0))
        df.set_index('Date', inplace=True)
        
        # drop all but the adjusted close prices

        df.rename(columns = {'Adj Close': ticker}, inplace=True)
        df.drop(['Open','High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        #print(df.head(), '\n\n\n')
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.merge(df, how='outer', on='Date')
            #print(main_df.head())
        if count % 10 == 0: print(count)
      except: print('problem with {} contining iteration'.format(ticker))
    print(main_df.head())
    main_df=main_df.pct_change()
    main_df.to_csv('sp500_joined_closes_pct_change.csv')

"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++ adjust main df of above csv ++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# visualize data with correlation heatmap
def visualize_data():
    df =  pd.read_csv('C:\\Users\LENOVO\Desktop\sp500_joined_closes.csv', )
    df.set_index('Date', inplace=True)
#    df['AAPL'].plot()
#    plt.show()
    df_corr=df.corr()
	
    df_corr=[df_corr < 0.0]
    
    data = df_corr.values
    print(sorted(df_corr.min()), df_corr.idxmin())
    fig = plt.figure()
    ax=fig.add_subplot(111)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[1]) + .5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) +.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index
    
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    fig.savefig('sp500_corr_plot')
    plt.show()
#

    #take the bottom triangle since it repeats itself

    #mask = np.zeros_like(df_corr)
    #mask[np.triu_indices_from(mask)] = True

#    #generate plot


##################################################################################################
##################################################################################################

#reprocess data for Machine learning algorithm

def process_data_labels(ticker):
    hm_days=60
    df = pd.read_csv('C:\\Users\LENOVO\Desktop\sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace=True)
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
    return tickers, df

# buy sell hold measure checking input arguements vs the requirement %-change
# adjust here to add further metrics of risk, volitility, and momentum
def buy_sell_hold1(*args):
    cols = [c for c in args]
    requirement = 0.025
    #print(cols)
    #for i in range(1,hm_days+1):
    #cols.append('{}-{}d'.format(ticker,i))
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

# run tickers through buy/sell/hold funct for each day and gather feature vectors
def extract_feat_sets(ticker):
    tickers, df = process_data_labels(ticker)
    #print(df.head())
    #for i in range(1,hm_days):
    #    df['{}_{}target'.format ( ticker,i )] =  list(map ( buy_sell_hold, df['{}_{}d'.format ( ticker, i )] ))
    #df['{}_target'.format(ticker)] = df.apply(lambda row:buy_sell_hold1(row),axis=1)
    #df['{}_target'.format(ticker)] = list(map(buy_sell_hold1, [df['{}_{}d'.format(ticker, i)] for i in range(1, 8)]))
    df['{}_target'.format(ticker)] = list(map( buy_sell_hold1,
                                                df['{}_1d'.format(ticker)],
                                                df['{}_2d'.format(ticker)],
                                                df['{}_3d'.format(ticker)],
                                                df['{}_4d'.format(ticker)],
                                                df['{}_5d'.format(ticker)],
                                                df['{}_6d'.format(ticker)],
                                                df['{}_7d'.format(ticker)] ))
    
    #print(df.head())
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread for {}:'.format(ticker), Counter(str_vals))
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    df_vals = df[[ticker for ticker in tickers]].pct_change() #Normalize values from previous close
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    
    return X, y, df
    
# use buy/sell/hold check to make feature sets and then classify 
# check accuracy first with historical data split 
def do_ml(ticker):
    X, y, df = extract_feat_sets(ticker)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35)
    
    #clf = neighbors.KNeighborsClassifier()
    
 
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                     hidden_layer_sizes=(5, 2), random_state=1)
  
    clf.fit(X_train, y_train)
   
    confidence = clf.score(X_test, y_test)
  
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts for {}:'.format(ticker), Counter(predictions))
    print()
    print()
    return confidence


#do_ml('BAC')

"""sort tickers by industry for better visualization by column groups
    or clustering algorithm, implement Scaling and lag-cross-correlation
    attempt google news parsing and financial statement parsing"""

#save_sp500_tickers()
#get_data_from_yahoo()
compile_data()
#visualize_data()

