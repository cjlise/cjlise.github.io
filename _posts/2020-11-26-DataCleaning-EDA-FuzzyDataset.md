---
title: "Data Cleaning And Exploratory Data Analysis Of A Fuzzy Dataset"
date: 2020-11-26
categories: machine-learning
tags: [Data Cleaning, EDA, Data Analysis, Data Visualization, Pandas, Record Linkage, Machine Learning, Python]
header: 
   image: "/images/MachineLearning/landscape-4527525_200.jpg"
excerpt: "Data Cleaning, EDA, Data Analysis, Data Visualization, Pandas, Record Linkage, Machine Learning, Python"
---

  
The goal of this data analysis is to cleanup a fuzzy dataset full of duplicates rows, and afterwards to carry out an exploratory analysis of the cleaned data. The dataset is a 20000 rows COVID-19 PCR test results. The analysis is in French. 





## 1. Data Quality Assessment 
The Jupyter notebook below covers the evaluation of data quality, and the available options to fix the duplicate rows issue. 

[Data Quality Assessment Notebook](https://github.com/cjlise/inria/blob/master/Evaluation%20de%20la%20qualit%C3%A9%20des%20donn%C3%A9es.ipynb) 


## 2. detect_duplicates.py function
The implementation of the detect_duplicates.py function below, detects duplicates from an imput dataframe, remove them and return the cleaned dataframe.   

```python
# detect_duplicates function to dectect duplicates rows and remove them in a Pandas dataframe
# Author: Jose Lise
# September 2020

import pandas as pd
import recordlinkage

def detect_duplicates (df_in):
    '''
    Function used to detected and drop duplicates due to fuzzy data (e.g. due to typos) in a df_patient dataframe.
    
    Parameters
    ----------    
    df_in : Dataframe
        df_patient dataframe to clean from duplicates 
        We assume that the input dataframe has the following columns: 
        given_name, surname, street_number, address_1, suburb, postcode, state, date_of_birth, age, phone_number, address_2
        
    Returns
    -------
    df : Dataframe
        The clean df_patient dataframe with the duplicates rows removed 
    '''

    # Copy the the input Dataframe
    df = df_in.copy()

    # Indexation step
    # The index is built from some columns to make possible pairs of duplicates
    # We choose the columns: 
    #   - address_1: because it has potentially the more characters 
    #   - phone_number: it has only digits 
    #   - surname
    dupe_indexer = recordlinkage.Index()
    dupe_indexer.sortedneighbourhood(left_on='address_1', window=3)
    dupe_indexer.sortedneighbourhood(left_on='phone_number', window=3)
    dupe_indexer.sortedneighbourhood(left_on='surname', window=3)
    
    dupe_candidate_links = dupe_indexer.index(df) 
    
    # Comparison step
    compare_dupes = recordlinkage.Compare()
    
    # We added below all the columns that we want to compare for duplicates (all of them except patient_id)
    compare_dupes.string('given_name', 'given_name', method='jarowinkler', threshold=0.85, label='given_name')
    compare_dupes.string('surname', 'surname', method='jarowinkler', threshold=0.85, label='surname')
    compare_dupes.exact('street_number', 'street_number', label='street_number')
    compare_dupes.string('address_1', 'address_1', threshold=0.85, label='address_1')
    compare_dupes.string('suburb', 'suburb', threshold=0.85, label='suburb')
    compare_dupes.exact('postcode', 'postcode', label='postcode')
    compare_dupes.exact('state', 'state', label='state')
    compare_dupes.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
    compare_dupes.exact('age', 'age', label='age')
    compare_dupes.string('phone_number', 'phone_number', threshold=0.85, label='phone_number')
    compare_dupes.exact('address_2', 'address_2', label='address_2')
    
    dupes_features = compare_dupes.compute(dupe_candidate_links, df)
    # dupes_features is a dataframe with 2 row index that represent a pair of row (row 1, row2), 
    # and the same number of column as df. each value is a score 0. or 1. 
    # 0 for one column means the row1 and row2 values for this column are not duplicates
    # 1 means duplicates    
    
    # We setup the threshold score for the potential dupes rows at 5.0: If 5 or more columns values are duplicates then full row is duplicate 
    # Items with score > 4 are duplicates
    # potential_dupes contains all the row pairs above the threshold plus and additional score column
    potential_dupes = dupes_features[dupes_features.sum(axis=1) > 4].reset_index()
    potential_dupes['Score'] = potential_dupes.loc[:, 'given_name':'address_2'].sum(axis=1)
    
    # As we ran reset_index() potential_dupes df has now additionnal columns level_0 and level_1 
    # Level_1 column contains the index of the duplicates to 
    # remove. We do that with the instruction below.     
    df = df.drop(axis=0, index=potential_dupes['level_1']).reset_index(drop=True)
    
    # compute percentage, then print percentage and number of rows removed
    print ("Percentage of duplicate rows: ", 100 * (len(df_in) - len(df))/len(df_in))
    print ("Number of rows removed: ", len(df_in) - len(df))
    
    return df

```


## 2. Exploratory Data Analysis

[Exploratory Data Analysis Notebook](https://github.com/cjlise/inria/blob/master/Analyse%20Exploratoire%20des%20donn%C3%A9es.ipynb)






	