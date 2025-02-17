#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Started out with code:
Created on Sat Jan  5 13:48:20 2019
@author: zhanglemei and peng

Group added functions:
def load_dataset_activeTime(df)
def clean_dataset_0_1_1(df)
def clean_dataset_0_1_5(df)
def clean_activeTime_zeros(data)
def AnalyzeUser(data, u)

"""

import json
import os
import pandas as pd
import numpy as np
import ExplicitMF as mf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# ADDED
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def load_data(path):
    """
        Load events from files and convert to dataframe.
    """
    map_lst=[]
    for f in os.listdir(path):
        file_name=os.path.join(path,f)
        if os.path.isfile(file_name):
            for line in open(file_name):
                obj = json.loads(line.strip())
                if not obj is None:
                    map_lst.append(obj)
    return pd.DataFrame(map_lst) 

def statistics(df):
    """
        Basic statistics based on loaded dataframe
    """
    total_num = df.shape[0]
    
    print("Total number of events(front page incl.): {}".format(total_num))
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    df_ref = df[df['documentId'].notnull()]
    num_act = df_ref.shape[0]
    
    print("Total number of events(without front page): {}".format(num_act))
    num_docs = df_ref['documentId'].nunique()
    
    print("Total number of documents: {}".format(num_docs))
    print('Sparsity: {:4.3f}%'.format(float(num_act) / float(1000*num_docs) * 100))
    df_ref.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    print("Total number of events(drop duplicates): {}".format(df_ref.shape[0]))
    print('Sparsity (drop duplicates): {:4.3f}%'.format(float(df_ref.shape[0]) / float(1000*num_docs) * 100))
    
    user_df = df_ref.groupby(['userId']).size().reset_index(name='counts')
    print("Describe by user:")
    print(user_df.describe())
        
def load_dataset(df):
    """
        Convert dataframe to user-item-interaction matrix, which is used for 
        Matrix Factorization based recommendation.
        In rating matrix, clicked events are refered as 1 and others are refered as 0.
    """
    df = df[~df['documentId'].isnull()]
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df = df.sort_values(by=['userId', 'time'])
    n_users = df['userId'].nunique()
    n_items = df['documentId'].nunique()

    ratings = np.zeros((n_users, n_items))
    new_user = df['userId'].values[1:] != df['userId'].values[:-1]
    new_user = np.r_[True, new_user]
    df['uid'] = np.cumsum(new_user)
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
    df = pd.merge(df, new_df, on='documentId', how='outer')
    df_ext = df[['uid', 'tid']]
    
    for row in df_ext.itertuples():
        ratings[row[1]-1, row[2]-1] = 1.0
    return ratings 


# START OF ADDED FUNCTIONS

def load_dataset_activeTime(df):
    """
        Convert dataframe to user-item-interaction matrix
        Rows are userId, columns are documentId, value is sum of 
        activeTime for each unique user-document, that is
        if a user watched an article more than once, the total
        activeTime is given.
    """
    df = df[~df['documentId'].isnull()]
    df = df.sort_values(by=['userId', 'time'])
    newset = df.loc[:,('userId', 'documentId', 'activeTime')]
    newset.replace('None', np.nan, inplace=True)
    newset2 = newset.dropna()
    
    # Sum up all activeTime for one particular user and document
    df1 = newset2.groupby(['userId', 'documentId']).sum()
  
    # Unstack to turn userId into rows and documentId into columns
    return df1.unstack()

def clean_dataset_0_1_1(df):
    """
    -1 : negative by very short activeTime
    1  : positive by relatively higher activeTime
    0  : neutral, as not watching an article does not imply dislike
    
    Takes a dataset from load_dataset_activeTime(df).
    Calculates mean activeTime row-wise (per user) and "normalises" each
    users activeTime by taking activeTime/mean.
    This is done with NaN in the table, as they do not inflict on 
    these calculations.
    
    Then 50% below the median (after normalisation) for 
    the whole df is set to -1 (as being very short activeTime), 
    all above is set to 1.
    
    The dataset can then be run through the clean_zeros(df) which
    outputs a numpy version of the dataset, replacing all NaN
    with 0.
    """
    
    # Normalize activeTime across each users mean activeTime
    df = df * (1.0/df.mean())
    
    # Set value to distinguish between like and dislike
    setValue = df.median().median() * 0.5
    
    # Avoid changing NaN - only cells with floats
    mask = df.activeTime.isnull()
    df.activeTime[~mask] = np.where(df.activeTime < setValue,  -1.0, 1.0)

    return df

def clean_dataset_0_1_5(df):
 
    # HERE YOU CAN TWEAK TIMESTEPS
    # Set value to distinguish between like and dislike
    # Adjust these as suited!!!
    setValue1 = 5.0   # -1
    setValue2 = 15.0  #  1
    setValue3 = 25.0  #  2
    setValue4 = 135.0  #  3
    setValue5 = 245.0  #  4
    setValue6 = 600.0  #  5   (should be higer than the max-value in activeTime)
    
    # DO NOT CHANGE ANYTHING BELOW (use setValue to tweak!)
    # Use mask to avoid changing NaN - only cells with floats
    mask = df.activeTime.isnull()
    
    # This is working:
    #df.activeTime[~mask] = np.where(df.activeTime < setValue6, -1, 1)
    df.activeTime[~mask] = np.where(df.activeTime < setValue1,  9000, df.activeTime)
    df.activeTime[~mask] = np.where(df.activeTime < setValue2,  1000, df.activeTime)
    df.activeTime[~mask] = np.where(df.activeTime < setValue3,  2000, df.activeTime)
    df.activeTime[~mask] = np.where(df.activeTime < setValue4,  3000, df.activeTime)
    df.activeTime[~mask] = np.where(df.activeTime < setValue5,  4000, df.activeTime)
    df.activeTime[~mask] = np.where(df.activeTime < setValue6,  5000, df.activeTime)

    df.activeTime[~mask] = np.where(df.activeTime == 9000,  -1000, df.activeTime)
    
    df.activeTime[~mask] = np.where(df.activeTime < 5001,  df.activeTime*0.001, df.activeTime)
    
    return df
    
def clean_activeTime_zeros(data):
    npSet = data.to_numpy()
    # Flipp all nan to 0
    np.nan_to_num(npSet,0)
    # Return a numpy array with nan replaced by zero
    return npSet

def getrecommendations(user, pivot_table, topRecs, number_of_recs = 30):
    assert type(pivot_table) is pd.core.frame.DataFrame, "Fail to assert (in getrecommendations) pivot_table as pandas dataframe"
    if (user > pivot_table.shape[0]):
        print('New User, Out of range'.format(pivot_table.shape[0]))
    else:
        #print("These are all the documents you have viewed in the past: \n\n{}".format('\n'.join(read[user])))
        #print()
        print("We recommend you these documents \n")
        
    for k,v in topRecs.items():
        if user == k:
            for i in v[:number_of_recs]:
                print('{} with similarity: {:.4f}'.format(i[0], 1 - i[1]))


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    
    return sqrt(mean_squared_error(prediction, ground_truth))


def AnalyzeUser(df, u=1):
    docuTitle = df[['documentId', 'title', 'time']].copy()
    docuTitle = docuTitle[~docuTitle['documentId'].isnull()]
    docuTitle.drop_duplicates(subset=['documentId'], inplace=True)
    
    docuTitle.insert(0, 'DocuID', range(0, 20344))
    
    users = df[['userId']].copy()
    users = users[~users['userId'].isnull()]
    users.drop_duplicates(subset=['userId'], inplace=True)
    users.insert(0, 'UserID', range(0, 1000))
    event_logs = df[['userId', 'documentId', 'activeTime']].copy()
    event_logs = event_logs[~event_logs['activeTime'].isnull()]
    event_logs = event_logs[~event_logs['documentId'].isnull()]
    
    # TODO EXPLAIN ALL BELOW:
    column_names_to_normalize = ['activeTime']
    x = event_logs[column_names_to_normalize].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = event_logs.index)
    event_logs[column_names_to_normalize] = df_temp
    
    collabDocu = pd.merge(docuTitle, event_logs, on='documentId')
    collabUserDocu = pd.merge(users, collabDocu, on='userId')
    missing_pivot = collabUserDocu.pivot_table(values = 'activeTime', index = 'UserID', columns = 'DocuID')
    
    read = {}
    rows_indexes = {}
    for i,row in missing_pivot.iterrows():
        rows = [x for x in range(0,len(missing_pivot.columns))]
        combine = list(zip(row.index, row.values, rows))
        readd = [(x,z) for x,y,z in combine if str(y) != 'nan']
        index = [i[1] for i in readd]
        row_names = [i[0] for i in readd]
        rows_indexes[i] = index
        read[i] = row_names
    
    pivot_table = collabUserDocu.pivot_table(values = 'activeTime', index = 'UserID', columns = 'DocuID').fillna(0)
    pivot_table = pivot_table.apply(np.sign)
    
    assert type(pivot_table) is pd.core.frame.DataFrame, "Fail to assert (1) pivot_table as pandas dataframe"
    
    notread = {}
    notread_indexes = {}
    for i,row in pivot_table.iterrows():
        rows = [x for x in range(0,len(missing_pivot.columns))]
        combine = list(zip(row.index, row.values, row))
        idx_row = [(idx,col) for idx, val, col in combine if not val > 0]
        indices = [i[1] for i in idx_row]
        row_names = [i[0] for i in idx_row]
        notread_indexes[i] = indices
        notread[i] = row_names

    n = 5
    cosine_knn = NearestNeighbors(n_neighbors = n, algorithm = 'brute', metric = 'cosine')
    docu_cosine_knn_fit = cosine_knn.fit(pivot_table.T.values)
    docu_distances, docu_indices = docu_cosine_knn_fit.kneighbors(pivot_table.T.values)
    
    docus_dic = {}
    copyPivot = pivot_table.copy()
    for i in range(pivot_table.T.shape[0]):
        docu_idx = docu_indices[i]
        col_names = copyPivot.T.index[docu_idx].tolist()
        docus_dic[copyPivot.T.index[i]] = col_names
        
    topRecs = {}
    for k,v in rows_indexes.items():
        docu_idx = [j for i in docu_indices[v] for j in i]
        docu_dist = [j for i in docu_distances[v] for j in i]
        combine = list(zip(docu_dist, docu_idx))
        diction = {i:d for d,i in combine if i not in v}
        zipped = list(zip(diction.keys(),diction.values()))
        sort = sorted(zipped, key = lambda x: x[1])
        recommendations = [(pivot_table.columns[i], d) for i,d in sort]
        topRecs[k] = recommendations
        
        
    docu_distances = 1 - docu_distances
    ground_truth = pivot_table.T.values[docu_distances.argsort()[0]]
    predictions = docu_distances.T.dot(pivot_table.T.values) / np.array([np.abs(docu_distances.T).sum(axis = 1)]).T
    accuracy = rmse(predictions,ground_truth)
    
    assert type(pivot_table) is pd.core.frame.DataFrame, "Fail to assert (2) pivot_table as pandas dataframe"
    
    getrecommendations(u, pivot_table, topRecs)
    
    return accuracy


    
# END OF ADDED FUNCTIONS

    
def train_test_split(ratings, fraction=0.2):
    """Leave out a fraction of dataset for test use"""
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        size = int(len(ratings[user, :].nonzero()[0]) * fraction)
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=size, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
    return train, test

def evaluate(pred, actual, k):
    """
    Evaluate recommendations according to recall@k and ARHR@k
    """
    total_num = len(actual)
    tp = 0.
    arhr = 0.
    for p, t in zip(pred, actual):
        if t in p:
            tp += 1.
            arhr += 1./float(p.index(t) + 1.)
    recall = tp / float(total_num)
    arhr = arhr / len(actual)
    print("Recall@{} is {:.4f}".format(k, recall))
    print("ARHR@{} is {:.4f}".format(k, arhr))
    

def content_processing(df):
    """
        Remove events which are front page events, and calculate cosine similarities between
        items. Here cosine similarity are only based on item category information, others such
        as title and text can also be used.
        Feature selection part is based on TF-IDF process.
    """
    df = df[df['documentId'].notnull()]
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df['category'] = df['category'].str.split('|')
    df['category'] = df['category'].fillna("").astype('str')
    
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
    df = pd.merge(df, new_df, on='documentId', how='outer')
    df_item = df[['tid', 'category']].drop_duplicates(inplace=False)
    df_item.sort_values(by=['tid', 'category'], ascending=True, inplace=True)
    
    # select features/words using TF-IDF 
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0)
    tfidf_matrix = tf.fit_transform(df_item['category'])
    print('Dimension of feature vector: {}'.format(tfidf_matrix.shape))
    # measure similarity of two articles with cosine similarity
    
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    print("Similarity Matrix:")
    print(cosine_sim[:4, :4])
    return cosine_sim, df

def content_recommendation(df, k=20):
    """
        Generate top-k list according to cosine similarity
    """
    cosine_sim, df = content_processing(df)
    df = df[['userId','time', 'tid', 'title', 'category']]
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    print(df[:20]) # see how the dataset looks like
    pred, actual = [], []
    puid, ptid1, ptid2 = None, None, None
    for row in df.itertuples():
        uid, tid = row[1], row[3]
        if uid != puid and puid != None:
            idx = ptid1
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]
            sim_scores = [i for i,j in sim_scores]
            pred.append(sim_scores)
            actual.append(ptid2)
            puid, ptid1, ptid2 = uid, tid, tid
        else:
            ptid1 = ptid2
            ptid2 = tid
            puid = uid
    
    evaluate(pred, actual, k)
    
    
def collaborative_filtering(df):
    # get rating matrix
    ratings = load_dataset(df)
    # split ratings into train and test sets
    train, test = train_test_split(ratings, fraction=0.2)
    # train and test model with matrix factorization
    mf_als = mf.ExplicitMF(train, n_factors=40, 
                           user_reg=0.0, item_reg=0.0)
    iter_array = [1, 2, 5, 10, 25, 50, 100]
    mf_als.calculate_learning_curve(iter_array, test)
    # plot out learning curves
    #plot_learning_curve(iter_array, mf_als)
    

def plot_learning_curve(iter_array, model):
    """ Plot learning curves (hasn't been tested) """
    plt.plot(iter_array, model.train_mse, \
             label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse, \
             label='Test', linewidth=5)

    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel('iterations', fontsize=30);
    plt.ylabel('MSE', fontsize=30);
    plt.legend(loc='best', fontsize=20);
    

if __name__ == '__main__':
    df=load_data("active1000")
    
    ###### Get Statistics from dataset ############
    print("Basic statistics of the dataset...")
    statistics(df)
    
    ###### Recommendations based on Collaborative Filtering (Matrix Factorization) #######
    print("Recommendation based on MF...")
    collaborative_filtering(df)
    
    ###### Recommendations based on Content-based Method (Cosine Similarity) ############
    print("Recommendation based on content-based method...")
    content_recommendation(df, k=20)
    
    
    
  