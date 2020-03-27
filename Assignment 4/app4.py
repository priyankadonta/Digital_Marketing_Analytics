##import streamlit as st
"""
import pandas as pd
import numpy as np

data=pd.read_csv("snackratings.csv")
data.head()

chart_data = pd.DataFrame(
     data,
     columns=['a', 'b', 'c'])

st.bar_chart(data)
"""

import sys
sys.path.append("../../")
import time
import os
import itertools
from reco_utils.dataset.python_splitters import (
    python_random_split, 
    python_chrono_split, 
    python_stratified_split
)
import pandas as pd
import numpy as np
#import papermill as pm
import torch, fastai
from fastai.collab import EmbeddingDotBias, collab_learner, CollabDataBunch, load_learner

from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.recommender.fastai.fastai_utils import cartesian_product, score
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.evaluation.python_evaluation import rmse, mae, rsquared, exp_var

import streamlit as st
from fastai.collab import *
from fastai.tabular import *
import seaborn as sns
import matplotlib.pyplot as plt




USER, ITEM, RATING, TIMESTAMP, PREDICTION, TITLE = 'UserId', 'SnackId', 'Rating', 'Timestamp', 'Prediction', 'Title'

# SIDEBARS
st.sidebar.header("Assignment 4")
st.sidebar.text("Fast Ai using Embedded Dot Bias")


# top k items to recommend
TOP_K = 10

# Slider
#level = st.slider("Set the k-factor",1,5)
#st.write("You selected this option ",level)
#TOP_K=level

nfactors= st.selectbox("Select Nfactors",[10,20,30,40])
st.write("You selected this option ",nfactors)
N_FACTORS=nfactors

epoch= st.selectbox("Select epoch",[1,2,3,4,5])
st.write("You selected this option ",epoch)
EPOCHS=epoch
if  st.button("Submit"):
        st.write("You selected this option ",nfactors)
        st.write("You selected this option ",epoch)


        USER, ITEM, RATING, TIMESTAMP, PREDICTION, TITLE = 'UserId', 'SnackId', 'Rating', 'Timestamp', 'Prediction', 'Title'

        data = pd.read_csv("snackratings.csv")
        #data.head


        # # make sure the IDs are loaded as strings to better prevent confusion with embedding ids
        data[USER] = data[USER].astype('str')
        data[ITEM] = data[ITEM].astype('str')

        data.head()

        # data = pd.read_csv("/content/snackratings.csv")
        # data.head()

        print(
            "Total number of ratings are\t{}".format(data.shape[0]),
            "Total number of users are\t{}".format(data[USER].nunique()),
            "Total number of items are\t{}".format(data[ITEM].nunique()),
            sep="\n"
        )

        #st.subheader("data loaded")

        data_train, data_test = python_random_split(data, ratio=0.7)

        split =data_train.shape[0], data_test.shape[0]
        st.write("Splitting_Ratio:",split)

        data = CollabDataBunch.from_df(data_train, seed=42, valid_pct=0.1)

        y_range = [0.5,5.5]
        st.write(y_range)

        factor=N_FACTORS
        st.write("No. of factors:",factor)

        learn = collab_learner(data, n_factors=factor, y_range=y_range, wd=1e-1)
        
        learn.model

        #st.subheader("data loaded")

        fit_onecycle=learn.fit_one_cycle(5, 3e-4)
        st.write(fit_onecycle)


        #st.subheader("data loaded")

        learn.recorder.plot()
        st.pyplot()


        learn.export('export.pkl')

        learner = load_learner(path="/Users/rohittikle/Documents/GitHub/recommenders/")
        #st.write(learner)
        total_users, total_items = learner.data.train_ds.x.classes.values()
        total_items = total_items[1:]
        total_users = total_users[1:]
        #st.write("Items:",total_items,"Users:",total_users)
        #st.write("Users:",total_users)

        test_users = data_test[USER].unique()
        test_users = np.intersect1d(test_users, total_users)
        #st.write("Test Users:",test_users)

        users_items = cartesian_product(np.array(test_users),np.array(total_items))
        users_items = pd.DataFrame(users_items, columns=[USER,ITEM])
       # st.write("Users:",users_items)
        training_removed = pd.merge(users_items, data_train.astype(str), on=[USER, ITEM], how='left')
        training_removed = training_removed[training_removed['Rating'].isna()][[USER,ITEM]]
       # st.write("Training:",training_removed)
        start_time = time.time()
       # st.write("Start Time:",start_time)

        top_k_scores = score(learner, 
                             test_df=training_removed,
                             user_col=USER, 
                             item_col=ITEM, 
                             prediction_col=PREDICTION)
        #st.write("TOP K SCORES:",top_k_scores)
        test_time = time.time() - start_time
       # st.write("Test_Time:",test_time)
        print("Took {} seconds for {} predictions.".format(test_time, len(training_removed)))

        top_k_scores[USER] = top_k_scores['UserId']
        top_k_scores[ITEM] = top_k_scores['SnackId']
        top_k_scores[PREDICTION] = top_k_scores['Prediction']

        top_k_scores.head()
       # st.write("Top k Scores:",top_k_scores)

        data_test.head()
      #  st.write("Data_Test:",data_test)

        eval_map = map_at_k(data_test, top_k_scores, col_user=USER, col_item=ITEM, 
                            col_rating=RATING, col_prediction=PREDICTION,
                            relevancy_method="top_k", k=TOP_K)
        st.write("MAP:",eval_map)
        eval_ndcg = ndcg_at_k(data_test, top_k_scores, col_user=USER, col_item=ITEM, 
                              col_rating=RATING, col_prediction=PREDICTION, 
                              relevancy_method="top_k", k=TOP_K)
        st.write("NDCG:",eval_ndcg)
        eval_precision = precision_at_k(data_test, top_k_scores, col_user=USER, col_item=ITEM, 
                                        col_rating=RATING, col_prediction=PREDICTION, 
                                        relevancy_method="top_k", k=TOP_K)
        st.write("Precision:",eval_precision)
        eval_recall = recall_at_k(data_test, top_k_scores, col_user=USER, col_item=ITEM, 
                                  col_rating=RATING, col_prediction=PREDICTION, 
                                  relevancy_method="top_k", k=TOP_K)
        st.write("Recall:",eval_recall)
        print("Model:\t" + learn.__class__.__name__,
              "Top K:\t%d" % TOP_K,
              "MAP:\t%f" % eval_map,
              "NDCG:\t%f" % eval_ndcg,
              "Precision@K:\t%f" % eval_precision,
              "Recall@K:\t%f" % eval_recall, sep='\n')

# scores = score(learner, 
#                test_df=data_test.copy(), 
#                user_col=USER, 
#                item_col=ITEM, 
#                prediction_col=PREDICTION)

#scores = score(learner, 
                    # test_df=training_removed,
                  #   user_col=USER, 
                    # item_col=ITEM, 
                    # prediction_col=PREDICTION)

# top_k_scores = score(learner, 
#                      test_df=training_removed,
#                      user_col=USER, 
#                      item_col=ITEM, 
#                      prediction_col=PREDICTION)
#scores.head()

#scores[USER] = scores['UserId']
#scores[ITEM] = scores['SnackId']
#scores[PREDICTION] = scores['Prediction']

#eval_r2 = rsquared(data_test, scores, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION)
#eval_rmse = rmse(data_test, scores, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION)
#eval_mae = mae(data_test, scores, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION)
#eval_exp_var = exp_var(data_test, scores, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION)

#print("Model:\t" + learn.__class__.__name__,
#      "RMSE:\t%f" % eval_rmse,
#      "MAE:\t%f" % eval_mae,
#      "Explained variance:\t%f" % eval_exp_var,
#      "R squared:\t%f" % eval_r2, sep='\n')
