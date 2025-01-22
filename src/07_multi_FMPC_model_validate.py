####################################################
#                      readme                      #  
####################################################

# use genetic algorithm for feature selection
# followed by knn algorithm

####################################################
#                        env                       #  
####################################################

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler # use because outliers
import math
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

####################################################
#                     functions                    #  
####################################################

std_scaler = StandardScaler()

####################################################
#                    variables                     #  
####################################################

cln = pd.read_csv("supp_dataset_1.csv", index_col = 0)
cln['Response'] = np.where(cln['Study_Clin_Response'] == 'Responder', 0, 1)

pitt_full = pd.read_csv("supp_dataset_4-$PITT.csv", index_col = 0)
ny_full = pd.read_csv("supp_dataset_4-$NY.csv", index_col = 0)
dallas_full = pd.read_csv("supp_dataset_4-$DALLAS.csv", index_col = 0)
houston_full = pd.read_csv("supp_dataset_4-$HOUSTON.csv", index_col = 0)

full_sig = [3552, 5575, 6662, 1878, 1195, 4285, 2549, 9543, 6380, 3085]

####################################################
#                       main                       #  
####################################################

# run through to see best 
rf_params = {'max_depth' : [2,3,4,5,6,7,8],
             'bootstrap' : [True],
             'max_samples' : [112],
             'random_state' : [i for i in range(0, 1000)]}

pitt = pitt_full.loc[pitt_full.index.isin(full_sig), :]
pitt = pitt.transpose()
cln_pitt = cln[cln.index.isin(pitt.index)].reindex(pitt.index)
x_pitt = (pitt > pitt.median()).astype('int')
x_pitt = pd.DataFrame(x_pitt, columns = pitt.columns)
y_pitt = np.where(cln_pitt['Study_Clin_Response'] == 'Non_Responder', 1, 0)
# new york
ny = ny_full.loc[ny_full.index.isin(full_sig), :]
ny = ny.transpose()
cln_ny = cln[cln.index.isin(ny.index)].reindex(ny.index)
x_ny = (ny > ny.median()).astype('int')
x_ny = pd.DataFrame(x_ny, columns = ny.columns)
y_ny = np.where(cln_ny['Study_Clin_Response'] == 'Non_Responder', 1, 0)
# dallas
dallas = dallas_full.loc[dallas_full.index.isin(full_sig), :]
dallas = dallas.transpose()
cln_dallas = cln[cln.index.isin(dallas.index)].reindex(dallas.index)
x_dallas = (dallas > dallas.median()).astype('int')
x_dallas = pd.DataFrame(x_dallas, columns = dallas.columns)
y_dallas = np.where(cln_dallas['Study_Clin_Response'] == 'Non_Responder', 1, 0)
# model
x_train = pd.concat([x_pitt, x_ny])
y_train = np.concatenate((y_pitt, y_ny), axis = None)

################### just run once ##################
#rf = RandomForestClassifier()
#grid_res = GridSearchCV(rf, rf_params, cv = 5).fit(x_train, y_train)
#results = pd.DataFrame(grid_res.cv_results_)
#results.loc[:, 'mean_test_score'] *= 100
#results.to_csv('results/rf_grid_results.csv')
####################################################


x_train = pd.concat([x_pitt, x_ny, x_dallas])
y_train = np.concatenate((y_pitt, y_ny, y_dallas), axis = None)
# houston
houston = houston_full.loc[houston_full.index.isin(full_sig), :]
houston = houston.transpose()
cln_houston = cln[cln.index.isin(houston.index)].reindex(houston.index)
x_houston = (houston > houston.median()).astype('int')
x_houston = pd.DataFrame(x_houston, columns = houston.columns)
y_houston = np.where(cln_houston['Study_Clin_Response'] == 'Non_Responder', 1, 0)

train_roc = []
valid_roc = []
test_roc = []
feats = []

for i in range(10000,11000):
    rf_model = RandomForestClassifier(random_state = i, max_depth = 6)
    rf_model.fit(x_train, y_train)
    train_predicted = rf_model.predict_proba(x_train)
    dallas_predicted = rf_model.predict_proba(x_dallas)
    houston_predicted = rf_model.predict_proba(x_houston)
    train_roc.append(roc_auc_score(y_train, train_predicted[:,1]))
    valid_roc.append(roc_auc_score(y_dallas, dallas_predicted[:, 1]))
    test_roc.append(roc_auc_score(y_houston, houston_predicted[:,1]))
    feats.append([f for f in rf_model.feature_importances_])


df = pd.DataFrame(feats, columns = full_sig)
df['train'] = train_roc
df['validate'] = valid_roc
df['test'] = test_roc
df.to_csv("results/random_forest_final_result.csv")

# Exploratory
# Iteratively remove ones that make the model worse -> 
full_sig = [3552, 5575, 6662, 1195, 4285, 2549, 9543, 6380, 3085]
full_sig = [3552, 5575, 6662, 1195, 4285, 2549, 9543, 6380]
full_sig = [3552, 6662, 1195, 4285, 2549, 9543, 6380]
full_sig = [3552, 1195, 4285, 2549, 9543, 6380]

# using all cohorts to train
# subset to just significant clusters
for i in range(0, len(full_sig)):
    sig = full_sig[:i] + full_sig[i+1:]
    print(sig)
    print("Without " + str(full_sig[i]))
    # pitt
    pitt = pitt_full.loc[pitt_full.index.isin(sig), :]
    pitt = pitt.transpose()
    cln_pitt = cln[cln.index.isin(pitt.index)].reindex(pitt.index)
    x_pitt = (pitt > pitt.median()).astype('int')
    x_pitt = pd.DataFrame(x_pitt, columns = pitt.columns)
    y_pitt = np.where(cln_pitt['Study_Clin_Response'] == 'Non_Responder', 1, 0)
    # new york
    ny = ny_full.loc[ny_full.index.isin(sig), :]
    ny = ny.transpose()
    cln_ny = cln[cln.index.isin(ny.index)].reindex(ny.index)
    x_ny = (ny > ny.median()).astype('int')
    x_ny = pd.DataFrame(x_ny, columns = ny.columns)
    y_ny = np.where(cln_ny['Study_Clin_Response'] == 'Non_Responder', 1, 0)
    # dallas
    dallas = dallas_full.loc[dallas_full.index.isin(sig), :]
    dallas = dallas.transpose()
    cln_dallas = cln[cln.index.isin(dallas.index)].reindex(dallas.index)
    x_dallas = (dallas > dallas.median()).astype('int')
    x_dallas = pd.DataFrame(x_dallas, columns = dallas.columns)
    y_dallas = np.where(cln_dallas['Study_Clin_Response'] == 'Non_Responder', 1, 0)
    # houston
    houston = houston_full.loc[houston_full.index.isin(sig), :]
    houston = houston.transpose()
    cln_houston = cln[cln.index.isin(houston.index)].reindex(houston.index)
    x_houston = (houston > houston.median()).astype('int')
    x_houston = pd.DataFrame(x_houston, columns = houston.columns)
    y_houston = np.where(cln_houston['Study_Clin_Response'] == 'Non_Responder', 1, 0)
    # model
    x_train = pd.concat([x_pitt, x_ny, x_dallas])
    y_train = np.concatenate((y_pitt, y_ny, y_dallas), axis = None)
    rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    rf_model.fit(x_train, y_train)
    train_predicted = rf_model.predict_proba(x_train)
    houston_predicted = rf_model.predict_proba(x_houston)
    print(roc_auc_score(y_train, train_predicted[:,1]))
    print(roc_auc_score(y_houston, houston_predicted[:,1]))


