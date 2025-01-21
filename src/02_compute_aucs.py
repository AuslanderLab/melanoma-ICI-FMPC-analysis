import pandas as pd
import numpy as np
import scipy
from sklearn.metrics import roc_auc_score
import glob

####################################################
#                     functions                    #  
####################################################

flatten = lambda l: [item for sublist in l for item in sublist]

####################################################
#                     variables                    #  
####################################################

# The following assumes that supp_dataset_4.csv is subsetted into cohort-specific datasets

cln = pd.read_csv('data/supp_dataset_1.csv',index_col='Sample',na_values = ["N_A", "NA", ""])
pitt = pd.read_csv('data/supp_dataset_4-$PITT.csv',index_col=0)
ny = pd.read_csv('data/supp_dataset_4-$NY.csv',index_col=0)
dall = pd.read_csv('data/supp_dataset_4-$DALLAS',index_col=0)

####################################################
#                       main                       #  
####################################################

s1 = list(set(cln.index)&set(pitt.columns))
s2 = list(set(cln.index)&set(ny.columns))
s3 = list(set(cln.index)&set(dall.columns))

pitd = pitt[s1]
pitc = cln.loc[s1]

nyd=ny[s2]
nyc=cln.loc[s2]

dad=dall[s3]
dac=cln.loc[s3]

l0 = [i=='Responder' for i in pit0c['Study_Clin_Response']]
# 60
l1 = [i=='Responder' for i in pitc['Study_Clin_Response']]
# 94
l2 = [i=='Responder' for i in nyc['Study_Clin_Response']]
# 46
l3 = [i=='Responder' for i in dac['Study_Clin_Response']]
# 14

aucs = []
for i in range(len(chd.index)):
    r1 = roc_auc_score(l1,pitd.iloc[i])
    r2 = roc_auc_score(l2,nyd.iloc[i])
    r3 = roc_auc_score(l3, dad.iloc[i])
    aucs.append([r1,r2,r3])


# stopping point to do the risk score analysis
# only should use AUCs from pitt & ny then validate on dallas

auc_df = pd.DataFrame(aucs) 
auc_df.columns = ['pitt', 'nyc', 'dls']
auc_df.to_csv('results/aucs-response.csv')


l11 = [i=='AE_reported' for i in pitc['Grades_1to4_irAE_dichot']]
au =[]
for i in range(len(chd.index)):
    ir1 = roc_auc_score(l11,pitd.iloc[i])
    au.append([ir1])


au_df = pd.DataFrame(au) 
au_df.columns = ['pitt']
au_df.to_csv('results/aucs-irae.csv')


