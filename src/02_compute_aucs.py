####################################################
#                     TRAINING                     #  
####################################################

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
# i.e. output from the 00_split_cohorts.py

cln = pd.read_csv('data/supp_dataset_1.csv',index_col='Sample',na_values = ["N_A", "NA", ""])
pitt = pd.read_csv('results/supp_dataset_4-PITT.csv',index_col=0)
ny = pd.read_csv('results/supp_dataset_4-NY.csv',index_col=0)
dall = pd.read_csv('results/supp_dataset_4-DALLAS.csv',index_col=0)

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

l1 = [i=='Responder' for i in pitc['Study_Clin_Response']]

l2 = [i=='Responder' for i in nyc['Study_Clin_Response']]

l3 = [i=='Responder' for i in dac['Study_Clin_Response']]


aucs = []
for i in range(len(pitd.index)):
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
for i in range(len(pitd.index)):
    ir1 = roc_auc_score(l11,pitd.iloc[i])
    au.append([ir1])


au_df = pd.DataFrame(au) 
au_df.columns = ['pitt']
au_df.to_csv('results/aucs-irae.csv')


####################################################
#                         STOP                     #  
####################################################

# Return after running 03_filter_aucs.R

####################################################
#                      TESTING                     #  
####################################################

# RESPONSE
hou = pd.read_csv('data/supp_dataset_7.csv',index_col=0)

s1 = list(set(cln.index)&set(hou.columns))

houd = hou[s1]
houc = cln.loc[s1]

lh = [i=='Responder' for i in houc['Study_Clin_Response']]

au = []
for i in list(hou.index):
    val = roc_auc_score(lh,houd.loc[i])
    au.append([val])

aucs = pd.DataFrame(au)
aucs['clst'] = [i.split(' ')[1].replace(':', '').strip() for i in list(hou.index)] 
aucs.to_csv('results/aucs-response-validation.csv')

# irAE
cln1 = pd.read_csv('data/supp_dataset_2.csv', index_col = 'SampleID')
rad = pd.read_csv('data/supp_dataset_10.csv', index_col = 0)
cln1['batch'] = ['B1' if 'UPCC' in name else 'B2' for name in cln1.index]

s1 = list(cln1.index[cln1['batch'] == 'B1'])
s2 = list(cln1.index[cln1['batch'] == 'B2'])
s3 = list(rad.columns)

rad1d = rad[s1]
rad1c = cln1.loc[s1]

rad2d = rad[s2]
rad2c = cln1.loc[s2]

rad3d = rad[s3]
rad3c = cln1.loc[s3]

l1 = [i=='Tox' for i in rad1c['irAE classification']]
l2 = [i=='Tox' for i in rad2c['irAE classification']]
l3 = [i=='Tox' for i in rad3c['irAE classification']]

aucs = []
for i in list(rad.index):
    r1 = roc_auc_score(l1,rad1d.loc[i])
    r2 = roc_auc_score(l2,rad2d.loc[i])
    r3 = roc_auc_score(l3,rad3d.loc[i])
    aucs.append([i.split(' ')[1].replace(':', '').strip(), r1,r2,r3])


auc_df = pd.DataFrame(aucs) 
auc_df.columns = ['clst', 'rad1', 'rad2', 'combined']
auc_df.to_csv('results/aucs-irae-validation.csv')


####################################################
#               PERMUTATION TESTING                #  
####################################################

import random

hou = pd.read_csv('data/supp_dataset_7.csv',index_col=0)

s1 = list(set(cln.index)&set(hou.columns))

houd = hou[s1]
houc = cln.loc[s1]

l1 = [i=='Responder' for i in houc['Study_Clin_Response']]

aucs = pd.DataFrame()
for j in range(0, 10000):
    print(j)
    perm_aucs = []
    random.seed(j)
    for i in list(hou.index):
        r1 = roc_auc_score(random.sample(l1, len(l1)),houd.loc[i])
        perm_aucs.append(r1)
        perm_df = pd.DataFrame({'AUC' : perm_aucs, 'cluster': i.split(' ')[1].replace(':', '')})
        perm_df['perm'] = j
    aucs = pd.concat([aucs, perm_df], axis = 0)

aucs.to_csv('results/aucs-response-perm.csv')

