####################################################
#                      readme                      #  
####################################################

# use genetic algorithm for feature selection
# using just binary above/below median

####################################################
#                        env                       #  
####################################################

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler # use because outliers
import math
import random

####################################################
#                     functions                    #  
####################################################

std_scaler = StandardScaler()

classifiers = ['LinearSVM', 'RadialSVM', 
               'Logistic',  'RandomForest', 
               'AdaBoost',  'DecisionTree', 
               'KNeighbors','GradientBoosting']

models = [svm.SVC(kernel='linear'),
          svm.SVC(kernel='rbf'),
          LogisticRegression(max_iter = 1000),
          RandomForestClassifier(n_estimators = 100, random_state = 42),
          AdaBoostClassifier(random_state = 42),
          DecisionTreeClassifier(random_state = 42),
          KNeighborsClassifier(),
          GradientBoostingClassifier(random_state = 42)]


def acc_score(X_train, Y_train, X_test, Y_test):
    Score = pd.DataFrame({"Classifier":classifiers})
    j = 0
    acc = []
    for i in models:
        model = i
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        acc.append(roc_auc_score(Y_test, predictions))
        j = j+1     
    Score["Accuracy"] = acc
    Score.sort_values(by = "Accuracy", ascending = False, inplace = True)
    Score.reset_index(drop = True, inplace = True)
    return Score


#' size = size of population
#' nfeat = number of features to use
#' total_features = total number of starting features
def initialize_population(size, nfeat, total_feat):
    population = []
    for i in range(size):
        n = total_feat - np.random.choice(nfeat)
        chromosome = np.ones(total_feat, dtype = bool)     
        chromosome[:n] = False  
        # print(j, chromosome.sum())
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population, X_train, X_test, Y_train, Y_test):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:,chromosome],Y_train)         
        predictions = logmodel.predict(X_test.iloc[:,chromosome])
        scores.append(roc_auc_score(Y_test, list(predictions)))
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(-scores)                                    
    return list(scores[inds]), list(population[inds]) 


# always chooses top half of population
def selection(pop_after_fit, scores):
    population_nextgen = [i for i,j in zip(pop_after_fit, scores) if j > np.median(scores)]
    return population_nextgen


def crossover(pop_after_sel, size):
    pop_nextgen = pop_after_sel
    for i in range(0, size):
        new_par = np.random.choice(list(range(0, len(pop_after_sel))), size = 2, replace = False)
        prob = [np.random.choice([0,1], p = [0.5,0.5]) for i in pop_after_sel[0]]
        child = np.array([pop_nextgen[new_par[0]][i] if prob[i] == 0 else pop_nextgen[new_par[1]][i] for i in range(0, len(prob))])
        if child.sum() > 1:
            pop_nextgen.append(child)
    return pop_nextgen


def mutation(pop_after_cross, mutation_rate, total_feat):   
    mutation_range = int(mutation_rate*total_feat)
    pop_next_gen = []
    for n in range(0, len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = [] 
        for i in range(0,mutation_range):
            pos = random.randint(0, total_feat-1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]  
        pop_next_gen.append(chromo)
    return pop_next_gen


def generations(label, size, nfeat, mutation_rate, n_gen, X_train, X_test, Y_train, Y_test):
    best_chromo = []
    best_score = []
    population_nextgen = initialize_population(size, nfeat, X_train.shape[1])
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen, X_train = X_train, X_test = X_test, 
                                        Y_train = Y_train, Y_test = Y_test)
        print('Best score in generation', i+1, ':', scores[0])  
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
        # print([i.sum() for i in pop_after_fit])
        pop_after_sel = selection(pop_after_fit, scores)
        if len(pop_after_sel) >= 2:
            population_nextgen = crossover(pop_after_sel, size = 20)
        else:
            break
    return best_chromo,best_score


####################################################
#                    variables                     #  
####################################################

cln = pd.read_csv("supp_dataset_1.csv", index_col = 0)
cln['Response'] = np.where(cln['Study_Clin_Response'] == 'Responder', 0, 1)

pitt = pd.read_csv("supp_dataset_4-$PITT.csv", index_col = 0)
ny = pd.read_csv("supp_dataset_4-$NY.csv", index_col = 0)
dallas = pd.read_csv("supp_dataset_4-$DALLAS.csv", index_col = 0)

sig = pd.read_csv("results/aucs_signif-mrs.csv")

####################################################
#                       main                       #  
####################################################

# subset to just significant clusters

pitt = pitt.loc[pitt.index.isin(sig['V1']), :]
pitt = pitt.transpose()
cln_pitt = cln[cln.index.isin(pitt.index)].reindex(pitt.index)
x_pitt = (pitt > pitt.median()).astype('int')
x_pitt = pd.DataFrame(x_pitt, columns = pitt.columns)
y_pitt = np.where(cln_pitt['Study_Clin_Response'] == 'Non_Responder', 1, 0)

ny = ny.loc[ny.index.isin(sig['V1']), :]
ny = ny.transpose()
cln_ny = cln[cln.index.isin(ny.index)].reindex(ny.index)
x_ny = (ny > ny.median()).astype('int')
x_ny = pd.DataFrame(x_ny, columns = ny.columns)
y_ny = np.where(cln_ny['Study_Clin_Response'] == 'Non_Responder', 1, 0)

dallas = dallas.loc[dallas.index.isin(sig['V1']), :]
dallas = dallas.transpose()
cln_dallas = cln[cln.index.isin(dallas.index)].reindex(dallas.index)
x_dallas = (dallas > dallas.median()).astype('int')
x_dallas = pd.DataFrame(x_dallas, columns = dallas.columns)
y_dallas = np.where(cln_dallas['Study_Clin_Response'] == 'Non_Responder', 1, 0)

x_train = pd.concat([x_pitt, x_ny])
y_train = np.concatenate((y_pitt, y_ny), axis = None)

acc_score(x_train, y_train, x_dallas, y_dallas)

feature_subsets = np.arange(5, 6, 1)
n_parents = math.floor(len(feature_subsets)/2)*2

# not really a log model now...
logmodel = RandomForestClassifier(n_estimators = 100, random_state = 42)

all_results = []
for i in range(0,10000):
    print(i)
    np.random.seed(i)
    chromo_df_bc,score_bc = generations(label = y_train, size = 20, nfeat = feature_subsets, 
                                        mutation_rate = 0.1, n_gen = 9, 
                                        X_train = x_train, X_test = x_dallas, Y_train = y_train, Y_test = y_dallas)
    all_results.append([chromo_df_bc, score_bc])


fw = open("results/feature_selection_results.txt", "w")
fw.write("Run\tAcc\tClusters\n")
for i in range(0, len(all_results)):
    # print(str(i) + '\t' + ','.join([str(i) for i in np.array(x_pitt.columns)[all_results[i][0][-1]]]) + "\n")
    fw.write(str(i) + '\t' +  str(all_results[i][1][-1]) + '\t' + ','.join([str(i) for i in np.array(x_pitt.columns)[all_results[i][0][-1]]]) + "\n")


fw.close()
