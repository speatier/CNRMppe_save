# 
#
###############################################################
#	File to define functions
###############################################################
#
# Import MODULES 
#
# Computational modules 
import xarray as xr
import glob
import os
import numpy as np
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import re
from array import array
from pylab import *

# Plotting modules 
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import pandas.plotting
import matplotlib.ticker as ticker
# scatter plot matrix des variables quantitatives
from pandas.plotting import scatter_matrix
import seaborn as sns; sns.set()

# Scikit-learn
from sklearn import linear_model
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
#
#...................................................................
# TO TUNE A LASSO MODEL 
#
def Lasso_tunage(X,y) :
    # LassoCV: coordinate descent

    # Compute paths
    print("Computing regularization path using the coordinate descent lasso...")
    t1 = time.time()
    model = LassoCV(cv=10).fit(X, y)
    t_lasso_cv = time.time() - t1

    # Display results
    alphas = model.alphas_

    plt.figure(figsize=(10, 7))
    ymin, ymax = 0, 5
    xmin, xmax = 0, 0.5
    plt.plot(alphas, model.mse_path_, ':')
    plt.plot(alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
    #plt.axvline(model.alpha_, linestyle='--', color='k',
    #            label='alpha: CV estimate')

    plt.legend()

    plt.xlabel('alpha')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
    plt.axis('tight')
    plt.ylim(ymin, ymax)
    #plt.xlim(xmin, xmax)

    # Enregistrer la figure .....................

    #plt.savefig("/data/home/globc/peatier/figures/PPE_Lasso_CV.png", 
    #    orientation='portrait', bbox_inches='tight', pad_inches=0.1)

    # Show la figure .................
    plt.show()
    
    # LassoCV: coordinate descent

    # Compute paths
    print("Computing regularization path using the coordinate descent lasso...")
    t1 = time.time()
    model = LassoCV(cv=5).fit(X, y)
    t_lasso_cv = time.time() - t1

    # Display results
    alphas = model.alphas_

    plt.figure(figsize=(10, 7))
    ymin, ymax = 0, 5
    xmin, xmax = 0, 0.5
    plt.plot(alphas, model.mse_path_, ':')
    plt.plot(alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
    #plt.axvline(model.alpha_, linestyle='--', color='k',
    #            label='alpha: CV estimate')

    plt.legend()

    plt.xlabel('alpha')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
    plt.axis('tight')
    plt.ylim(ymin, ymax)
    #plt.xlim(xmin, xmax)

    # Enregistrer la figure .....................

    #plt.savefig("/data/home/globc/peatier/figures/PPE_Lasso_CV5.png", 
    #    orientation='portrait', bbox_inches='tight', pad_inches=0.1)

    # Show la figure .................
    plt.show()
    
#
# ....................................................................................
# TO CREATE A LASSO MODEL
#
def Lasso_model(X,y,alpha, nb_p_list) :
    
    # Perform the lasso multi linear regression with the alpha found before
    lasso = Lasso(alpha=alpha)

    lasso.fit(X, y)

    print(lasso)
    print('Intercept: \n', lasso.intercept_)
    print('Coefficients: \n', lasso.coef_)
    print('Score: \n', lasso.score(X, y))
    
    #Coeffs = pd.DataFrame([lasso.coef_], columns=param_names).iloc[0]
    Coeffs = pd.DataFrame([lasso.coef_]).iloc[0]
    Coeffs
    #Coeffs_sorted = Coeffs.sort_values()
    #Coeffs_sorted
    
    # Let's write the equation : 
    X_df = pd.DataFrame(data=X)
    R = lasso.intercept_

    N=len(X_df.values)
    tmp = [0]*N
    y_eq = [0]*N
    i=0
    Ycpt=0
    while i<N:
        tmp[i] = Coeffs.values*X_df.iloc[i]
        y_eq[i] = tmp[i].sum()+R
        i+=1
    
    #y_eq 
    
    y_true = y

    DFYeq_lasso = pd.DataFrame([y_true, y_eq], index=['y_true', 'y_eq']).transpose()
    DFYeq_lasso['members'] = nb_p_list
    DFYeq_lasso['members_0'] = range(0,102,1)
    #DF=DFYeq.sort_values(by='y_true')
    return DFYeq_lasso
#
# ....................................................................................
# PLOT MODEL SKILL WITH CORRELATION
#
def plot_model_skill(df, title, xmin, xmax, ymin, ymax, name) :

    ax = plt.gca()
    title = title

    diag = pd.DataFrame(range(-100,100,1))
    diag['x'] = diag[0]
    diag['y'] = diag[0]
    diag = diag.drop(columns = 0)
    diag.plot(kind='line', x='x', y='y', color='gray', alpha=0.5, legend = False,linestyle='-.', ax = ax)


    df.plot(kind='scatter', x='y_true', y='y_eq', color='black', figsize=(10, 7), 
                     style='.', ax = ax)
    #plt.plot(X_test, y_pred_lasso, color='blue', linewidth=1.0)

    plt.xlabel('y_true')
    plt.ylabel('y_eq')
    plt.title(title)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # Enregistrer la figure .....................

    plt.savefig("/data/home/globc/peatier/figures/"+name, 
        orientation='portrait', bbox_inches='tight', pad_inches=0.1)

    # Show la figure .................
    plt.show() 
#
# ........................................................................................
# PLOT MODEL SKILL WITH LINES 
#
def plot_model_skill_lines(df, title, xlabel, ylabel, ymin, ymax) :
    
    # Plot y_pred and y_test  
    fig, ax = plt.subplots(figsize=(15,10))
    ax = sns.lineplot(x="members_0", y="y_eq", data=df, color='navy',ax=ax)
    ax = sns.lineplot(x="members_0", y="y_true", data=df, color='red',ax=ax)
    plt.title(title, fontsize=20)
    plt.legend(['y_eq','y_true'],fontsize=15)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.ylim(ymin, ymax)
    ax.tick_params(axis='both', labelsize=15)
#
# .....................................................................................
# TO USE LASSO MODEL FOR PREDICTION
#
def Lasso_pred(LHS, X, y, alpha) :
    
    LHS_df = pd.DataFrame(LHS)

    lhs = LHS_df.values
    #LHS_df
    
    # Perform the lasso multi linear regression with the alpha found before
    lasso = Lasso(alpha=alpha)

    lasso.fit(X, y)

    print(lasso)
    print('Intercept: \n', lasso.intercept_)
    print('Coefficients: \n', lasso.coef_)
    print('Score: \n', lasso.score(X, y))
    
    #Coeffs = pd.DataFrame([lasso.coef_], columns=param_names).iloc[0]
    Coeffs = pd.DataFrame([lasso.coef_]).iloc[0]
    Coeffs
    #Coeffs_sorted = Coeffs.sort_values()
    #Coeffs_sorted
    
    # Let's use the model equation : 

    X_df = pd.DataFrame(data=X)
    R = lasso.intercept_

    N=len(LHS_df.values)
    tmp = [0]*N
    y_pred = [0]*N
    i=0
    Ycpt=0
    while i<N:
        tmp[i] = Coeffs.values*LHS_df.iloc[i]
        y_pred[i] = tmp[i].sum()+R
        i+=1
    
    y_pred
    
    members = arange(102,100102,1)
    DFYpred_lasso = pd.DataFrame([y_pred, members], index=["y_pred", "members"]).transpose()
    return DFYpred_lasso
#
# ....................................................................................
# TO CREATE A MULTI LINEAR REGRESSION MODEL 
#
def MultiLinReg_model(X,y,param_names, nb_p_list) :
    
    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    print('Score: \n', regr.score(X, y))
    #print('Score: \n', regr.score(X_test, y_test))
    
    Coeffs = pd.DataFrame([regr.coef_]*30, columns=param_names).iloc[0]
    #Coeffs = pd.DataFrame([regr.coef_]).iloc[0]
    Coeffs
    #Coeffs_sorted = Coeffs.sort_values()
    #Coeffs_sorted
    
    # Let's write the equation : 
    X_df = pd.DataFrame(data=X)
    R = regr.intercept_

    N=len(X_df.values)
    tmp = [0]*N
    y_eq = [0]*N
    i=0
    Ycpt=0
    while i<N:
        tmp[i] = Coeffs.values*X_df.iloc[i]
        y_eq[i] = tmp[i].sum()+R
        i+=1    
    #y_eq 
    
    y_true = y

    DFYeq = pd.DataFrame([y_true, y_eq], index=['y_true', 'y_eq']).transpose()
    DFYeq['members'] = nb_p_list
    DFYeq['members_0'] = range(0,102,1)
    #DF=DFYeq.sort_values(by='y_true')
    return DFYeq
#
# ...................................................................................
# TO USE MULTILINREG FOR PREDICTIONS 
#
def MultiLinReg_pred(LHS, X ,y, param_names) :
    
    LHS_df = pd.DataFrame(LHS)

    lhs = LHS_df.values
    #LHS_df
    
    # Let's use the model equation : 

    X_df = pd.DataFrame(data=X)
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    R = regr.intercept_
    Coeffs = pd.DataFrame([regr.coef_]*30, columns=param_names).iloc[0]

    N=len(LHS_df.values)
    tmp = [0]*N
    y_pred = [0]*N
    i=0
    Ycpt=0
    while i<N:
        tmp[i] = Coeffs.values*LHS_df.iloc[i]
        y_pred[i] = tmp[i].sum()+R
        i+=1
    
    #y_pred
    members = arange(102,100102,1)
    DFYpred = pd.DataFrame([y_pred, members], index=["y_pred", "members"]).transpose()
    return DFYpred
#
