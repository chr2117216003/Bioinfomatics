
# coding: utf-8

# In[ ]:

# !/use/bin/env python

import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import KFold  
from sklearn import svm
from sklearn.model_selection import train_test_split
import math
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import easy_excel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import *
import sklearn.ensemble
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB  
import subprocess
from sklearn.utils import shuffle
import itertools
from sklearn.ensemble import GradientBoostingClassifier
import sys
from sklearn.feature_selection import  f_classif
import warnings

warnings.filterwarnings('ignore')
path=""
inputname=sys.argv[1]
outputname=inputname.split('.')[0]
crossvalidation_values=int(sys.argv[2])
CPU_values=int(sys.argv[3])
name=outputname


# In[ ]:

def performance(labelArr, predictArr):
    #labelArr[i] is actual value,predictArr[i] is predict value
    TP = 0.; TN = 0.; FP = 0.; FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    if (TP + FN)==0:
        SN=0
    else:
        SN = TP/(TP + FN) #Sensitivity = TP/P  and P = TP + FN
    if (FP+TN)==0:
        SP=0
    else:
        SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    if (TP+FP)==0:
        precision=0
    else:
        precision=TP/(TP+FP)
    if (TP+FN)==0:
        recall=0
    else:
        recall=TP/(TP+FN)
    GM=math.sqrt(recall*SP)
    #MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return precision,recall,SN,SP,GM,TP,TN,FP,FN

if __name__=="__main__":
    # In[ ]:
    
    """
        cross validation and f-score and xgboost
    """
    datapath =path+outputname+".csv"
    classifier="xgboost_f-score"
    mode="crossvalidation"
    print "start"
    train_data = pd.read_csv(datapath, header=None, index_col=None)
    Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
    Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
    Y.extend(Y2)
    Y = np.array(Y)
    F, pval = f_classif(train_data, Y)
    idx = np.argsort(F)
    selected_list_=idx[::-1]
    F_sort_value=[F[e] for e in selected_list_]
    with open(classifier+mode+"all_dimension_results.txt",'w') as f:
            f.write(str(F_sort_value)+"\n")
    print F_sort_value
    with open(classifier+mode+"all_dimension_results.txt",'a') as f:
            f.write(str(selected_list_)+"\n")
    print selected_list_
    
    print "deal with data"
    selected_list_=[a  for a,b in zip(selected_list_,F_sort_value) if not math.isnan(b)]
    print selected_list_
    
    
    bestACC=0
    bestn_estimators=0
    bestmax_depth=0
    best_dimension=0
    bestlearning_rate=0
    all_dimension_results=[]
    select_list=[]
    best_savedata=""
    for select_num,temp_data in enumerate(selected_list_):
        train_data2=train_data.values
        select_list.append(int(temp_data))
        X_train=pd.DataFrame(train_data2)
        X_train=X_train.iloc[:,select_list]
        X = np.array(X_train)
        parameters = [{'n_estimators':range(1,500,10),
                          'max_depth':range(3,7,1),
                          'learning_rate':[0.2,0.3,0.4]
    #                       ,'subsample':[0.75,0.8,0.85,0.9]
                          }]
        clf = GridSearchCV(XGBClassifier(), parameters, cv=crossvalidation_values, n_jobs=CPU_values, scoring='accuracy')
        clf.fit(X, Y)
        n_estimators=clf.best_params_['n_estimators']
        max_depth=clf.best_params_['max_depth']
        learning_rate=clf.best_params_['learning_rate']
        # subsample=clf.best_params_['subsample']
        # joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
        # print clf.best_score_
        y_predict=cross_val_predict(XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate,
                                                               max_depth=max_depth),X,Y,cv=crossvalidation_values,n_jobs=CPU_values)
        # y_predict=cross_val_predict(XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate,
        #                                                        subsample=subsample,max_depth=max_depth),X,Y,cv=10,n_jobs=1)
        y_predict_prob=cross_val_predict(XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate,
                                                               max_depth=max_depth),X,Y,cv=crossvalidation_values,n_jobs=CPU_values,method='predict_proba')
        
        joblib.dump(clf,path+classifier+mode+outputname+str(select_num)+".model")
        predict_save=[Y.astype(int),y_predict.astype(int),y_predict_prob[:,1]]
        predict_save=np.array(predict_save).T
        pd.DataFrame(predict_save).to_csv(path+classifier+mode+outputname+"_"+'_predict_crossvalidation.csv',header=None,index=False)
        ROC_AUC_area=metrics.roc_auc_score(Y,y_predict)
        ACC=metrics.accuracy_score(Y,y_predict)
        precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
        F1_Score=metrics.f1_score(Y, y_predict)
        F_measure=F1_Score
        MCC=metrics.matthews_corrcoef(Y, y_predict)
        pos=TP+FN
        neg=FP+TN
        savedata=[[['xgboost'+"n_estimators:"+str(n_estimators)+"max_depth:"+str(max_depth)+"learning_rate:"+str(learning_rate),ACC,precision, recall,SN, SP, GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
        if ACC>bestACC:
            bestACC=ACC
            bestn_estimators=n_estimators
            bestlearning_rate=learning_rate
            best_savedata=savedata
            bestmax_depth=max_depth
            best_dimension=X.shape[1]
        print savedata
        print X.shape[1]
        with open(classifier+mode+"all_dimension_results.txt",'a') as f:
            f.write(str(savedata)+"\n")
        all_dimension_results.append(savedata)
    print bestACC
    print bestn_estimators
    print bestlearning_rate
    print bestmax_depth
    print best_dimension
    easy_excel.save("xgboost_crossvalidation",[str(best_dimension)],best_savedata,path+classifier+mode+'cross_validation_'+name+'.xls')

