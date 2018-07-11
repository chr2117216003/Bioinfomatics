
# coding: utf-8

# In[1]:

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
path=""
inputname=sys.argv[1]
inputname=inputname.split(".")[0]
outputname=inputname
name=outputname
CV_num=sys.argv[2]
n_jobs_value=sys.argv[3]


# In[5]:

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


# In[ ]:

"""
    jackknife svm
"""
classifier="SVM"
datapath =path+outputname+".csv"
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
X_test=None
Y_test=None
y_predict=[]
y_predict_prob=[]
print "start"
parameters = {'kernel': ['rbf'], 'C':map(lambda x:2**x,np.linspace(-2,5,7)), 'gamma':map(lambda x:2**x,np.linspace(-5,2,7))}
clf = GridSearchCV(svm.SVC(), parameters, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X,Y)
C=clf.best_params_['C']
gamma=clf.best_params_['gamma']
for index in xrange(len(train_data)):
    X_test=np.array(X[index]).reshape(1,-1)
    Y_test=np.array(Y[index]).reshape(1,-1)
    X_train=None
    Y_train=None
    if index==0:
        X_train=np.array(X[(index+1):])
        Y_train=np.array(Y[(index+1):])
    elif index==len(Y)-1:
        X_train=np.array(X[0:index])
        Y_train=np.array(Y[0:index])
    else:
        X_train=np.concatenate([np.array(X[0:index]),np.array(X[(index+1):])],axis=0)
        Y_train=np.concatenate([np.array(Y[0:index]),np.array(Y[(index+1):])],axis=0)
    svc = svm.SVC(probability=True,C=C,gamma=gamma) 
    svc.fit(X_train, Y_train)
    # joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
    # print clf.best_score_
    y_p=svc.predict(X_test)
    y_p_p=svc.predict_proba(X_test)
    y_predict.append(y_p)
    y_predict_prob.append(y_p_p)
ROC_AUC_area=metrics.roc_auc_score(Y, y_predict)
predict_save=[Y.astype(int),np.array(y_predict).astype(int)[:,0],np.array(y_predict_prob)[:,0,1]]
print np.array(y_predict_prob)[:,0,1]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict_jackknife.csv',header=None,index=False)
ACC=metrics.accuracy_score(Y,y_predict)
precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
F1_Score=metrics.f1_score(Y, y_predict)
F_measure=F1_Score
MCC=metrics.matthews_corrcoef(Y, y_predict)
pos=TP+FN
neg=FP+TN
savedata=[[[classifier+"C:"+str(C)+"gamma:"+str(gamma),ACC,precision, recall,SN, SP, GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
print savedata
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'jackknife.xls')
print "end"


# In[ ]:

"""
    cross validation GBDT
"""
classifier="GBDT"
datapath =path+outputname+".csv"
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
X_test=None
Y_test=None
y_predict=[]
y_predict_prob=[]
print "start"
GBDT = GradientBoostingClassifier(learning_rate=0.1,max_features='sqrt')
parameters = {'n_estimators':range(10,120,1),'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200),'min_samples_leaf':range(60,101,10)}
clf = GridSearchCV(GBDT, parameters, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X, Y)
n_estimators=clf.best_params_['n_estimators']
max_depth=clf.best_params_['max_depth']
min_samples_split=clf.best_params_['min_samples_split']
min_samples_leaf=clf.best_params_['min_samples_leaf']
for index in xrange(len(train_data)):
    X_test=np.array(X[index]).reshape(1,-1)
    Y_test=np.array(Y[index]).reshape(1,-1)
    X_train=None
    Y_train=None
    if index==0:
        X_train=np.array(X[(index+1):])
        Y_train=np.array(Y[(index+1):])
    elif index==len(Y)-1:
        X_train=np.array(X[0:index])
        Y_train=np.array(Y[0:index])
    else:
        X_train=np.concatenate([np.array(X[0:index]),np.array(X[(index+1):])],axis=0)
        Y_train=np.concatenate([np.array(Y[0:index]),np.array(Y[(index+1):])],axis=0)
    GBDT = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=0.1,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features='sqrt')
    GBDT.fit(X_train,Y_train)
    # joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
    # print clf.best_score_
    y_p=GBDT.predict(X_test)
    y_p_p=GBDT.predict_proba(X_test)
    y_predict.append(y_p)
    y_predict_prob.append(y_p_p)
ROC_AUC_area=metrics.roc_auc_score(Y, y_predict)
predict_save=[Y.astype(int),np.array(y_predict).astype(int)[:,0],np.array(y_predict_prob)[:,0,1]]
print np.array(y_predict_prob)[:,0,1]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict_jackknife.csv',header=None,index=False)
ACC=metrics.accuracy_score(Y,y_predict)
precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
F1_Score=metrics.f1_score(Y, y_predict)
F_measure=F1_Score
MCC=metrics.matthews_corrcoef(Y, y_predict)
pos=TP+FN
neg=FP+TN
savedata=[[[classifier+"n_estimators:"+str(n_estimators)+"max_depth:"+str(max_depth)+"min_samples_split:"+str(min_samples_split)+"min_samples_leaf:"+str(min_samples_leaf),ACC,precision, recall,SN, SP,
            GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
print savedata
print X.shape[1]
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'jackknife.xls')
print "end"


# In[ ]:

"""
    cross validation RF
"""
classifier="RF"
datapath =path+outputname+".csv"
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
y_predict=[]
y_predict_prob=[]
RF = RandomForestClassifier(max_features='sqrt')
parameters = {'n_estimators':range(10,120,1),'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200),'min_samples_leaf':range(60,101,10)}
clf = GridSearchCV(RF, parameters, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X, Y)
n_estimators=clf.best_params_['n_estimators']
max_depth=clf.best_params_['max_depth']
min_samples_split=clf.best_params_['min_samples_split']
min_samples_leaf=clf.best_params_['min_samples_leaf']
for index in xrange(len(train_data)):
    X_test=np.array(X[index]).reshape(1,-1)
    Y_test=np.array(Y[index]).reshape(1,-1)
    X_train=None
    Y_train=None
    if index==0:
        X_train=np.array(X[(index+1):])
        Y_train=np.array(Y[(index+1):])
    elif index==len(Y)-1:
        X_train=np.array(X[0:index])
        Y_train=np.array(Y[0:index])
    else:
        X_train=np.concatenate([np.array(X[0:index]),np.array(X[(index+1):])],axis=0)
        Y_train=np.concatenate([np.array(Y[0:index]),np.array(Y[(index+1):])],axis=0)
    RF = RandomForestClassifier(n_estimators=n_estimators,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features='sqrt')
    RF.fit(X_train, Y_train)
    # joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
    # print clf.best_score_
    y_p=RF.predict(X_test)
    y_p_p=RF.predict_proba(X_test)
    y_predict.append(y_p)
    y_predict_prob.append(y_p_p)

ROC_AUC_area=metrics.roc_auc_score(Y, y_predict)
predict_save=[Y.astype(int),np.array(y_predict).astype(int)[:,0],np.array(y_predict_prob)[:,0,1]]
print np.array(y_predict_prob)[:,0,1]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict_jackknife.csv',header=None,index=False)
ACC=metrics.accuracy_score(Y,y_predict)
precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
F1_Score=metrics.f1_score(Y, y_predict)
F_measure=F1_Score
MCC=metrics.matthews_corrcoef(Y, y_predict)
pos=TP+FN
neg=FP+TN
savedata=[[[classifier+"n_estimators:"+str(n_estimators)+"max_depth:"+str(max_depth)+"min_samples_split:"+str(min_samples_split)+"min_samples_leaf:"+str(min_samples_leaf),ACC,precision, recall,SN, SP,
            GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
print savedata
print X.shape[1]
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'jackknife.xls')
# y_predict_proba=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X,Y,cv=10,method="predict_proba")
# Y=pd.DataFrame(Y)    
# y_predict_proba=pd.DataFrame(y_predict_proba)
# y_predict_proba=pd.concat([Y,y_predict_proba],axis=1)
# pd.DataFrame(y_predict_proba).to_csv('/home02/chenhuangrong/RFH_ten_cross_validation_label.csv',header=None,index=False)
# y_predict=pd.DataFrame(y_predict)
# y_predict_=pd.concat([Y,y_predict],axis=1)
# pd.DataFrame(y_predict_).to_csv('/home02/chenhuangrong/RFH_ten_cross_validation_predict.csv',header=None,index=False)


# In[ ]:

"""
    cross validation NB
"""
classifier="NB"
datapath =path+outputname+".csv"
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
y_predict=[]
y_predict_prob=[]
for index in xrange(len(train_data)):
    X_test=np.array(X[index]).reshape(1,-1)
    Y_test=np.array(Y[index]).reshape(1,-1)
    X_train=None
    Y_train=None
    if index==0:
        X_train=np.array(X[(index+1):])
        Y_train=np.array(Y[(index+1):])
    elif index==len(Y)-1:
        X_train=np.array(X[0:index])
        Y_train=np.array(Y[0:index])
    else:
        X_train=np.concatenate([np.array(X[0:index]),np.array(X[(index+1):])],axis=0)
        Y_train=np.concatenate([np.array(Y[0:index]),np.array(Y[(index+1):])],axis=0)
    clf=GaussianNB()
    clf.fit(X_train,Y_train)
    y_p=clf.predict(X_test)
    y_p_p=clf.predict_proba(X_test)
    y_predict.append(y_p)
    y_predict_prob.append(y_p_p)
ROC_AUC_area=metrics.roc_auc_score(Y, y_predict)
predict_save=[Y.astype(int),np.array(y_predict).astype(int)[:,0],np.array(y_predict_prob)[:,0,1]]
print np.array(y_predict_prob)[:,0,1]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict_jackknife.csv',header=None,index=False)
ACC=metrics.accuracy_score(Y,y_predict)
precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
F1_Score=metrics.f1_score(Y, y_predict)
F_measure=F1_Score
MCC=metrics.matthews_corrcoef(Y, y_predict)
pos=TP+FN
neg=FP+TN
savedata=[[[classifier,ACC,precision, recall,SN, SP, GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
print savedata
print X.shape[1]
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'jackknife.xls')


# In[ ]:

"""
    cross validation
"""
classifier="LR"
datapath =path+outputname+".csv"
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
y_predict=[]
y_predict_prob=[]
logisReg = LogisticRegression()
C_range = 10.0 ** np.arange(-4,3,1)
grid_parame = dict(C=C_range)
clf = GridSearchCV(logisReg, grid_parame, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X,Y)
C=clf.best_params_['C']
for index in xrange(len(train_data)):
    X_test=np.array(X[index]).reshape(1,-1)
    Y_test=np.array(Y[index]).reshape(1,-1)
    X_train=None
    Y_train=None
    if index==0:
        X_train=np.array(X[(index+1):])
        Y_train=np.array(Y[(index+1):])
    elif index==len(Y)-1:
        X_train=np.array(X[0:index])
        Y_train=np.array(Y[0:index])
    else:
        X_train=np.concatenate([np.array(X[0:index]),np.array(X[(index+1):])],axis=0)
        Y_train=np.concatenate([np.array(Y[0:index]),np.array(Y[(index+1):])],axis=0)
    logisReg = LogisticRegression(C=C)
    logisReg.fit(X_train,Y_train)
    y_p=logisReg.predict(X_test)
    y_p_p=logisReg.predict_proba(X_test)
    y_predict.append(y_p)
    y_predict_prob.append(y_p_p)

ROC_AUC_area=metrics.roc_auc_score(Y, y_predict)
predict_save=[Y.astype(int),np.array(y_predict).astype(int)[:,0],np.array(y_predict_prob)[:,0,1]]
print np.array(y_predict_prob)[:,0,1]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict_jackknife.csv',header=None,index=False)
ACC=metrics.accuracy_score(Y,y_predict)
precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
F1_Score=metrics.f1_score(Y, y_predict)
F_measure=F1_Score
MCC=metrics.matthews_corrcoef(Y, y_predict)
pos=TP+FN
neg=FP+TN
savedata=[[[classifier+"C:"+str(C),ACC,precision, recall,SN, SP, GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
print savedata
print X.shape[1]
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'_jackknife.xls')
# y_predict_proba=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X,Y,cv=10,method="predict_proba")
# Y=pd.DataFrame(Y)    
# y_predict_proba=pd.DataFrame(y_predict_proba)
# y_predict_proba=pd.concat([Y,y_predict_proba],axis=1)
# pd.DataFrame(y_predict_proba).to_csv('/home02/chenhuangrong/RFH_ten_cross_validation_label.csv',header=None,index=False)
# y_predict=pd.DataFrame(y_predict)
# y_predict_=pd.concat([Y,y_predict],axis=1)
# pd.DataFrame(y_predict_).to_csv('/home02/chenhuangrong/RFH_ten_cross_validation_predict.csv',header=None,index=False)


# In[ ]:

"""
    cross validation
"""
classifier="KNN"
datapath =path+outputname+".csv"
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
y_predict=[]
y_predict_prob=[]
knn = KNeighborsClassifier()
#设置k的范围
k_range = list(range(1,10))
leaf_range = list(range(1,2))
weight_options = ['uniform','distance']
algorithm_options = ['auto','ball_tree','kd_tree','brute']
param_gridknn = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options,leaf_size=leaf_range)
clf = GridSearchCV(knn, param_gridknn, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X,Y)
n_neighbors=clf.best_params_['n_neighbors']
weights=clf.best_params_['weights']
algorithm=clf.best_params_['algorithm']
leaf_size=clf.best_params_['leaf_size']
for index in xrange(len(train_data)):
    X_test=np.array(X[index]).reshape(1,-1)
    Y_test=np.array(Y[index]).reshape(1,-1)
    X_train=None
    Y_train=None
    if index==0:
        X_train=np.array(X[(index+1):])
        Y_train=np.array(Y[(index+1):])
    elif index==len(Y)-1:
        X_train=np.array(X[0:index])
        Y_train=np.array(Y[0:index])
    else:
        X_train=np.concatenate([np.array(X[0:index]),np.array(X[(index+1):])],axis=0)
        Y_train=np.concatenate([np.array(Y[0:index]),np.array(Y[(index+1):])],axis=0)
    knn = KNeighborsClassifier(n_neighbors = n_neighbors,weights = weights,algorithm=algorithm,leaf_size=leaf_size)
    knn.fit(X_train,Y_train)
    y_p=knn.predict(X_test)
    y_p_p=knn.predict_proba(X_test)
    y_predict.append(y_p)
    y_predict_prob.append(y_p_p)

# joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
# print clf.best_score_
ROC_AUC_area=metrics.roc_auc_score(Y, y_predict)
predict_save=[Y.astype(int),np.array(y_predict).astype(int)[:,0],np.array(y_predict_prob)[:,0,1]]
print np.array(y_predict_prob)[:,0,1]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict_jackknife.csv',header=None,index=False)
ACC=metrics.accuracy_score(Y,y_predict)
precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
F1_Score=metrics.f1_score(Y, y_predict)
F_measure=F1_Score
MCC=metrics.matthews_corrcoef(Y, y_predict)
pos=TP+FN
neg=FP+TN
savedata=[[[classifier+"n_neighbors:"+str(n_neighbors)+"weights:"+str(weights)+"algorithm:"+str(algorithm)+"leaf_size:"+str(leaf_size),
            ACC,precision, recall,SN, SP, GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
print savedata
print X.shape[1]
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'_jackknife.xls')
# y_predict_proba=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X,Y,cv=10,method="predict_proba")
# Y=pd.DataFrame(Y)    
# y_predict_proba=pd.DataFrame(y_predict_proba)
# y_predict_proba=pd.concat([Y,y_predict_proba],axis=1)
# pd.DataFrame(y_predict_proba).to_csv('/home02/chenhuangrong/RFH_ten_cross_validation_label.csv',header=None,index=False)
# y_predict=pd.DataFrame(y_predict)
# y_predict_=pd.concat([Y,y_predict],axis=1)
# pd.DataFrame(y_predict_).to_csv('/home02/chenhuangrong/RFH_ten_cross_validation_predict.csv',header=None,index=False)


# In[ ]:

"""
    cross validation XGBoost
"""
classifier="XGBoost"
datapath =path+outputname+".csv"
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
y_predict=[]
y_predict_prob=[]
parameters = [{'n_estimators':range(1,10,1),
                          'max_depth':range(2,10),
                          'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
#                           ,'subsample':[0.75,0.8,0.85,0.9]
                      }]
clf = GridSearchCV(XGBClassifier(), parameters, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X_train,Y_train)
n_estimators=clf.best_params_['n_estimators']
max_depth=clf.best_params_['max_depth']
learning_rate=clf.best_params_['learning_rate']
# subsample=clf.best_params_['subsample']
for index in xrange(len(train_data)):
    X_test=np.array(X[index]).reshape(1,-1)
    Y_test=np.array(Y[index]).reshape(1,-1)
    X_train=None
    Y_train=None
    if index==0:
        X_train=np.array(X[(index+1):])
        Y_train=np.array(Y[(index+1):])
    elif index==len(Y)-1:
        X_train=np.array(X[0:index])
        Y_train=np.array(Y[0:index])
    else:
        X_train=np.concatenate([np.array(X[0:index]),np.array(X[(index+1):])],axis=0)
        Y_train=np.concatenate([np.array(Y[0:index]),np.array(Y[(index+1):])],axis=0)
#     xgboost=XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,subsample=subsample)
    xgboost=XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
    xgboost.fit(X_train,Y_train)
    y_p=xgboost.predict(X_test)
    y_p_p=clf.predict_proba(X_test)
    y_predict.append(y_p)
    y_predict_prob.append(y_p_p)

# joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
# print clf.best_score_
ROC_AUC_area=metrics.roc_auc_score(Y, y_predict)
predict_save=[Y.astype(int),np.array(y_predict).astype(int)[:,0],np.array(y_predict_prob)[:,0,1]]
print np.array(y_predict_prob)[:,0,1]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict_jackknife.csv',header=None,index=False)
ACC=metrics.accuracy_score(Y,y_predict)
precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
F1_Score=metrics.f1_score(Y, y_predict)
F_measure=F1_Score
MCC=metrics.matthews_corrcoef(Y, y_predict)
pos=TP+FN
neg=FP+TN
savedata=[[[classifier+"n_estimators:"+str(n_estimators)+"max_depth:"+str(max_depth)+"learning_rate:"+str(learning_rate),ACC,precision, recall,SN, SP,
            GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]

# savedata=[[[classifier+"n_estimators:"+str(n_estimators)+"max_depth:"+str(max_depth)+"learning_rate:"+str(learning_rate)+"subsample:"+str(subsample),ACC,precision, recall,SN, SP,
#             GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]


print savedata
print X.shape[1]
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'jackknife.xls')
# y_predict_proba=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X,Y,cv=10,method="predict_proba")
# Y=pd.DataFrame(Y)    
# y_predict_proba=pd.DataFrame(y_predict_proba)
# y_predict_proba=pd.concat([Y,y_predict_proba],axis=1)
# pd.DataFrame(y_predict_proba).to_csv('/home02/chenhuangrong/RFH_ten_cross_validation_label.csv',header=None,index=False)
# y_predict=pd.DataFrame(y_predict)
# y_predict_=pd.concat([Y,y_predict],axis=1)
# pd.DataFrame(y_predict_).to_csv('/home02/chenhuangrong/RFH_ten_cross_validation_predict.csv',header=None,index=False)

