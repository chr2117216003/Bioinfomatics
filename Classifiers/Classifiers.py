
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
path=""
inputname=sys.argv[1]
outputname=inputname
name=outputname
CV_num=int(sys.argv[2])
n_jobs_value=int(sys.argv[3])


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


# In[ ]:

"""
    cross validation
"""
classifier="SVM"
datapath =path+outputname
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
svc = svm.SVC()
parameters = {'kernel': ['rbf'], 'C':map(lambda x:2**x,np.linspace(-2,5,7)), 'gamma':map(lambda x:2**x,np.linspace(-5,2,7))}
clf = GridSearchCV(svc, parameters, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X, Y)
C=clf.best_params_['C']
# joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
# print clf.best_score_
gamma=clf.best_params_['gamma']
y_predict=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma),X,Y,cv=CV_num,n_jobs=n_jobs_value)
y_predict_prob=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X,Y,cv=CV_num,n_jobs=n_jobs_value,method='predict_proba')
ROC_AUC_area=metrics.roc_auc_score(Y, y_predict_prob)
predict_save=[Y.astype(int),y_predict.astype(int),y_predict_prob[:,1]]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict.csv',header=None,index=False)
ACC=metrics.accuracy_score(Y,y_predict)
precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
F1_Score=metrics.f1_score(Y, y_predict)
F_measure=F1_Score
MCC=metrics.matthews_corrcoef(Y, y_predict)
pos=TP+FN
neg=FP+TN
savedata=[[[classifier+"C:"+str(C)+"gamma:"+str(gamma),ACC,precision, recall,SN, SP, GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
print savedata
print X.shape[1]
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'.xls')
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
    cross validation GBDT
"""
classifier="GBDT"
datapath =path+outputname
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
GBDT = GradientBoostingClassifier(learning_rate=0.1,min_samples_split=300,min_samples_leaf=20,max_depth=8,max_features='sqrt')
parameters = {'n_estimators':range(10,120,1),'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200),'min_samples_leaf':range(60,101,10)}
clf = GridSearchCV(GBDT, parameters, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X, Y)
n_estimators=clf.best_params_['n_estimators']
max_depth=clf.best_params_['max_depth']
min_samples_split=clf.best_params_['min_samples_split']
min_samples_leaf=clf.best_params_['min_samples_leaf']
# joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
# print clf.best_score_
y_predict=cross_val_predict(GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=0.1,min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features='sqrt'),X,Y,cv=CV_num,n_jobs=n_jobs_value)

y_predict_prob=cross_val_predict(GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=0.1,min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features='sqrt'),X,Y,cv=CV_num,n_jobs=n_jobs_value,method='predict_proba')
ROC_AUC_area=metrics.roc_auc_score(Y, y_predict_prob)
predict_save=[Y.astype(int),y_predict.astype(int),y_predict_prob[:,1]]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict.csv',header=None,index=False)
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
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'.xls')
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
    cross validation RF
"""
classifier="RF"
datapath =path+outputname
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
RF = RandomForestClassifier(min_samples_split=300,min_samples_leaf=20,max_depth=8,max_features='sqrt')
parameters = {'n_estimators':range(10,120,1),'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200),'min_samples_leaf':range(60,101,10)}
clf = GridSearchCV(RF, parameters, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X, Y)
n_estimators=clf.best_params_['n_estimators']
max_depth=clf.best_params_['max_depth']
min_samples_split=clf.best_params_['min_samples_split']
min_samples_leaf=clf.best_params_['min_samples_leaf']
# joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
# print clf.best_score_
y_predict=cross_val_predict(RandomForestClassifier(n_estimators=n_estimators,min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features='sqrt'),X,Y,cv=CV_num,n_jobs=n_jobs_value)

y_predict_prob=cross_val_predict(RandomForestClassifier(n_estimators=n_estimators,min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features='sqrt'),X,Y,cv=CV_num,n_jobs=n_jobs_value,method='predict_proba')
ROC_AUC_area=metrics.roc_auc_score(Y, y_predict_prob)
predict_save=[Y.astype(int),y_predict.astype(int),y_predict_prob[:,1]]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict.csv',header=None,index=False)
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
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'.xls')
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
from sklearn.naive_bayes import MultinomialNB
classifier="NB"
datapath =path+outputname
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
# joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
# print clf.best_score_
y_predict=cross_val_predict(MultinomialNB(),X,Y,cv=CV_num,n_jobs=n_jobs_value)

y_predict_prob=cross_val_predict(MultinomialNB(),X,Y,cv=CV_num,n_jobs=n_jobs_value,method='predict_proba')

ROC_AUC_area=metrics.roc_auc_score(Y, y_predict_prob)

predict_save=[Y.astype(int),y_predict.astype(int),y_predict_prob[:,1]]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict.csv',header=None,index=False)
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
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'.xls')
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
classifier="LR"
datapath =path+outputname
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
logisReg = LogisticRegression()
C_range = 10.0 ** np.arange(-4,3,1)
grid_parame = dict(C=C_range)
clf = GridSearchCV(logisReg, grid_parame, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X, Y)
C=clf.best_params_['C']
# joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
# print clf.best_score_

y_predict=cross_val_predict(LogisticRegression(C=C),X,Y,cv=CV_num,n_jobs=n_jobs_value)

y_predict_prob=cross_val_predict(LogisticRegression(C=C),X,Y,cv=CV_num,n_jobs=n_jobs_value,method='predict_proba')
ROC_AUC_area=metrics.roc_auc_score(Y, y_predict_prob)
predict_save=[Y.astype(int),y_predict.astype(int),y_predict_prob[:,1]]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict.csv',header=None,index=False)
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
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'.xls')
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
datapath =path+outputname
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
knn = KNeighborsClassifier()
#设置k的范围
k_range = list(range(1,10))
leaf_range = list(range(1,2))
weight_options = ['uniform','distance']
algorithm_options = ['auto','ball_tree','kd_tree','brute']
param_gridknn = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options,leaf_size=leaf_range)
clf = GridSearchCV(knn, param_gridknn, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X, Y)
n_neighbors=clf.best_params_['n_neighbors']
weights=clf.best_params_['weights']
algorithm=clf.best_params_['algorithm']
leaf_size=clf.best_params_['leaf_size']
# joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
# print clf.best_score_

y_predict=cross_val_predict(KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size),X,Y,cv=CV_num,n_jobs=n_jobs_value)

y_predict_prob=cross_val_predict(KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size),X,Y,cv=CV_num,n_jobs=n_jobs_value,method='predict_proba')

ROC_AUC_area=metrics.roc_auc_score(Y, y_predict_prob)

predict_save=[Y.astype(int),y_predict.astype(int),y_predict_prob[:,1]]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict.csv',header=None,index=False)
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
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'.xls')
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
datapath =path+outputname
train_data = pd.read_csv(datapath, header=None, index_col=None)
X = np.array(train_data)
Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
Y.extend(Y2)
Y = np.array(Y)
parameters = [{'n_estimators':range(1,100,5),
                      'max_depth':range(1,100,5),
                      'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
                      ,'subsample':[0.75,0.8,0.85,0.9]
                      }]
clf = GridSearchCV(XGBClassifier(), parameters, cv=CV_num, n_jobs=n_jobs_value, scoring='accuracy')
clf.fit(X, Y)
n_estimators=clf.best_params_['n_estimators']
max_depth=clf.best_params_['max_depth']
learning_rate=clf.best_params_['learning_rate']
subsample=clf.best_params_['subsample']
# joblib.dump(clf,'/home02/chenhuangrong/'+name+'.model')
# print clf.best_score_
# y_predict=cross_val_predict(XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate,
#                                                        max_depth=max_depth),X,Y,cv=CV_num,n_jobs=12)
y_predict=cross_val_predict(XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate,
                                                       subsample=subsample,max_depth=max_depth),X,Y,cv=CV_num,n_jobs=n_jobs_value)

# y_predict_prob=cross_val_predict(XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate,
#                                                        max_depth=max_depth),X,Y,cv=CV_num,n_jobs=12,method='predict_proba')

y_predict_prob=cross_val_predict(XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate,
                                                       subsample=subsample,max_depth=max_depth),X,Y,cv=CV_num,n_jobs=n_jobs_value,method='predict_proba')
ROC_AUC_area=metrics.roc_auc_score(Y, y_predict_prob)
predict_save=[Y.astype(int),y_predict.astype(int),y_predict_prob[:,1]]
predict_save=np.array(predict_save).T
pd.DataFrame(predict_save).to_csv(path+outputname+"_"+classifier+'_predict.csv',header=None,index=False)
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
easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'.xls')
# y_predict_proba=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X,Y,cv=10,method="predict_proba")
# Y=pd.DataFrame(Y)    
# y_predict_proba=pd.DataFrame(y_predict_proba)
# y_predict_proba=pd.concat([Y,y_predict_proba],axis=1)
# pd.DataFrame(y_predict_proba).to_csv('/home02/chenhuangrong/RFH_ten_cross_validation_label.csv',header=None,index=False)
# y_predict=pd.DataFrame(y_predict)
# y_predict_=pd.concat([Y,y_predict],axis=1)
# pd.DataFrame(y_predict_).to_csv('/home02/chenhuangrong/RFH_ten_cross_validation_predict.csv',header=None,index=False)


# In[ ]:




# In[ ]:



