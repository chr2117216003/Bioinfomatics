
# coding: utf-8

# In[ ]:

# !/use/bin/env python

import pandas as pd
import numpy as np
import csv
import re
import os
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


# In[ ]:

"""
    jackknife + mrmr + xgboost
"""
if __name__=="__main__":
	print "start"
	classifier="xgboost_mrmr"
	mode="jackknife"
	datapath =path+outputname+".csv"
	outputpath=path+outputname+"_output.csv"
	train_data = pd.read_csv(datapath, header=None, index_col=None)
	X = np.array(train_data)

	Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
	Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
	Y.extend(Y2)
	Y = np.array(Y)
	row=Y.shape[0]
	Y1=Y.reshape(row,1)
	#concatenate class and data
	full_csv_with_class=np.concatenate([Y1,X],axis=1)
	print full_csv_with_class

	#print the results of original csv data and final full data
	print "the shape of data:"+str(X.shape)
	print "the shape of data and class:"+str(full_csv_with_class.shape)

	#generating virtual headers
	columns=["class"]
	columns_numbers=np.arange(full_csv_with_class.shape[1]-1)
	columns.extend(columns_numbers)

	# Write data into files
	csvFile2 = open(outputpath,'w') 
	writer = csv.writer(csvFile2)
	m = len(full_csv_with_class)
	writer.writerow(columns)
	for i in range(m):
		writer.writerow(full_csv_with_class[i])
	csvFile2.close()

	print "start running mrmr procedure"
	print str(full_csv_with_class.shape[1]-1)

	os.system("./mRMR/mrmr -i "+outputpath+" -n "+str(full_csv_with_class.shape[1]-1)+" >mRMR/"+classifier+mode+outputname+"_output.mrmrout")
	print "mrmr complete "


	location_mark=0
	final_set=[]
	final_set_values=[]
	fn=open("mRMR/"+classifier+mode+outputname+"_output.mrmrout",'r')
	for line in fn.readlines():
		if line.strip() =="":
			location_mark=0
		if location_mark==1 and line.split()[1]!="Fea":
			 final_set.append(int(line.split()[1]))
		if location_mark==1 and line.split()[3]!="Score":
			final_set_values.append(float(line.split()[3]))
		if re.findall(r"mRMR",line) and re.findall(r"feature",line):
			location_mark=1
	print "the set is:",final_set
	with open(classifier+mode+"all_dimension_results.txt",'w') as f:
			f.write(str(final_set)+"\n")
	print final_set_values
	with open(classifier+mode+"all_dimension_results.txt",'a') as f:
			f.write(str(final_set_values)+"\n")
	bestACC=0
	bestn_estimators=0
	bestmax_depth=0
	best_dimension=0
	bestlearning_rate=0
	all_dimension_results=[]
	select_list=[]
	best_savedata=""
	for select_num,value in enumerate(final_set):
		select_list.append(int(value)-1)
		X_test=None
		Y_test=None
		y_predict=[]
		y_predict_prob=[]
		train_data2=train_data
		X_train=pd.DataFrame(train_data2)
		X_train=X_train.iloc[:,select_list]
		X = np.array(X_train)
		parameters = [{'n_estimators':range(1,500,20),
						  'max_depth':range(3,7,1),
						  'learning_rate':[0.2,0.3,0.4]
	#                       ,'subsample':[0.75,0.8,0.85,0.9]
						  }]
		clf = GridSearchCV(XGBClassifier(), parameters, cv=crossvalidation_values, n_jobs=CPU_values, scoring='accuracy')
		clf.fit(X, Y)
		n_estimators=clf.best_params_['n_estimators']
		max_depth=clf.best_params_['max_depth']
		learning_rate=clf.best_params_['learning_rate']
		print n_estimators
		print max_depth
		print learning_rate
		joblib.dump(clf,path+classifier+mode+outputname+str(select_num)+".model")
		kf = KFold(n_splits=len(X))
		for train_index,test_index in kf.split(X):
			X_train,X_test,Y_train,Y_test=X[train_index],X[test_index],Y[train_index],Y[test_index]
			xgboost=XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
			xgboost.fit(X_train,Y_train)
			y_p=xgboost.predict(X_test)
			y_p_p=xgboost.predict_proba(X_test)
			y_predict.append(y_p)
			y_predict_prob.append(y_p_p)
		ROC_AUC_area=metrics.roc_auc_score(Y, y_predict)
		predict_save=[Y.astype(int),np.array(y_predict).astype(int)[:,0],np.array(y_predict_prob)[:,0,1]]
		predict_save=np.array(predict_save).T
		pd.DataFrame(predict_save).to_csv(path+classifier+mode+outputname+"_"+'_predict_jackknife.csv',header=None,index=False)
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
			best_savedata=savedata
			bestlearning_rate=learning_rate
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
	easy_excel.save("xgboost_jackknife",[str(best_dimension)],best_savedata,path+classifier+mode+'jackknife_'+outputname+'.xls')

