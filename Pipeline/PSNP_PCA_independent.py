#!/usr/bin/env python
# encoding:utf-8
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import KFold  
from sklearn import svm
from sklearn.model_selection import train_test_split
import math
from sklearn.model_selection import *
import sklearn.ensemble
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import sys
from sklearn.model_selection import GridSearchCV
import subprocess



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
        recall==0
    else:
        recall=TP/(TP+FN)
    GM=math.sqrt(recall*SP)
    #MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return precision,recall,SN,SP,GM,TP,TN,FP,FN
final_out_to_excel=[]
row0 = [u'特征集', u'样本个数', u'分类器', u'Accuracy', u'Precision', u'Recall', u'SN', u'SP',
                u'Gm', u'F_measure', u'F_score', u'MCC', u'ROC曲线面积', u'tp', u'fn', u'fp', u'tn']
final_out_to_excel.append(row0)
seq=[]
m6a_2614_sequence="m6a_data.txt"
RNA_code='ACGU'
k=1
interval=1
divided_num=10.0
division_num=10
fh=open(m6a_2614_sequence)
for line in fh:
    if line.startswith('>'):
        continue
    else:
        seq.append(line.replace('\n',''))
fh.close()

def make_kmer_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError
positive_seq=seq[:len(seq)/2]
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data, train_target, test_size=0.1, random_state=0)
negative_seq=seq[len(seq)/2:]
kf = KFold(n_splits=division_num,shuffle=False)  
for select_num in xrange(1,):
    y_pred_prob_all=[]
    y_pred_all=[]
    Y_all=[]
    ACC_all=0
    precision_all=0
    recall_all=0
    SN_all=0
    SP_all=0
    GM_all=0
    TP_all=0
    TN_all=0
    FP_all=0
    FN_all=0
    F_measure_all=0
    F1_Score_all=0
    pos_all=0
    neg_all=0
    MCC_all=0
    for train_index , test_index in kf.split(positive_seq):  
        positive_df=pd.DataFrame(positive_seq)
        positive_x_train=positive_df.iloc[train_index,:]
        positive_y_train=positive_df.iloc[test_index,:]
        negative_df=pd.DataFrame(negative_seq)
        negative_x_train=negative_df.iloc[train_index,:]
        negative_y_train=negative_df.iloc[test_index,:]
        positive_negative_x_train=pd.concat([positive_x_train,negative_x_train],axis=0)
        positive_negative_y_train=pd.concat([positive_y_train,negative_y_train],axis=0)
    # for interval in xrange(1,k+1):
        final_seq_value=[[0 for ii in xrange(len(seq[0])-interval)] for jj in xrange(len(positive_negative_x_train))]
        code_values=make_kmer_list(interval,RNA_code)
        code_len=len(code_values)
        positive_seq_value=[[0 for jj in xrange(len(seq[0])-interval)] for ii in xrange(code_len)]
        negative_seq_value=[[0 for jj in xrange(len(seq[0])-interval)] for ii in xrange(code_len)]
        for i,line_value in enumerate(positive_x_train.values):
            for j,code_value in enumerate(line_value[0]):
                if j<= len(line_value[0])-interval-1 :
                    for p,c_value in enumerate(code_values):
                        if c_value==line_value[0][j:j+interval]:
                            positive_seq_value[p][j]+=1
        positive_seq_value=np.matrix(positive_seq_value)*1.0/(len(seq)/2)
        for i,line_value in enumerate(negative_x_train.values):
            for j,code_value in enumerate(line_value[0]):
                if j<= len(line_value[0])-interval-1 :
                    for p,c_value in enumerate(code_values):
                        if c_value==line_value[0][j:j+interval]:
                            negative_seq_value[p][j]+=1
        negative_seq_value=np.matrix(negative_seq_value)*1.0/(len(seq)/2)
        for i,line_value in enumerate(positive_negative_x_train.values):
            for j,code_value in enumerate(line_value[0]):
                if j<= len(line_value[0])-interval-1 :
                    for p,c_value in enumerate(code_values):
                        if c_value==line_value[0][j:j+interval]:
                              final_seq_value[i][j]=positive_seq_value[p,j]-negative_seq_value[p,j]
        y_final_seq_value=[[0 for ii in xrange(len(seq[0])-interval)] for jj in xrange(len(positive_negative_y_train))]
        

        for i,line_value in enumerate(positive_negative_y_train.values):
            for j,code_value in enumerate(line_value[0]):
                if j<= len(line_value[0])-interval-1 :
                    for p,c_value in enumerate(code_values):
                        if c_value==line_value[0][j:j+interval]:
                              y_final_seq_value[i][j]=positive_seq_value[p,j]-negative_seq_value[p,j]
        


        final_seq_value=pd.DataFrame(final_seq_value)
        y_final_seq_value=pd.DataFrame(y_final_seq_value)
        pca=PCA(n_components=select_num) 
        train=pca.fit_transform(final_seq_value) 
        test=pca.transform(y_final_seq_value)
        print "PCA success"

		
        X_train=np.array(train)
        Y_train = list(map(lambda x: 1, xrange(len(X_train) / 2)))
        Y2_train = list(map(lambda x: 0, xrange(len(X_train) / 2)))
        Y_train.extend(Y2_train)
        Y_train = np.array(Y_train)


		
        X_test=np.array(test)
        Y_test  = list(map(lambda x: 1, xrange(len(y_final_seq_value) / 2)))
        Y2_test  = list(map(lambda x: 0, xrange(len(y_final_seq_value) / 2)))
        Y_test.extend(Y2_test )
        Y_test  = np.array(Y_test)
        print "already get train and test data"
        svc = svm.SVC(probability=True)
        parameters = {'kernel': ['rbf'], 'C':map(lambda x:2**x,np.linspace(-2,5,28)), 'gamma':map(lambda x:2**x,np.linspace(-5,2,28))}
        print "begin ten cross validation"
        clf = GridSearchCV(svc, parameters, cv=10, n_jobs=8, scoring='accuracy')
        clf.fit(X_train, Y_train)
        C=clf.best_params_['C']
        y_pred_prob=clf.predict_proba(X_test)

        gamma=clf.best_params_['gamma']
        print 'c:',C,'gamma:',gamma
        y_pred=clf.predict(X_test)

        y_pred_prob_all.extend(y_pred_prob)
        y_pred_all.extend(y_pred)
        Y_all.extend(Y_test)
        ACC=metrics.accuracy_score(Y_test,y_pred)
        print ACC
        precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y_test, y_pred) 
        F1_Score=metrics.f1_score(Y_test, y_pred)
        F_measure=F1_Score
        MCC=metrics.matthews_corrcoef(Y_test, y_pred)

        pos=TP+FN
        neg=FP+TN
        ACC_all=ACC_all+ACC
        print "ACC_all:",ACC_all
        precision_all=precision_all+precision
        recall_all=recall_all+recall
        SN_all=SN_all+SN
        SP_all=SP_all+SP
        GM_all=GM_all+GM
        TP_all=TP_all+TP
        TN_all=TN_all+TN
        FP_all=FP_all+FP
        FN_all=FN_all+FN
        F_measure_all=F_measure_all+F_measure
        F1_Score_all=F1_Score_all+F1_Score
        pos_all=pos_all+pos
        neg_all=neg_all+neg
        MCC_all=MCC_all+MCC
    all_y=[np.array(Y_all).astype(int),np.array(y_pred_all).astype(int),np.array(y_pred_prob_all).astype(list)[:,1]]
    pd.DataFrame(np.matrix(all_y).T).to_csv('independent_test_with_mrmr%d.csv'%select_num,header=None,index=False)
    fpr, tpr, thresholds = roc_curve(np.array(Y_all).T, list(np.array(y_pred_prob_all).astype(list)[:,1]))
    roc_auc = auc(fpr, tpr)
   
    savedata=[str(X_train.shape[1]),"正"+str(pos_all)+'负'+str(neg_all),'svm'+"C:"+str(C)+"gamma:"+str(gamma),ACC_all/divided_num,precision_all/divided_num, recall_all/divided_num,SN_all/divided_num,
                SP_all/divided_num, GM_all/divided_num,F_measure_all/divided_num,F1_Score_all/divided_num,MCC_all/divided_num,roc_auc,TP_all,
                FN_all,FP_all,TN_all]
    final_out_to_excel.append(savedata)
print savedata
pd.DataFrame(final_out_to_excel).to_excel("PSNP_mrmr.xlsx",sheet_name="independent_test",index=False,header=False)