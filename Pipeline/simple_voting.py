# encoding:utf-8
import getopt
from sklearn.preprocessing import MinMaxScaler
import os,time
from multiprocessing import Process, Manager
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import KFold  
from sklearn import svm
# from sklearn.cross_validation import train_test_split
import math
from sklearn.model_selection import *
import sklearn.ensemble
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import sys
from sklearn.model_selection import GridSearchCV
import warnings 
whole_result=[]
input_files=""
whole_dimension=[]
default_l = 1
cross_validation_value = 10
CPU_value = 1
opts, args = getopt.getopt(sys.argv[1:], "hi:l:c:n:", )
final_out_to_excel=[]
row0 = [u'特征集', u'样本个数', u'分类器', u'Accuracy', u'Precision', u'Recall', u'SN', u'SP',
                u'Gm', u'F_measure', u'F_score', u'MCC', u'ROC曲线面积', u'tp', u'fn', u'fp', u'tn']
final_out_to_excel.append(row0) #above was used to generate xlsx format Excel file
for op, value in opts:
    if op == "-i":
        input_files = str(value)
        input_files = input_files.replace(" ", "").split(',')
        for input_file in input_files:
            if input_file == "":
                print("Warning: please insure no blank in your input files !")
                sys.exit()
    elif op == "-l":
        if int(value) == 1:
            default_l = 1
        else:
            default_l = -1
    elif op == "-c":
        cross_validation_value = int(value)
    
    elif op == "-n":
        CPU_value = int(value)

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

def worker(X_train, y_train, cross_validation_value, CPU_value, input_file, share_y_predict_dict, share_y_predict_proba_dict):
    print("子进程执行中>>> pid={0},ppid={1}".format(os.getpid(),os.getppid()))
    svc = svm.SVC(probability=True)
    parameters = {'kernel': ['rbf'], 'C':map(lambda x:2**x,np.linspace(-2,5,7)), 'gamma':map(lambda x:2**x,np.linspace(-5,2,7))}
    clf = GridSearchCV(svc, parameters, cv=cross_validation_value, n_jobs=CPU_value, scoring='accuracy')
    clf.fit(X_train, y_train)
    C=clf.best_params_['C']
    gamma=clf.best_params_['gamma']
    print('c:',C,'gamma:',gamma)

    
    y_predict=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,),X_train,y_train,cv=cross_validation_value,n_jobs=CPU_value)
    y_predict_prob=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X_train,y_train,cv=cross_validation_value,n_jobs=CPU_value,method='predict_proba')
    input_file = input_file.replace(".csv","")
    y_predict_path = input_file + "_predict.csv"
    y_predict_proba_path = input_file + "_predict_proba.csv"
    share_y_predict_dict[input_file] = y_predict
    share_y_predict_proba_dict[input_file] = y_predict_prob[:,1]
    pd.DataFrame(y_predict).to_csv(y_predict_path, header = None, index = False)
    pd.DataFrame(y_predict_prob[:,1]).to_csv(y_predict_proba_path, header = None, index = False)
    print("子进程终止>>> pid={0}".format(os.getpid()))
        
if __name__=="__main__":
    print("主进程执行中>>> pid={0}".format(os.getpid()))
    manager = Manager()
    share_y_predict_dict = manager.dict()
    share_y_predict_proba_dict = manager.dict()
    ps=[]
    if default_l == 1:
        data = ""
        x_len = 1000
        y_len = 1000
        file_len = len(input_files)
        threshold = file_len/2
        for index, input_file in enumerate(input_files):
            data = pd.read_csv(input_file,header=None)
            (x_len,y_len) = data.shape

            X_train = data.iloc[:,0:y_len-1]
            y_train = data.iloc[:,[y_len-1]]
            X_train = X_train.values
            y_train = y_train.values
            y_train = y_train.reshape(-1)
            p=Process(target=worker,name="worker"+str(index),args=(X_train, y_train, cross_validation_value, CPU_value,input_file,share_y_predict_dict,share_y_predict_proba_dict))
            ps.append(p)
        # 开启进程
        for index, input_file in enumerate(input_files):
            ps[index].start()

        # 阻塞进程
        for index, input_file in enumerate(input_files):
            ps[index].join()
        ensembling_prediction = 0
        ensembling_prediction_proba = 0
        for key, value in share_y_predict_dict.items():
            ensembling_prediction = ensembling_prediction + value
        ensembling_prediction = [1 if e > threshold else 0 for e in ensembling_prediction]
        print(ensembling_prediction)
        for key, value in share_y_predict_proba_dict.items():
            ensembling_prediction_proba = ensembling_prediction_proba + value
        ensembling_prediction_proba = ensembling_prediction_proba/3.0
        print(ensembling_prediction_proba/3.0)
        ACC=metrics.accuracy_score(y_train,ensembling_prediction)
        print("ACC",ACC)
        precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(y_train, ensembling_prediction) 
        F1_Score=metrics.f1_score(y_train, ensembling_prediction)
        F_measure=F1_Score
        MCC=metrics.matthews_corrcoef(y_train, ensembling_prediction)
        auc = metrics.roc_auc_score(y_train, ensembling_prediction_proba)
        pos=TP+FN
        neg=FP+TN
        savedata=[str(input_files),"正："+str(len(y_train[y_train == 1]))+'负：'+str(len(y_train[y_train == 1])),'svm',ACC,precision, recall,SN,SP, GM,F_measure,F1_Score,MCC,auc,TP,FN,FP,TN]
        final_out_to_excel.append(savedata)
        print("final_out_to_excel",final_out_to_excel)
        pd.DataFrame(ensembling_prediction).to_csv("voting_prediction_label.csv", header = None, index = False)
        pd.DataFrame(ensembling_prediction_proba).to_csv("voting_prediction_proba_label.csv", header = None, index = False)
        pd.DataFrame(final_out_to_excel).to_excel('output'+'.xlsx',sheet_name="results",index=False,header=False)
        print("主进程终止")
    else:
        data = ""
        x_len = 1000
        y_len = 1000
        file_len = len(input_files)
        threshold = file_len/2
        for index, input_file in enumerate(input_files):
            data = pd.read_csv(input_file,header=None)
            (x_len,y_len) = data.shape
            X_train = data.values
            half_sequence_number = x_len / 2
            y_train = np.array([1 if e < half_sequence_number else 0 for (e,value) in enumerate(X_train)])
            y_train = y_train.reshape(-1)
            print("default y_train: ", y_train)
            p=Process(target=worker,name="worker"+str(index),args=(X_train, y_train, cross_validation_value, CPU_value,input_file,share_y_predict_dict,share_y_predict_proba_dict))
            ps.append(p)
        # 开启进程
        for index, input_file in enumerate(input_files):
            ps[index].start()

        # 阻塞进程
        for index, input_file in enumerate(input_files):
            ps[index].join()
        ensembling_prediction = 0
        ensembling_prediction_proba = 0
        for key, value in share_y_predict_dict.items():
            ensembling_prediction = ensembling_prediction + value
        ensembling_prediction = [1 if e > threshold else 0 for e in ensembling_prediction]
        print(ensembling_prediction)
        for key, value in share_y_predict_proba_dict.items():
            ensembling_prediction_proba = ensembling_prediction_proba + value
        ensembling_prediction_proba = ensembling_prediction_proba/3.0
        print(ensembling_prediction_proba/3.0)
        ACC=metrics.accuracy_score(y_train,ensembling_prediction)
        print("ACC",ACC)
        precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(y_train, ensembling_prediction) 
        F1_Score=metrics.f1_score(y_train, ensembling_prediction)
        F_measure=F1_Score
        MCC=metrics.matthews_corrcoef(y_train, ensembling_prediction)
        auc = metrics.roc_auc_score(y_train, ensembling_prediction_proba)
        pos=TP+FN
        neg=FP+TN
        savedata=[str(input_files),"正："+str(len(y_train[y_train == 1]))+'负：'+str(len(y_train[y_train == 1])),'svm',ACC,precision, recall,SN,SP, GM,F_measure,F1_Score,MCC,auc,TP,FN,FP,TN]
        final_out_to_excel.append(savedata)
        print("final_out_to_excel",final_out_to_excel)
        pd.DataFrame(ensembling_prediction).to_csv("voting_prediction_label.csv", header = None, index = False)
        pd.DataFrame(ensembling_prediction_proba).to_csv("voting_prediction_proba_label.csv", header = None, index = False)
        pd.DataFrame(final_out_to_excel).to_excel('output'+'.xlsx',sheet_name="results",index=False,header=False)
        print("主进程终止")