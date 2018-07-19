# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import itertools
from sklearn import svm
import math
from sklearn.model_selection import *
import sklearn.ensemble
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import sys
import warnings
warnings.filterwarnings("ignore")
path=""
inputname=sys.argv[1]
outputname=inputname.split('.')[0]
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
classifier="svm"
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
seq=[]
m6a_2614_sequence=path+inputname
RNA_code='ACGU'
# k=4
interval=int(sys.argv[2])
gap=int(sys.argv[3])
division_num=int(sys.argv[4])
divided_num=division_num*1.0

crossvalidation_value=division_num
cpu_values=int(sys.argv[5])
positive_num_index=int(sys.argv[6])
negative_num_index=int(sys.argv[7])

outputname=outputname+"_PS"+str(interval)+"NP_"+"gap:"+str(gap)

mark_n=False
def read_fasta_file(path):
    '''
    used for load fasta data and transformd into numpy.array format
    '''
    global mark_n
    fh=open(path)
    seq=[]
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            temp_data=["U" if e=="T" else e for e in line.replace('\r','').replace('\n','')]
            if "N" in temp_data:
                mark_n=True
            seq.append(temp_data)
            
    fh.close()
    matrix_data=np.array([list(e) for e in seq])
    return matrix_data
if mark_n==True:
    RNA_code+="N"
seq=read_fasta_file(m6a_2614_sequence)

def make_kmer_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError
positive_labels=np.ones(positive_num_index)
negative_labels=np.zeros(negative_num_index)
labels=np.concatenate([positive_labels,negative_labels],axis=0)



positive_seq=seq[:positive_num_index]
negative_seq=seq[positive_num_index:]

seq=pd.DataFrame(seq)
echo=0
kf = KFold(n_splits=positive_num_index+negative_num_index,shuffle=False)  
if __name__=="__main__":
    for (train_index , test_index) in kf.split(seq):  
        echo+=1
        print("echo: ",echo)
        positive_df=pd.DataFrame(positive_seq)
        positive_negative_x_train=seq.iloc[train_index]
        positive_negative_x_test=seq.iloc[test_index]
        Y_train=pd.DataFrame(labels).iloc[train_index]
        Y_test=pd.DataFrame(labels).iloc[test_index]
        
        
        positive_train_index=[e for e in train_index if labels[e]==1]
        negative_train_index=[e for e in train_index if labels[e]==0]
        
        
        
        
        
        positive_x_train=seq.iloc[positive_train_index]
        negative_df=pd.DataFrame(negative_seq)
        negative_x_train=seq.iloc[negative_train_index]
    # for interval in xrange(1,k+1):
        final_seq_value=[[0 for ii in xrange(seq.shape[1]-interval+1-gap*(interval-1))] for jj in xrange(len(positive_negative_x_train))]
        code_values=make_kmer_list(interval,RNA_code)
        code_len=len(code_values)
        positive_seq_value=[[0 for jj in xrange(seq.shape[1]-interval+1-gap*(interval-1))] for ii in xrange(code_len)]
        negative_seq_value=[[0 for jj in xrange(seq.shape[1]-interval+1-gap*(interval-1))] for ii in xrange(code_len)]
        for i,line_value in enumerate(positive_x_train.values):
            for j,code_value in enumerate(line_value):
                if j<= len(line_value)-interval-gap*(interval-1) :
                    for p,c_value in enumerate(code_values):
                        temp_value=np.array([y for x,y in enumerate(line_value[j:j+interval+gap*(interval-1)]) if x%(gap+1)==0])
                        temp_value="".join(temp_value)
                        if c_value==temp_value:
                            positive_seq_value[p][j]+=1
        positive_seq_value=np.matrix(positive_seq_value)*1.0/positive_num_index
        for i,line_value in enumerate(negative_x_train.values):
            for j,code_value in enumerate(line_value):
                if j<= len(line_value)-interval-gap*(interval-1):
                    for p,c_value in enumerate(code_values):
                        temp_value=np.array([y for x,y in enumerate(line_value[j:j+interval+gap*(interval-1)]) if x%(gap+1)==0])
                        temp_value="".join(temp_value)
                        if c_value==temp_value:
                            negative_seq_value[p][j]+=1
        negative_seq_value=np.matrix(negative_seq_value)*1.0/negative_num_index
        for i,line_value in enumerate(positive_negative_x_train.values):
            for j,code_value in enumerate(line_value):
                if j<= len(line_value)-interval-gap*(interval-1) :
                    for p,c_value in enumerate(code_values):
                        temp_value=np.array([y for x,y in enumerate(line_value[j:j+interval+gap*(interval-1)]) if x%(gap+1)==0])
                        temp_value="".join(temp_value)
                        if c_value==temp_value:
                              final_seq_value[i][j]=positive_seq_value[p,j]-negative_seq_value[p,j]
        y_final_seq_value=[[0 for ii in xrange(seq.shape[1]-interval+1-gap*(interval-1))] for jj in xrange(len(positive_negative_x_test))]
        for i,line_value in enumerate(positive_negative_x_test.values):
            for j,code_value in enumerate(line_value):
                if j<= len(line_value)-interval-gap*(interval-1) :
                    for p,c_value in enumerate(code_values):
                        temp_value=np.array([y for x,y in enumerate(line_value[j:j+interval+gap*(interval-1)]) if x%(gap+1)==0])
                        temp_value="".join(temp_value)
                        if c_value==temp_value:
                              y_final_seq_value[i][j]=positive_seq_value[p,j]-negative_seq_value[p,j]
                              
                              
                              
  
            X_train1 = np.array(final_seq_value)
            X_test1 = np.array(y_final_seq_value)
            svc = svm.SVC(probability=True)
            parameters = {'kernel': ['rbf'], 'C':map(lambda x:2**x,np.linspace(-2,5,7)), 'gamma':map(lambda x:2**x,np.linspace(-5,2,7))}
            clf = GridSearchCV(svc, parameters, cv=crossvalidation_value, n_jobs=cpu_values, scoring='accuracy')
            Y_train1=Y_train.values
            
            clf.fit(X_train1, Y_train1)
            C=clf.best_params_['C']
            y_pred_prob=clf.predict_proba(np.array(X_test1).reshape(1,-1))

            gamma=clf.best_params_['gamma']
            y_pred=clf.predict(np.array(X_test1).reshape(1,-1))

            y_pred_prob_all.extend(y_pred_prob)
            y_pred_all.extend(y_pred)
            Y_all.append(Y_test)

    all_y=[np.array(Y_all).astype(int).T[0],np.array(y_pred_all).astype(int),np.array(y_pred_prob_all).astype(list)[:,1]]
    pd.DataFrame(np.matrix(all_y).T).to_csv(outputname+"_"+classifier+"_jackknife_predict.csv",header=None,index=False)
    fpr, tpr, thresholds = roc_curve(np.array(Y_all).T[0], list(np.array(y_pred_prob_all).astype(list)[:,1]))

    roc_auc = metrics.roc_auc_score(np.array(Y_all).T[0],np.array(y_pred_prob_all).astype(list)[:,1])
    ACC=metrics.accuracy_score(np.array(Y_all).T[0],list(np.array(y_pred_all).astype(int)))
    print "ACC:",ACC
    precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(np.array(Y_all).T[0], list(np.array(y_pred_all).astype(int))) 
    F1_Score=metrics.f1_score(np.array(Y_all).T[0], list(np.array(y_pred_all).astype(int)))
    F_measure=F1_Score
    MCC=metrics.matthews_corrcoef(np.array(Y_all).T[0], list(np.array(y_pred_all).astype(int)))
    pos=TP+FN
    neg=FP+TN


    savedata=[str(X_train1.shape[1]),u"正："+str(pos_all)+u'负：'+str(neg_all),'svm'+"C:"+str(C)+"gamma:"+str(gamma),ACC,precision, recall,SN,
                SP, GM,F_measure,F1_Score,MCC,roc_auc,TP,
                FN,FP,TN]
    row0 = [u'特征集', u'样本个数', u'分类器', u'Accuracy', u'Precision', u'Recall', u'SN', u'SP',
                    u'Gm', u'F_measure', u'F_score', u'MCC', u'ROC曲线面积', u'tp', u'fn', u'fp', u'tn']
    final_out_to_excel=[]
    final_out_to_excel.append(row0)
    final_out_to_excel.append(savedata)
    print savedata
    pd.DataFrame(final_out_to_excel).to_excel('jackknife_'+classifier+"_"+outputname+'.xlsx',sheet_name="jackknife",header=None,index=None)