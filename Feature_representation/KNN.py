
# coding: utf-8

# In[ ]:


#KNN  Jia, C.-Z., J.-J. Zhang, and W.-Z. Gu, RNA-MethylPred: a high-accuracy predictor to identify N6-methyladenosine in RNA. Analytical biochemistry, 2016. 510: p. 72-75.
import numpy as np
import pandas as pd
import sys
def read_fasta_file(path):
    '''
    used for load fasta data and transformd into numpy.array format
    '''
    fh=open(m6a_benchmark_dataset)
    seq=[]
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            seq.append(line.replace('\r','').replace('\n',''))
    fh.close()
    matrix_data=np.array([list(e) for e in seq])
    return matrix_data


def Comparing_score(query_sequence,original_sequence):
    
    score=0
    for index,data in enumerate(query_sequence):
        if data==original_sequence[index]:
            score=score+2
        else:
            score=score-1
    return score


def generating_one_column(matrix_data):
    whole_comparison_score=[]
    the_begin_of_index=len(matrix_data)/2
    for index_1,data_1 in enumerate(matrix_data):
        one_line_comparison_score=np.zeros(len(matrix_data))
        mark_origin_label=np.zeros(len(matrix_data))
        for index_2,data_2 in enumerate(matrix_data):
            if index_1!=index_2:
                one_line_comparison_score[index_1]=-100
                if index_2<the_begin_of_index:
                    mark_origin_label[index_2]=1
                one_line_comparison_score[index_2]=Comparing_score(data_1,data_2)
        temp=[]
        temp=[one_line_comparison_score,mark_origin_label]
        whole_comparison_score.append(temp)
    return whole_comparison_score


def generating_features(matrix_data,K_list):
    matrix=generating_one_column(matrix_data)
    print np.asarray(matrix).shape
    whole_=[]
    for index,K_data in enumerate(K_list):
        line=[]
        for data in matrix:
            idx=np.argsort(data[0])[::-1]
            idx=idx[xrange(K_data)]
            data[1]=pd.DataFrame(data[1])
            datas=data[1].iloc[idx]
            datas=datas.values
            line.append(sum(datas)/K_data)
        whole_.append(line)
    whole_=np.array(whole_).T
    return whole_
            
            


m6a_benchmark_dataset=sys.argv[1]
matrix_data=read_fasta_file(m6a_benchmark_dataset)
final_feature_matrix=generating_features(matrix_data,xrange(10,201))
print np.array(final_feature_matrix).shape
pd.DataFrame(final_feature_matrix[0]).to_csv(sys.argv[2],header=None,index=False)



