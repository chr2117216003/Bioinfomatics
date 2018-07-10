
# coding: utf-8

# In[ ]:


#PseEIIP He, W., et al., 70ProPred: a predictor for discovering sigma70 promoters based on combining multiple features. BMC systems biology, 2018. 12(4): p. 44.
import itertools
import sys
import numpy as np
import pandas as pd
gene_value="U"

gene_type=sys.argv[3]

if gene_type=="RNA":
    gene_value="U"
elif gene_type=="DNA":
    gene_value="T"
fill_NA=sys.argv[4]
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

def AthMethPre_extract_one_line(data_line):
    '''
    extract features from one line, such as one m6A sample
    '''
#    A=[0,0,0,1]
#    U=[0,0,1,0]
#    C=[0,1,0,0]
#    G=[1,0,0,0]
#    N=[0,0,0,0]
#    feature_representation={"A":A,gene_value:U,"C":C,"G":G,"N":N}
#    beginning=0
    end=len(data_line)-1
    one_line_feature=[]
    alphabet='ACG'
    alphabet+=gene_value
    if fill_NA=="1":
        alphabet+="N"
    matrix_three=["".join(e) for e in itertools.product(alphabet, repeat=3)]# AAA AAU AAC ...
#    print(matrix_three)
    feature_three=np.zeros(len(matrix_three))
    A=0.1260
    U=0.1335
    C=0.1340
    G=0.0806
    N=0.0000
    temp=[A,C,G,U]
    if fill_NA=="1":
        temp.append(N)
#    print(temp)
    AUCG=[sum(e) for e in itertools.product(temp, repeat=3)]# AAA AAU AAC ...
    for index,data in enumerate(data_line):
        if "".join(data_line[index:(index+3)]) in matrix_three and index <= end-2:
            feature_three[matrix_three.index("".join(data_line[index:(index+3)]))]+=1     
    sum_three=np.sum(feature_three)
    feature_three=feature_three/sum_three
    feature_three=feature_three*AUCG
    one_line_feature.extend(feature_three)
    return one_line_feature

def AthMethPre_feature_extraction(matrix_data):
    final_feature_matrix=[AthMethPre_extract_one_line(e) for e in matrix_data]
    return final_feature_matrix


m6a_benchmark_dataset=sys.argv[1]
matrix_data=read_fasta_file(m6a_benchmark_dataset)
final_feature_matrix=AthMethPre_feature_extraction(matrix_data)
print(np.array(final_feature_matrix).shape)
pd.DataFrame(final_feature_matrix).to_csv(sys.argv[2],header=None,index=False)



