# !/mnt/anaconda_chr/anaconda2/bin/python
# encoding:utf-8
import pandas as pd
import numpy as np
import itertools
import sys
path=""
inputname=sys.argv[1]
outputname=sys.argv[2]
name=outputname

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


def fetch_singleline_features_withN(sequence):
    alphabet="AUGCN"
    k_num=2
    char_num=5
    two_sequence=[]
    for index,data in enumerate(sequence):
        if index <(len(sequence)-k_num+1):
            two_sequence.append("".join(sequence[index:(index+k_num)]))
    parameter=[e for e in itertools.product([0,1],repeat=char_num)][:int(pow(char_num,k_num))]
    record=[0 for x in range(int(pow(char_num,k_num)))]
    matrix=["".join(e) for e in itertools.product(alphabet, repeat=k_num)] # AA AU AC AG UU UC ...
    final=[]
    for index,data in enumerate(two_sequence):
        if data in matrix:
            final.extend(parameter[matrix.index(data)])
            record[matrix.index(data)]+=1
            final.append(record[matrix.index(data)]*1.0/(index+1))
    return final



def fetch_singleline_features_withoutN(sequence):
    alphabet="AUGC"
    k_num=2
    two_sequence=[]
    for index,data in enumerate(sequence):
        if index <(len(sequence)-k_num+1):
            two_sequence.append("".join(sequence[index:(index+k_num)]))
    parameter=[e for e in itertools.product([0,1],repeat=4)]
    record=[0 for x in range(int(pow(4,k_num)))]
    matrix=["".join(e) for e in itertools.product(alphabet, repeat=k_num)] # AA AU AC AG UU UC ...
    final=[]
    for index,data in enumerate(two_sequence):
        if data in matrix:
            final.extend(parameter[matrix.index(data)])
            record[matrix.index(data)]+=1
            final.append(record[matrix.index(data)]*1.0/(index+1))
    return final

sequences=read_fasta_file(path+inputname)


features_data=[]
if mark_n==True:
    for index,sequence in enumerate(sequences):
        features_data.append(fetch_singleline_features_withN(sequence))
else:
    for index,sequence in enumerate(sequences):
        features_data.append(fetch_singleline_features_withoutN(sequence))
print(np.array(features_data).shape)
print("mark_n",mark_n)
pd.DataFrame(features_data).to_csv(path+outputname,header=None,index=None)