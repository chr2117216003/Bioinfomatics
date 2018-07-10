#PCP Liu, Z., et al., pRNAm-PC: Predicting N(6)-methyladenosine sites in RNA sequences via physical-chemical properties. Anal Biochem, 2016. 497: p. 60-7.
import pandas as pd
import numpy as np
import sys
import itertools
import os
path=""
inputname=sys.argv[1]
outputname=sys.argv[2]
gene_type=sys.argv[3]
fill_NA=sys.argv[4]
propertyname="physical_chemical_properties_RNA_without.txt"
if fill_NA=="1" and gene_type=="RNA":
    propertyname="physical_chemical_properties_RNA_without.txt"
elif fill_NA=="0" and gene_type=="RNA":
    propertyname="physical_chemical_properties_RNA.txt"
elif fill_NA=="1" and gene_type=="DNA":
    propertyname="physical_chemical_properties_DNA_without.txt"
elif fill_NA=="0" and gene_type=="DNA":
    propertyname="physical_chemical_properties_DNA.txt"



seq=[]
physical_chemical_properties_path=path+propertyname
m6a_2614_sequence=path+inputname

fh=open(m6a_2614_sequence)
for line in fh:#get the fasta sequence
    if line.startswith('>'):
        pass
    else:
        seq.append(line.replace('\n','').replace('\r',''))
fh.close()

data=pd.read_csv(physical_chemical_properties_path,header=None,index_col=None)#read the phisical chemichy proporties
prop_key=data.values[:,0]

if fill_NA=="1":
    prop_key[21]='NA'
print(prop_key)
prop_data=data.values[:,1:]
prop_data=np.matrix(prop_data)
DNC_value=np.array(prop_data).T
DNC_value_scale=[[]]*len(DNC_value)
for i in xrange(len(DNC_value)):
    average_=sum(DNC_value[i]*1.0/len(DNC_value[i]))
    std_=np.std(DNC_value[i],ddof=1)
    DNC_value_scale[i]=[round((e-average_)/std_,2) for e in DNC_value[i]]
prop_data_transformed=zip(*DNC_value_scale)
# prop_data_transformed=StandardScaler().fit_transform(prop_data)
prop_len=len(prop_data_transformed[0])

whole_m6a_seq=seq
i=0
phisical_chemichy_len=len(prop_data_transformed)#the length of properties
sequence_line_len=len(seq[0])#the length of one sequence
LAMDA=4
finally_result=[]#used to save the fanal result
for one_m6a_sequence_line in whole_m6a_seq:
    one_sequence_value=[[]]*(sequence_line_len-1)
    PC_m=[0.0]*prop_len
    PC_m=np.array(PC_m)
    for one_sequence_index in range(sequence_line_len-1):
        for prop_index in xrange(len(prop_key)):
            if one_m6a_sequence_line[one_sequence_index:one_sequence_index+2]==prop_key[prop_index]:
                one_sequence_value[one_sequence_index]=prop_data_transformed[prop_index]
        PC_m+=np.array(one_sequence_value[one_sequence_index])
    PC_m=PC_m/(sequence_line_len-1)
    auto_value=[]
    for LAMDA_index in xrange(1,LAMDA+1):
        temp = [0.0] * prop_len
        temp=np.array(temp)
        for auto_index in xrange(1,sequence_line_len-LAMDA_index):
            temp=temp+(np.array(one_sequence_value[auto_index-1])-PC_m)*(np.array(one_sequence_value[auto_index+LAMDA_index-1])-PC_m)
            temp=[round(e,8) for e in temp.astype(float)]
        x=[round(e/(sequence_line_len-LAMDA_index-1),8) for e in temp]
        auto_value.extend([round(e,8) for e in x])
    for LAMDA_index in xrange(1, LAMDA + 1):
        for i in xrange(1,prop_len+1):
            for j in xrange(1,prop_len+1):
                temp2=0.0
                if i != j:
                    for auto_index in xrange(1, sequence_line_len - LAMDA_index):
                            temp2+=(one_sequence_value[auto_index-1][i-1]-PC_m[i-1])*(one_sequence_value[auto_index+LAMDA_index-1][j-1]-PC_m[j-1])
                    auto_value.append(round(temp2/((sequence_line_len-1)-LAMDA_index),8))

    finally_result.append(auto_value)
# finally_result=MinMaxScaler().fit_transform(np.matrix(finally_result))
print(np.array(finally_result).shape)
pd.DataFrame(finally_result).to_csv(path+outputname,header=None,index=False)

