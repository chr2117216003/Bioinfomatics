import pandas as pd
import numpy as np
import os
import sys
import itertools

path=""
inputname=sys.argv[1]
outputname=sys.argv[2]
gene_type=sys.argv[3]
fill_NA=sys.argv[4]
propertyname="physical_chemical_properties_3_RNA_without.txt"
if fill_NA=="1" and gene_type=="RNA":
    propertyname="physical_chemical_properties_3_RNA_without.txt"
elif fill_NA=="0" and gene_type=="RNA":
    propertyname="physical_chemical_properties_3_RNA.txt"
elif fill_NA=="1" and gene_type=="DNA":
    propertyname="physical_chemical_properties_3_DNA_without.txt"
elif fill_NA=="0" and gene_type=="DNA":
    propertyname="physical_chemical_properties_3_DNA.txt"
print(propertyname)
phisical_chemical_proporties=pd.read_csv(path+propertyname,header=None,index_col=None)
m6a_sequence=open(path+inputname)
DNC_key=phisical_chemical_proporties.values[:,0]

if fill_NA=="1":
    DNC_key[21]='NA'
# DNC_key=np.array(['AA','AC','AG','AU','CA','CC','CG','CU','GA','GC','GG','GU','UA','UC','UG','UU'])
DNC_value=phisical_chemical_proporties.values[:,1:]
DNC_value=np.array(DNC_value).T
DNC_value_scale=[[]]*len(DNC_value)
for i in xrange(len(DNC_value)):
    average_=sum(DNC_value[i]*1.0/len(DNC_value[i]))
    std_=np.std(DNC_value[i],ddof=1)
    DNC_value_scale[i]=[round((e-average_)/std_,2) for e in DNC_value[i]]
DNC_value_scale=zip(*DNC_value_scale)


DNC_len=len(DNC_value_scale[0])
m6aseq=[]
for line in m6a_sequence:
    if line.startswith('>'):
        pass
    else:
        m6aseq.append(line.replace('\n','').replace("\r",''))
w=0.9
Lamda=6
result_value=[]
m6a_len=len(m6aseq[0])
m6a_num=len(m6aseq)
for m6a_line_index in xrange(m6a_num):
    frequency=[0]*len(DNC_key)
    m6a_DNC_value=[[]]*(m6a_len-1)
    for m6a_line_doublechar_index in xrange(m6a_len):
        for DNC_index in xrange(len(DNC_key)):
            if m6aseq[m6a_line_index][m6a_line_doublechar_index:m6a_line_doublechar_index+2]==DNC_key[DNC_index]:
                m6a_DNC_value[m6a_line_doublechar_index]=DNC_value_scale[DNC_index]
                frequency[DNC_index]+=1
    frequency=[e/float(sum(frequency)) for e in frequency]
    p=sum((frequency))
    #frequency=np.array(frequency)/float(sum(frequency))#(m6a_len-1)
    one_line_value_with = 0.0
    sita = [0] * Lamda
    for lambda_index in xrange(1, Lamda + 1):
        one_line_value_without_ = 0.0
        for m6a_sequence_value_index in xrange(1, m6a_len - lambda_index):
            temp = map(lambda (x, y): round((x - y) ** 2,8), zip(np.array(m6a_DNC_value[m6a_sequence_value_index - 1]), np.array(m6a_DNC_value[m6a_sequence_value_index - 1 + lambda_index])))
            temp_value = round(sum(temp) * 1.0 / DNC_len,8)
            one_line_value_without_ += temp_value
        one_line_value_without_ = round(one_line_value_without_ / (m6a_len - lambda_index-1),8)
        sita[lambda_index - 1] = one_line_value_without_
        one_line_value_with += one_line_value_without_
    dim = [0] * (len(DNC_key) + Lamda)
    for index in xrange(1, len(DNC_key) + Lamda+1):
        if index <= len(DNC_key):
            dim[index - 1] = frequency[index - 1] / (1.0 + w * one_line_value_with)
        else:
            dim[index - 1] = w * sita[index - len(DNC_key)-1] / (1.0 + w * one_line_value_with)
        dim[index-1]=round(dim[index-1],8)
    result_value.append(dim)
print(np.array(result_value).shape)
pd.DataFrame(result_value).to_csv(path+outputname, header=None, index=None)

m6a_sequence.close()