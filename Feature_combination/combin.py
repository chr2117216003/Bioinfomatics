import getopt
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
whole_result=[]
input_files=""
whole_dimension=[]
opts, args = getopt.getopt(sys.argv[1:], "hi:", )
for op, value in opts:
    if op == "-i":
        input_files = str(value)
        input_files = input_files.replace(" ", "").split(',')
        for input_file in input_files:
            if input_file == "":
                print "Warning: please insure no blank in your input files !"
                sys.exit()

if __name__=="__main__":
    for i in xrange(len(input_files)):
        for j in xrange(i+1,len(input_files)):
            first_name=input_files[i].strip("").split('.')[0]
            second_name=input_files[j].strip("").split('.')[0]
            output_name=first_name+'_'+second_name+'.csv'
            first_file = pd.read_csv(input_files[i], header=None, index_col=None)  # sys.argv[1])
            end = len(first_file.values[0])
            first_file = first_file.values[:, 0:]
            first_file = pd.DataFrame(first_file).astype(float)
            second_file = pd.read_csv(input_files[j], header=None, index_col=None)
            end= len(second_file.values[0])
            second_file=second_file.values[:,0:]
            second_file = pd.DataFrame(second_file).astype(float)
            print "first_file_num:", len(first_file)
            print "first_file_length:", len(first_file.values[0])
            print "second_file_num:", len(second_file)
            print "second_file_length:", len(second_file.values[0])
            output_file = pd.concat([first_file, second_file], axis=1)
            print "output_file_num:", len(output_file)
            print "output_file_len:", len(output_file.values[0])
            scaler = MinMaxScaler()
            output_file = scaler.fit_transform(np.array(output_file))
            print "normalization"
            pd.DataFrame(output_file).to_csv(output_name, header=None, index=False)