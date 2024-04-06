"""
Parses and does some basic stats on data from IBM csv files
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle as pick
import matplotlib.pyplot as plt
import seaborn as sns


if __name__=="__main__":
    import argparse
    import os
    import pandas as pd
    
    parser=argparse.ArgumentParser(description="Tool to parse stats in IBM csv file")                                                                      
    parser.add_argument('--quantum_machine',dest='quantum_machine',action="store",default="ibmq_manhattan",help="path to the data frame we want to analyze")                            
    args=parser.parse_args()

    if os.path.exists(os.path.join(args.quantum_machine,args.quantum_machine+".csv")):
        machine_df= pd.read_csv(os.path.join(args.quantum_machine,args.quantum_machine+".csv"))
    else:
        assert 0


    cnot_error_rates = machine_df["CNOT error rate"]

    visited_list=[]
    _sum=0
    n=0
    value_list=[]
    for qbit in cnot_error_rates:
        cnot_list = qbit.split(",")
        for cnot in cnot_list:
            if len(cnot.split(":"))<2: continue #saw a case where data was missing from IBM
            cnot_id = cnot.split(":")[0]
            cnot_error_rate=float(cnot.split(":")[1])
            if cnot_id not in visited_list:
                visited_list.append(cnot_id)
                value_list.append(cnot_error_rate)
                n+=1
                _sum+=cnot_error_rate
    average_cnot_rate=_sum/n
    print("Machine {} has average CNOT error rate of {} std {}".format(args.quantum_machine,average_cnot_rate,np.std(value_list)))
