"""
Gets data for a backend from the cloud.
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle as pick
import matplotlib.pyplot as plt
import seaborn as sns

# importing Qiskit
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit.compiler import transpile
from qiskit.compiler import assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute
from qiskit.transpiler.layout import Layout
from qiskit.transpiler import PassManager
from qiskit.transpiler import CouplingMap,InstructionDurations
from qiskit.converters import dag_to_circuit,circuit_to_dag

from pass_sets import get_passes, pass_names
from qiskit.visualization import plot_histogram





import pandas as pd
import time 
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ncsu')

if __name__=="__main__":
    import argparse
    import os
    
    parser=argparse.ArgumentParser(description="Tool to launch  a set of benchmarks and evaluate them across different pass sets")                                                                      
    parser.add_argument('--quantum_machine',dest='quantum_machine',action="store",default="ibmq_manhattan",help="path to the data frame we want to analyze")                            
    parser.add_argument('--benchmarks',action="store",default="noise_adaptive_benchmarks",help="path to the benchmarks to be evaluated")
    parser.add_argument('--max_qubits',action="store",type=int,default=10,help="max number of qubit a benchmark can have")
    #parser.add_argument('--fetch_backend',action="store_true",default=False,help="Fetch from the backend new data, rather than reusing stored dataframes")
    args=parser.parse_args()
    
    if not os.path.exists(args.quantum_machine):
        os.mkdir(args.quantum_machine) 

    if not os.path.exists(args.benchmarks):
        raise ValueError("Benchmark directory does not exist")

    exec("from "+args.benchmarks+".expected_output import expected_outputs") #import the benchmark specific expect results 
    
    backend=provider.get_backend(args.quantum_machine)
    backend_coupling_map=CouplingMap(provider.get_backend(args.quantum_machine).configuration().coupling_map)
    backend_basis=provider.get_backend(args.quantum_machine).configuration().basis_gates
    backend_physical_qubits=provider.get_backend(args.quantum_machine).configuration().n_qubits


    #collect a set of valid benchmarks
    benchmark_list=[]
    for file_name in os.listdir(args.benchmarks): #os.listdir is determinstic, so we don't need to worry about benchmark order
        if not os.path.isfile(os.path.join(args.benchmarks,file_name)): continue
        if not ".qasm" in file_name: continue
        if "~" in file_name or "#" in file_name: continue
        temp_circuit=QuantumCircuit()
        temp_circuit=temp_circuit.from_qasm_file(os.path.join(args.benchmarks,file_name))
        if temp_circuit.num_qubits>backend_physical_qubits or temp_circuit.num_qubits>args.max_qubits: want_results=False #leave out benchmarks with more logical than physical qubits
        else: want_results=True #filter results based on benchmark qubit set size 
        benchmark_base_name=file_name.split('.qasm')[0]
        benchmark_list.append((benchmark_base_name,os.path.join(args.benchmarks,file_name),want_results))
 
    pass_names = pass_names()
    results_dictionary={"benchmark":[],"pass":[],"PST":[],"IST":[],"Rank":[]} 


    if os.path.exists(os.path.join(args.quantum_machine,"raw_machine_results.csv")):
        #Have results dictionary now, convert to data frame and plot
        results_df=pd.read_csv(os.path.join(args.quantum_machine,"raw_machine_results.csv"))
    else:
        for p in pass_names:
            jobs_for_pass=list(filter(lambda x: x.name()==p, backend.jobs(limit=300,descending=True)))
            if(len(jobs_for_pass)==0): continue
            latest_job=jobs_for_pass[0]
            job_result=latest_job.result()
            #need to process each different benchmark
            adjust=0 #counter to adjust bench index when skipping benchmarks
            for bench_index, (bench_name, bench_path, want_results) in enumerate(benchmark_list):
                expected_outcome=expected_outputs[bench_name]
                if not want_results:
                    adjust+=1
                    continue
               # assert bench_index<len(job_result.get_counts()) #TODO: Handle older runs that did not use large benchmarks 
                if len(benchmark_list)>len(job_result.get_counts()):
                    counts=job_result.get_counts(bench_index-adjust)
                else:
                    counts=job_result.get_counts(bench_index)
                total_shots=sum(counts.values())
                if expected_outcome in counts:
                    PST=counts[expected_outcome]/total_shots
                    next_highest=[_ for _ in counts.items() if _[0]!=expected_outcome]
                    next_highest=sorted(next_highest,key=lambda x: x[1], reverse=True)
                    next_highest=next_highest[0][1]/float(total_shots)
                    IST=PST/next_highest
                    sorted_results=sorted(counts.items(), key=lambda x: x[1],reverse=True)
                    rank=[_ for _,res in enumerate(sorted_results) if res[0]==expected_outcome]
                    rank=1/(rank[0]+1)
                else:
                    PST=0
                    IST=0
                    rank=1/len(counts.keys())

                #collect PST,IST,Rank data for benchmark,pass combination
                results_dictionary["benchmark"].append(bench_name)
                results_dictionary["pass"].append(p)
                results_dictionary["PST"].append(PST)
                results_dictionary["IST"].append(IST)
            results_dictionary["Rank"].append(rank)
   
        #Have results dictionary now, convert to data frame and plot
        results_df=pd.DataFrame(results_dictionary)
        results_df.to_csv(os.path.join(args.quantum_machine,"raw_machine_results.csv"),index=False)
    
    #plot out the gat_df
    results_df['PST norm']=results_df['PST'].astype(np.float32)
    results_df['IST norm']=results_df['IST'].astype(np.float32)
    for bench_name,bench_path,want_results in benchmark_list:
        if want_results==False: continue
        PST_norm=results_df.loc[((results_df['benchmark']==bench_name) & (results_df['pass']=='noise+sabre')),'PST']
        IST_norm=results_df.loc[((results_df['benchmark']==bench_name) & (results_df['pass']=='noise+sabre')),'IST']
        for index,_ in enumerate(results_df['benchmark']):
            if _==bench_name:
                results_df.loc[index,'PST norm']=results_df['PST norm'][index]/float(PST_norm)
                results_df.loc[index,'IST norm']=results_df['IST norm'][index]/float(IST_norm)
    
    fig_array=[]
    #Construct Size and Codeword Distribution (% Total)
    ax=results_df.pivot("benchmark","pass","PST norm").plot.bar(figsize=(10,4),rot=0,title="Normalized PST "+args.quantum_machine)
    ax.set_ylabel("Normalized PST")
    ax.get_legend().remove()
    ax.get_figure().legend(fontsize = 6.4,loc='center',bbox_to_anchor=(0.93,0.6),ncol=1)
    plt.axhline(y=1, color='r', linestyle='--')
    fig_array.append((ax.get_figure(),os.path.join(args.quantum_machine,"normalized_pst.pdf")))

    #Construct Size and Codeword Distribution (% Total)
    ax=results_df.pivot("benchmark","pass","IST norm").plot.bar(figsize=(10,4),rot=0,title="Normalized IST"+args.quantum_machine)
    ax.set_ylabel("Normalized IST")
    ax.get_legend().remove()
    ax.get_figure().legend(fontsize = 6.4,loc='center',bbox_to_anchor=(0.93,0.6),ncol=1)
    plt.axhline(y=1, color='r', linestyle='--')
    fig_array.append((ax.get_figure(),os.path.join(args.quantum_machine,"normalized_ist.pdf")))


    #Construct Size and Codeword Distribution (% Total)
    ax=results_df.pivot("benchmark","pass","Rank").plot.bar(figsize=(10,4),rot=0,title="Rank of Correct Answer"+args.quantum_machine)
    ax.set_ylabel("Rank of Correct Answer")
    ax.get_legend().remove()
    ax.get_figure().legend(fontsize = 6.4,loc='center',bbox_to_anchor=(0.93,0.6),ncol=1)
    plt.axhline(y=1, color='r', linestyle='--')
    fig_array.append((ax.get_figure(),os.path.join(args.quantum_machine,"rank_of_correct_answer.pdf")))      
    results_df.to_csv(os.path.join(args.quantum_machine,"pass_data_norm.csv"),index=False)
    for fig,path in fig_array: fig.savefig(path,format="pdf")


    rank_dict={}
    average_rank={}

    for pass_name in pass_names:
        rank_dict[pass_name]={}
        for (bench_name,bench_path,want_results) in benchmark_list:
            rank_dict[pass_name][bench_name]=0
            average_rank[bench_name]=[]
        rank_dict[pass_name]["total"]=0

    #calculate ranks of all the passes for this machine
    for index, row in results_df.iterrows():
        bench=row["benchmark"]
        comp_pass=row["pass"]
        norm_pst=row["PST norm"]
        if norm_pst>=1.0:
            rank_dict[comp_pass][bench]+=1
            rank_dict[comp_pass]["total"]+=1
        average_rank[bench].append((comp_pass,norm_pst))

    
    rank_df_dict={"benchmark":[], "pass":[],"count":[]}
    for pass_name in rank_dict:
        for bench in rank_dict[pass_name]:
            rank_df_dict["benchmark"].append(bench)
            rank_df_dict["pass"].append(pass_name)
            rank_df_dict["count"].append(rank_dict[pass_name][bench])
    rank_df=pd.DataFrame(rank_df_dict)
    rank_df.to_csv(os.path.join(args.quantum_machine,"rank_data.csv"),index=False)

    #get average ranks across all benchmarks for this given machine
    average_rank_data={"pass":[],"avg rank":[]}
    for pass_name in pass_names:
        weighted_sum=0
        n=0
        for bench in average_rank:
            sorted_pst=sorted(average_rank[bench],key=lambda x:x[1],reverse=True)
            for pst_index, pst in enumerate(sorted_pst):
                if pst[0]==pass_name:
                    weighted_sum+=(len(sorted_pst)-pst_index) #larger is better rank
                    n+=1
        average_rank_data["pass"].append(pass_name)
        average_rank_data["avg rank"].append(weighted_sum/n)
    avg_df=pd.DataFrame(average_rank_data)
    avg_df.to_csv(os.path.join(args.quantum_machine,"avg_rank_data.csv"),index=False)

    #get average rank across benchmarks
    bench_rank_data={"pass":[],"avg rank":[],"bench type":[]}
    bench_types=["qft","bv","adder"]
    for pass_name in pass_names:
        for b_type in bench_types:
            weighted_sum=0
            n_type=0
            for bench in average_rank:
                if b_type in bench:
                    sorted_pst=sorted(average_rank[bench],key=lambda x:x[1],reverse=True)
                    for pst_index, pst in enumerate(sorted_pst):
                        if pst[0]==pass_name:
                            weighted_sum+=(len(sorted_pst)-pst_index) #larger is better rank
                            n_type+=1
            bench_rank_data["pass"].append(pass_name)
            bench_rank_data["avg rank"].append(weighted_sum/n_type)
            bench_rank_data["bench type"].append(b_type)
    bench_rank_df=pd.DataFrame(bench_rank_data)
    bench_rank_df.pivot("pass","bench type","avg rank").to_csv(os.path.join(args.quantum_machine,"avg_bench_rank.csv"),index=True)
