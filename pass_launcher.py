"""
Sets up the passes that will be compared, and runs the benchmarks for each pass
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


import pandas as pd
import time 




def analyze_single_qubit_depth(dag_circuit): #figure out how many single qubit gates occur in a row
    node_list={}
    visited=[]
    single_op_count=0
    total_ops=0
    for node in dag_circuit.front_layer():
        for x in dag_circuit.bfs_successors(node):
            current_node=x[0]
            if current_node.type=="op" and current_node.name!="measure" and current_node not in visited:
                total_ops+=1
                visited.append(current_node)
                if len(current_node.qargs)==1:
                    single_op_count+=1
                    #1 qubit operation
                    found=False
                    for single_ops in node_list:
                        if current_node in node_list[single_ops][1]:
                            entry=(1+node_list[single_ops][0],x[1])
                            found=True
                            break
                    if found==False:
                        node_list[current_node]=(1,x[1])
                    else:
                        node_list[current_node]=entry
    vals=[_[0] for _ in node_list.values()]
    print("Largest single qubit run {} % single op {} raw single op {}".format(max(vals),100*single_op_count/total_ops,single_op_count))
                    






# import basic plot tools
from qiskit.visualization import plot_histogram
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ncsu')

import logging
#logging.basicConfig(level='DEBUG')

if __name__=="__main__":
    import argparse
    import os
    
    parser=argparse.ArgumentParser(description="Tool to launch  a set of benchmarks and evaluate them across different pass sets")                                                                      
    parser.add_argument('--quantum_machine',dest='quantum_machine',action="store",default="ibmq_toronto",help="path to the data frame we want to analyze")                            
    parser.add_argument('--benchmarks',action="store",default="noise_adaptive_benchmarks",help="path to the benchmarks to be evaluated")
    parser.add_argument('--shots',action="store",type=int,default=8192,help="Number of shots to run each circuit")
    parser.add_argument('--overwrite',action="store_true", default=False,help="overwrite existing pass data")
    parser.add_argument('--no_sim',action="store_false",default=True,help="run simulator too")
    parser.add_argument('--max_qubits',action="store",type=int,default=10,help="max number of qubit a benchmark can have")
    parser.add_argument('--use_ibm',action="store_true",default=False,help="ibm base transpiler")
    parser.add_argument('--plot', action="store_true",default=False,help="just plot exisisting df")
    args=parser.parse_args()


    if not os.path.exists(args.quantum_machine):
        os.mkdir(args.quantum_machine) 


    if not os.path.exists(args.benchmarks):
        raise ValueError("Benchmark directory does not exist")


    if not os.path.exists(os.path.join(args.quantum_machine,"pass_gate_data.csv")) or args.overwrite:
        gate_df=pd.DataFrame()
    else:
        gate_df=pd.read_csv(os.path.join(args.quantum_machine,"pass_gate_data.csv"))

  
        
    #get backend information
    backend=provider.get_backend(args.quantum_machine)
    backend_coupling_map=CouplingMap(provider.get_backend(args.quantum_machine).configuration().coupling_map)
    backend_basis=provider.get_backend(args.quantum_machine).configuration().basis_gates
    backend_physical_qubits=provider.get_backend(args.quantum_machine).configuration().n_qubits
    backend_properties=provider.get_backend(args.quantum_machine).properties()
    backend_durations=InstructionDurations()
    backend_durations=backend_durations.from_backend(provider.get_backend(args.quantum_machine))
    from pass_sets import get_passes
    PASS_SETS=get_passes(backend_basis,backend_coupling_map,backend_properties,backend_durations)

    if args.quantum_machine!="ibmq_qasm_simulator":
        pass
    else:
        PASS_SETS={x: None for x in PASS_SETS}
        
    
    #collect a set of valid benchmarks
    benchmark_list=[]
    for file_name in os.listdir(args.benchmarks):
        if not os.path.isfile(os.path.join(args.benchmarks,file_name)): continue
        if not ".qasm" in file_name: continue
        if "~" in file_name or "#" in file_name: continue
        temp_circuit=QuantumCircuit()
        temp_circuit=temp_circuit.from_qasm_file(os.path.join(args.benchmarks,file_name))
        if temp_circuit.num_qubits>backend_physical_qubits or temp_circuit.num_qubits>args.max_qubits: continue #leave out benchmarks with more logical than physical qubits
        benchmark_base_name=file_name.split('.qasm')[0]
        benchmark_list.append((benchmark_base_name,os.path.join(args.benchmarks,file_name)))
    
   
    if args.plot:
        #plot out the gat_df
        gate_df['gates norm']=gate_df['gates'].astype(np.float32)
        gate_df['comp time norm']=gate_df['compile time'].astype(np.float32)
        gate_df['depth norm']=gate_df['circuit depth'].astype(np.float32)

        #filter the data frame based on the benchmarks selected by the max number of qubits
        new_gate_df=pd.DataFrame()
        for bench_name,_ in benchmark_list:
            if new_gate_df.empty:
                new_gate_df=gate_df[gate_df.benchmark == bench_name]
            else:
                new_gate_df=new_gate_df.append(gate_df[gate_df.benchmark==bench_name],ignore_index=True)
        gate_df=new_gate_df
        for bench_name,bench_path in benchmark_list:
            gate_norm=gate_df.loc[((gate_df['benchmark']==bench_name) & (gate_df['pass']=='noise+sabre')),'gates']
            depth_norm=gate_df.loc[((gate_df['benchmark']==bench_name) & (gate_df['pass']=='noise+sabre')),'circuit depth']
            comp_norm=gate_df.loc[((gate_df['benchmark']==bench_name) & (gate_df['pass']=='noise+sabre')),'compile time']
            for index,_ in enumerate(gate_df['benchmark']):
                if _==bench_name:
                    gate_df.loc[index,"gates norm"]=gate_df['gates norm'][index]/float(gate_norm)
                    gate_df.loc[index,'comp time norm']=gate_df['comp time norm'][index]/float(comp_norm)
                    gate_df.loc[index,"depth norm"]=gate_df['depth norm'][index]/float(depth_norm)
        fig_array=[]
        #Construct Size and Codeword Distribution (% Total)
        ax=gate_df.pivot("benchmark","pass","gates norm").plot.bar(figsize=(10,4),rot=0,title="Normalized Gates "+args.quantum_machine)
        ax.set_ylabel("Normalized Gates")
        ax.get_legend().remove()
        ax.set_ylim(0.0,3)

        ax.get_figure().legend(fontsize = 6.4,loc='center',bbox_to_anchor=(0.93,0.6),ncol=1)
        plt.axhline(y=1, color='r', linestyle='--')
        fig_array.append((ax.get_figure(),os.path.join(args.quantum_machine,"normalized_gate_counts.pdf")))

        #Construct Size and Codeword Distribution (% Total)
        ax=gate_df.pivot("benchmark","pass","depth norm").plot.bar(figsize=(10,4),rot=0,title="Normalized Depth "+args.quantum_machine)
        ax.set_ylabel("Normalized Depth")
        ax.get_legend().remove()
        ax.set_ylim(0.0,3)

        ax.get_figure().legend(fontsize = 6.4,loc='center',bbox_to_anchor=(0.93,0.6),ncol=1)
        plt.axhline(y=1, color='r', linestyle='--')
        fig_array.append((ax.get_figure(),os.path.join(args.quantum_machine,"normalized_depth.pdf")))

        
        #Construct Size and Codeword Distribution (% Total)
        ax=gate_df.pivot("benchmark","pass","comp time norm").plot.bar(figsize=(10,4),rot=0,title="Normalized Compile Time "+args.quantum_machine)
        ax.set_ylabel("Normalized Compile Time")
        ax.get_legend().remove()

        ax.get_figure().legend(fontsize = 6.4,loc='center',bbox_to_anchor=(0.93,0.6),ncol=1)
        plt.axhline(y=1, color='r', linestyle='--')
        fig_array.append((ax.get_figure(),os.path.join(args.quantum_machine,"normalized_compile_time.pdf")))      
        gate_df.to_csv(os.path.join(args.quantum_machine,"pass_gate_data_norm.csv"),index=False)
        for fig,path in fig_array: fig.savefig(path,format="pdf")
        exit() 

    #run a list of circuits
    new_data={"gates":[],"compile time":[],"circuit depth":[],"pass":[], "benchmark":[]}

    for pass_name,pass_set in PASS_SETS.items():
        circuit_list=[]
        execute_list=True
        print(pass_name)
        for bench_name,bench_path in benchmark_list:
            skip=False
            #check if we already have the data for this pair
            if "pass" in gate_df and "benchmark" in gate_df:
                for _p,_b in zip(gate_df["pass"],gate_df["benchmark"]):
                    if (_p,_b)==(pass_name,bench_name):
                        skip=True
                        break
            if skip:
                execute_list=False 
            print(bench_name)
            temp_pm=PassManager(pass_set)
            circuit_name=":".join([bench_name,pass_name])
            temp_circuit=QuantumCircuit(name=circuit_name)
            temp_circuit=temp_circuit.from_qasm_file(bench_path)
            time1=time.time()
            if args.quantum_machine!="ibmq_qasm_simulator":
                opt_circuit=temp_pm.run(temp_circuit)
            elif args.use_ibm:
                opt_circuit=transpile(temp_circuit,seed=11)
            else:
                opt_circuit=temp_circuit
            opt_time=time.time()-time1
            circuit_list.append(opt_circuit)
            dag_circuit=circuit_to_dag(opt_circuit)
            new_data["pass"].append(pass_name)
            new_data["benchmark"].append(bench_name)
            operation_counts=dag_circuit.count_ops()
            gate_count=0
            for ops in operation_counts:
                if ops !="barrier" and ops!="measure":
                    gate_count+=operation_counts[ops]
            circuit_depth=dag_circuit.depth()
            new_data["gates"].append(gate_count)
            new_data["circuit depth"].append(circuit_depth)
            new_data["compile time"].append(opt_time)
            if args.no_sim:
                simulator=Aer.get_backend('qasm_simulator')
                result=execute(opt_circuit,simulator).result()
                counts=result.get_counts()
                print(counts)
            print(operation_counts)
            analyze_single_qubit_depth(dag_circuit)
        #execute the circuits for each benchmark as a job
        if len(circuit_list)>0 and execute_list==True:
            if args.use_ibm: pass_name="ibm"
            temp_qobj=assemble(circuit_list,backend=backend,shots=args.shots)
            job=backend.run(temp_qobj,job_name=pass_name) #asynchronously run the job get results later 
        if args.use_ibm: break
    if gate_df.empty:
        gate_df=pd.DataFrame(new_data)
    elif len(new_data["benchmark"])>0:
        gate_df.append(new_data,ignore_index=True)
    gate_df.to_csv(os.path.join(args.quantum_machine,"pass_gate_data.csv"),index=False)

