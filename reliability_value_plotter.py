"""
filename: reliability_value_plotter.py
Description: Generates a set of plots for a machine showing the reliability_value_plotter
values for certain gates.
Author: Kevin Volkel
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pick
import os
import matplotlib.pyplot as plt
import seaborn as sns

# importing Qiskit
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit.compiler import transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.transpiler.layout import Layout

import pandas as pd
# import basic plot tools
from qiskit.visualization import plot_histogram
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ncsu')



if __name__=="__main__":
    import argparse
    import os
    
    parser=argparse.ArgumentParser(description="Tool to query the reliability data of a given quantum machine")                                                                      
    parser.add_argument('--quantum_machine',dest='quantum_machine',action="store",default="ibmq_toronto",help="path to the data frame we want to analyze")                            
    args=parser.parse_args()


    if not os.path.exists(args.quantum_machine):
        os.mkdir(args.quantum_machine) 

    
    backend_properties=provider.get_backend(args.quantum_machine).properties()
    nqbits=len(backend_properties.qubits)
    
    single_gate_types=[_ for _ in backend_properties._gates if _!="cx"]

    single_qubit_data={x:[0]*nqbits for x in single_gate_types}
    readout_error={"readout_error":[0]*nqbits}
    cnot_error_data={"Qubit IDs":[], "CNOT Error Rate":[]}
        
    #get gate errors for single qubits and cnot gates
    for gate in backend_properties.gates:
        if gate.gate!="cx":
            qubit=gate.qubits[0]
            for gate_info in gate.parameters:
                if gate_info.name=="gate_error":
                        single_qubit_data[gate.gate][qubit]=gate_info.value
        else:
            cnot_error_data["Qubit IDs"]+=gate.qubits
            for gate_info in gate.parameters:
                if gate_info.name=="gate_error":
                        cnot_error_data["CNOT Error Rate"]+=[gate_info.value]*2
    #get readout error rates
    for q in range(nqbits):
        r=backend_properties.qubit_property(q,"readout_error")[0]
        readout_error["readout_error"][q]=r
    
    single_qubit_df=pd.DataFrame(single_qubit_data)
    single_qubit_df["Qubit IDs"]=range(nqbits)

    readout_error_df=pd.DataFrame(readout_error)
    readout_error_df["Qubit IDs"]=range(nqbits)

    cnot_error_df=pd.DataFrame(cnot_error_data)
    
    
    fig_array=[]
   
    fig,ax = plt.subplots()
    single_qubit_df.set_index('Qubit IDs', inplace=True)
    sns.heatmap(single_qubit_df,linewidths=0.5,ax=ax,annot_kws={"fontsize":3},cmap="YlGnBu")
    ax.set_title(args.quantum_machine)
    fig_array.append((fig,os.path.join(args.quantum_machine,"single_gate.pdf")))
    
    
    fig,ax = plt.subplots()
    readout_error_df.set_index('Qubit IDs', inplace=True)
    sns.heatmap(readout_error_df,linewidths=0.5,ax=ax,annot_kws={"fontsize":3},cmap="YlGnBu")
    ax.set_title(args.quantum_machine)
    fig_array.append((fig,os.path.join(args.quantum_machine,"readout_error.pdf")))


    #line plot to compare readout errors and single qubit errors
    fig,ax=plt.subplots(figsize=(10,4))
    single_qubit_df.plot.line(lw=3,ax=ax)
    readout_error_df["readout_error"].plot.line(ax=ax,lw=3,logy=True,linestyle='--')
    ax.get_legend().remove()       
    ax.get_figure().legend()                                            
    ax.set_ylabel("Readout Error",fontsize=12)    
    ax.set_xlabel("Qubit IDs",fontsize=12)
    ax.set_ylabel("Single Gate Errors",fontsize=12)
    ax.set_title(args.quantum_machine)
    ax.set_ylim(0.000001,1)

    fig_array.append((fig,os.path.join(args.quantum_machine,"readout_single_error.pdf")))

    if max(cnot_error_data["CNOT Error Rate"])>0.5:
        #line plot to compare CNOT errors and single qubit errors
        fig,axes=plt.subplots(2,1,figsize=(10,4),sharex=True)
        #single_qubit_df.plot.line(lw=3,ax=axes[0])
        axes0_1=cnot_error_df.plot.line(ax=axes[0],secondary_y=True,x='Qubit IDs',y='CNOT Error Rate',c='DarkBlue',style='x')
        single_qubit_df.plot.line(lw=3,ax=axes[1])
        axes1_1=cnot_error_df.plot.line(ax=axes[1],secondary_y=True,x='Qubit IDs',y='CNOT Error Rate',c='DarkBlue',style='x')
        axes[0].get_legend().remove()                                            
        axes[1].get_legend().remove()
        axes1_1.set_ylabel("CNOT Error",fontsize=12)    
        axes0_1.set_xlabel("Qubit IDs",fontsize=12)
        axes[1].set_ylabel("Single Gate Errors",fontsize=12)
        _max=0
        for value in cnot_error_data["CNOT Error Rate"]:
            if value==1.0: continue
            _max=max(_max,value)
            #print(cnot_error_data["CNOT Error Rate"])
        axes0_1.set_ylim(_max+0.05, 1.05)
        axes1_1.set_ylim(0,_max+0.05)
        axes[0].spines['bottom'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[0].xaxis.tick_top()
        axes[0].tick_params(labeltop='off')
        axes[1].xaxis.tick_bottom()
        d = .015
        kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False)
        axes[0].plot((-d,+d),(-d,+d), **kwargs)
        axes[0].plot((1-d,1+d),(-d,+d), **kwargs)
        kwargs.update(transform=axes[1].transAxes)
        axes[1].plot((-d,+d),(1-d,1+d), **kwargs)
        axes[1].plot((1-d,1+d),(1-d,1+d), **kwargs)
        axes[0].set_title(args.quantum_machine)
        fig_array.append((fig,os.path.join(args.quantum_machine,"cnot_single_error.pdf")))

    else:
        #line plot to compare readout errors and single qubit errors
        fig,ax=plt.subplots(figsize=(10,4))
        single_qubit_df.plot.line(lw=3,ax=ax)
        ax1=cnot_error_df.plot.line(ax=ax,x='Qubit IDs',y='CNOT Error Rate',c='DarkBlue',style='x',logy=True)
        ax.get_legend().remove()       
        ax.get_figure().legend()
        ax.set_ylim(0.0001,0.1)
        ax.set_xlabel("Qubit IDs",fontsize=12)
        ax.set_ylabel("Gate Error Rate",fontsize=12)
        ax.set_title(args.quantum_machine)
        fig_array.append((fig,os.path.join(args.quantum_machine,"cnot_single_error.pdf")))

    #plot CNOT and readout error on the same chart
    fig,ax=plt.subplots(figsize=(10,4))
    cnot_error_df.plot.line(ax=ax,x='Qubit IDs',y='CNOT Error Rate',c='Red',style='x')
    readout_error_df["readout_error"].plot.line(ax=ax,lw=3,linestyle='--',logy=True)  
    ax.set_xlabel("Qubit IDs",fontsize=12)
    ax.set_ylabel("Error Rate",fontsize=12)
    ax.set_title(args.quantum_machine)
    ax.get_legend().remove()       
    ax.get_figure().legend()
    ax.set_ylim(0.0001,1)        
    fig_array.append((fig,os.path.join(args.quantum_machine,"readout_cnot_error.pdf")))


    #plot CNOT and readout error on the same chart
    fig,ax=plt.subplots(figsize=(10,4))
    cnot_error_df.plot.line(ax=ax,x='Qubit IDs',y='CNOT Error Rate',c='Red',style='x')
    readout_error_df["readout_error"].plot.line(ax=ax,lw=2,linestyle='--',logy=True)
    single_qubit_df.plot.line(lw=2,ax=ax)
    ax.set_xlabel("Qubit IDs",fontsize=12)
    ax.set_ylabel("Error Rate",fontsize=12)
    ax.set_title(args.quantum_machine)
    ax.get_legend().remove()       
    ax.get_figure().legend()
    ax.set_ylim(0.0001,1)        
    fig_array.append((fig,os.path.join(args.quantum_machine,"readout_cnot_single_error.pdf")))
    

    
    for fig,path in fig_array: fig.savefig(path,format="pdf")
