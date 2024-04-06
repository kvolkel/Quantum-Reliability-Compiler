**This is original research work completed as a requirement for the class ECE792 Quantum Computer Architecture at North Carolina State University**

This is the readme file for running the reliability aware SabreSwap passes. All implementation regarding compiler pass implementation is in 2 files within the qiskit-terra/qiskit/transpiler/passes/routing directory. One file is named noise_adaptive_sabre_swap.py, this is where the code that implements reliability aware heuristics is implemented. The other file is sabre_swap.py, this is the original file for the implementation of SabreSwap but some changes needed to be made in order to work with the reliability metrics implemented in the other file. All added code in sabre_swap.py is marked with a comment starting with my initials, e.g. #KV .... . Note all of this was done on an Ubuntu 16.04 machine, along with the aid of conda environments.

I
Create a conda environment using the following command:

conda env create -f final_proj.yml


After the previous command finishes, activate the environment wherever it get stored. Then, run the following 2 sets of commands to install qiskit (if not installed) and then to install the qiskit-terra portion of qiskit with the new implemntations:

#install qiskit
pip install qiskit

#install the local qiskit-terra

cd qiskit-terra
pip install cython
pip install -r requirements-dev.txt
pip install .

The last step should hopefully override the qiskit-terra portion of the already installed qiskit in the environment. I've mostly run with all compiled from source components of qiskit since compiling from source gets a more up-to-date version not available as release. So, you may see an error on the last command, saying you are using incompatible qiskit-terra 0.17.0. However, it seems that it still does work for what needs to be run for the project.  You should now have an environment that can run the analysis scripts. It should be noted that running noise-aware compiler passes involves querying the target backend, so your environment should be setup with your IBM-Quantum ID information so the following pieces of code work:

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ncsu')


This can be done by running the following in a python environment (script or command line):

from qiskit import IBMQ
IBMQ.save_account('MY_API_TOKEN') #'MY_API_Token' is the token provided by IBM for your account




The front end scripts are at the top level of the project directory, the following is a short description and what they were used for in the reports. Under each file is a description of its parameters:



pass_launcher.py  -- This handles launching the circuits to the provided backend quantum machine. The set of passes are imported from pass_sets.py, and are all applied to each benchmark chosen to be evaluated. Evaluated benchmarks are chosen from the "--benchmarks" option, and from the "--max_qubits" option with screens out all benchmarks that have program sizes larger that "max_qubits". As programs are compiled, their gate counts are logged in a csv that is output in the directory for the selected quantum machine, e.g. "ibmq_toronto/pass_gate_data.csv". If this csv already exists, then it won't be overwritten with new compilation data according to the new calibration data for the machine, nor will the pass be launched to the quantum machine. Launching new runs and getting up-to-date data can be done by either deleting the "pass_gate_data.csv", or using the option "--overwrite".  Instead of running the compilation process, "--plot" can be used to directly plot normalized depths and gate counts from an already existing "pass_gate_data".csv. Trying to launch passes to validate the paper's results will likely take a very long time, so the raw data generated on gate counts at the time of this report is given after the options

OPTIONS for pass_launcher.py:
--quantum_machine "machine name": IBM name of quantum machine to be used, defaults to toronto  e.g. ibmq_toronto,ibmq_16_melbourne,ibmq_manhattan
--benchmarks "benchmark path": Path to the benchmarks to be run, defaults to the noise_adaptive_benchmarks directory.
--shots "num_shots": integer for the number of times each circuit should run, default is 8192 shots.
--overwrite: flag to overwrite existing "pass_gate_data.csv", defaults to False (no overwriting)
--no_sim: used to toggle whether to run simulation or not, used to validate correct outputs of the circuits. adding this flag disables simulation.
--max_qubits "num_qubits": maximum number of qubits a program can have, defaults to 10

#generating normalized depth and gate counts figures with existing data
python pass_launcher.py --quantum_machine ibmq_toronto --max_qubits 10 --no_sim --plot 
python pass_launcher.py --quantum_machine ibmq_manhattan --max_qubits 10 --no_sim --plot 
python pass_launcher.py --quantum_machine ibmq_16_melbourne --max_qubits 10 --no_sim --plot 

#If you want to launch to the respective machines and make new data
python pass_launcher.py --quantum_machine ibmq_toronto --max_qubits 10 --no_sim --overwrite
python pass_launcher.py --quantum_machine ibmq_manhattan --max_qubits 10 --no_sim --overwrite
python pass_launcher.py --quantum_machine ibmq_16_melbourne --max_qubits 10 --no_sim --overwrite 






plot_machine_results.py  -- This script calls back to the backend provided by the user to get the most recent data from IBM for each pass and benchmark. The results of PST, IST, and correct answer rank are aggregated in a csv file at the path: "quantum_machine"/raw_machine_results.csv. This is a csv of raw results that can be loaded again and again to make changes to the plotting portion of the script, while skipping the long time of querying IBM for the data. If it exists, it is automatically used for plotting, get new results from IBM requires removing the file from the machine's directory.

OPTIONS for plot_machine_results.py

--quantum_machine "machine name" : machine to get results and make plots for
--benchmarks "benchmark directory path": Directory used for benchmarks defaults to "noise_adaptive_benchmarks"
--max_qubits "num_qubits": maximum number of qubits a program can have, defaults to 10

Using the existing csv files, the figures and average ranks in the report can be generated with the below 3 commands. Note the average benchmark ranks in Table 4 need a little hand work. Each of the following commands calculates the average across the 2 instances of program sizes, but do not average across machine. However, that is straightforward. The average across all benchmarks for a given machine is directly given after running the commands at "quantum machine"/avg_rank_data.csv.

#generating the normalized PST, rank, and average rank data
python plot_machine_results.py --max_qubits 10 --quantum_machine ibmq_16_melbourne
python plot_machine_results.py --max_qubits 10 --quantum_machine ibmq_manhattan
python plot_machine_results.py --max_qubits 10 --quantum_machine ibmq_toronto




reliability_value_plotter.py -- Queries the backend given in the arguments and plots out error rates for CNOTs, measurements, and single qubit gates. This was used to generate Figure 2, but I unfortunately did not log the exact data used to plot this, and it changes day to day. To generate a figure similar to Figure 2, use the following command. This script has only one option: --quantum_machine "machine name" which is the name of the machine that you want to get reliability data for.

#generate toronto error rate plot
python reliability_value_plotter.py --quantum_machine ibmq_toronto



pass_sets.py -- This script is imported by other front layer scripts. It's main purpose is to provide the set of passes that will be evaluated. It should not by directly invoked.




parse_ibm_file.py -- parses the csv file that holds backend error rate information and relaxation times for a certain machine, these csv files can be downloaded from the IBM Quantum Experience user interface online, they may change day to day though. So, the analyzed csvs are stored in each machines' respective directory e.g. ibmq_toronto/ibmq_toronto.csv. This file only really calculates average error rates for CNOT gates. It does not make the figures that plot out the error rates, that is handled by reliability_value_plotter.py

--quantum_machine "machine name" : name of the machine that is to be analyzed, this path must exist prior to invoking this script
