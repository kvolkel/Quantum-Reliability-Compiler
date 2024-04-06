# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Routing via SWAP insertion using the SABRE method from Li et al modified with noise aware heuristics."""

import logging
from copy import deepcopy
from itertools import cycle
import numpy as np
 
import math

import retworkx as rx
import networkx as nx


from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.routing.sabre_swap import SabreSwap
from qiskit.dagcircuit import DAGNode

EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

logger = logging.getLogger(__name__)

class NoiseAdaptiveSabreSwap(SabreSwap):
    r"""Map input circuit onto a backend topology via insertion of SWAPs.

    Noise adaptive heuristic extensions for the ``SabreSwap`` routing pass. This pass
    builds off of the base class SabreSwap which handles the main mechanism of inserting 
    SWAP gates into the DAG of the circuit that needs to be routed.

    Instead of relying on a heuristic cost function that purely relies on the 
    hop-distance between nodes in the coupling graph, a reliability distance is used
    to rate the quality of a path. The reliability distance is the probability of success
    when moving a logical qubit along a path in the coupling graph.

    There are 2 options: 
          1. Use solely the SWAP reliability to score SWAPs, this scoring mechanism
             tries to push swaps to align with the path of highest reliability.

          2. Use a heuristic that not only takes into acount the SWAP reliability
             of a path, but also the reliability of staying resident at that qubit.
             Long residence at a qubit implies that we should make sure that the "steady state"
             reliability of the qubit is high. That is, we should not want to execute a long string
             of instructions on a low reliability qubit. Furthermore, we should make sure that 
             we consider measurement error as a given qubit approaches the exection of its measurement operation.

    The main mechanism in implementing (2) is that there is a weight value assigned to each gate success probability.
    The weight is used to score the probability that the given operation will be executed at the current physical
    position for a logical qubit. Based on the execution model of ``SabreSwap`` once the mapping is completed for a pair of
    logical qubits, the mapping is completed for all following instructions that also need the same mapping requirements. This leads
    to unit probability in that sense. As we progress through the successors, scenarios can change how much the weight value should be.
    For example, is there a CNOT instruction with spatially separated qubits, are there active least paths that the current qubit is a part of.
    
    **References:**
    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(self,coupling_map,backend_prop,heuristic='swap_path',tie_heuristic="simple_swap",second_pass=None,seed=None):
        """ Noise adaptive SabreSwap initializer.

        Args:
            backend_prop (BackendProperties): backend properties object
        
        Attributes:
           swap_reliability_matrix: matrix that provides the path of highest swap reliability between 2 physical qubits
           single_gate_reliability: dict that provides the reliability of each different type of gate for each physical qubit
           cx_gate_reliability: matrix that provides the raw CX reliability value between two physical qubits.
        """
        super().__init__(coupling_map,heuristic,tie_heuristic,second_pass,seed) 
        self.swap_reliability_matrix=None
        self.single_gate_reliability=None #gate_type --> list of success rates for each physical qubit
        self.readout_reliability=None
        self.cx_gate_reliability=None
        self.path_matrix=None
        self.T1=0
        self.T2=0
        self.swap_graph = rx.PyDiGraph()
        self.backend_prop=backend_prop
        self._initialize_noise_maps()
        
    def _initialize_noise_maps(self):
        """ Get error rate values from backend properties to initialize reliabiltiy LUTS"""
        #get gate reliability values for gates
        swap_graph_edge_list=[]
        nqbits=len(self.backend_prop.qubits)
        self.cx_gate_reliability=np.zeros((nqbits,nqbits))
        self.single_gate_reliability={}
        self.path_matrix=[[ [] for x in range(nqbits)] for y in range(nqbits)]
        self.swap_reliability_matrix=np.zeros((nqbits,nqbits))
        for gate in self.backend_prop.gates:
            if gate.gate=="cx":
                assert len(gate.qubits)==2
                cx_error_rate=self.backend_prop.gate_property(gate.gate,gate.qubits,'gate_error')[0]
                swap_success_rate=math.pow(1-cx_error_rate,3) #success rate of using this edge on swaps, SWAP=3 CNOTS 
                try:
                    log_swap_success_rate=-math.log(swap_success_rate) #use logs so the cost function is additive
                except ValueError:
                    log_swap_success_rate=50#np.inf
                gate_qubits=gate.qubits
                gate_qubits=sorted(gate_qubits) #cx properties are bidirectional, so only have one cx matrix entry per pai
                try:
                    self.cx_gate_reliability[gate_qubits[0],gate_qubits[1]]=-math.log(1-cx_error_rate) #all success rates stored as -log(p)
                except ValueError:
                    self.cx_gate_reliability[gate_qubits[0],gate_qubits[1]]=50#np.inf
                swap_graph_edge_list.append((gate_qubits[0],gate_qubits[1],log_swap_success_rate))
                swap_graph_edge_list.append((gate_qubits[1],gate_qubits[0],log_swap_success_rate))
                if gate.gate not in self.gate_durations:
                    self.gate_durations[gate.gate]={}
                gate_tuple=tuple(gate_qubits)
                if gate_tuple not in self.gate_durations[gate.gate]:
                    self.gate_durations[gate.gate][tuple(gate_tuple)]=self.backend_prop.gate_length(gate.gate,gate_qubits)
                
            else:
                #gate is a single qubit gate
                assert len(gate.qubits)==1
                qubit=gate.qubits[0]
                gate_error_rate=self.backend_prop.gate_property(gate.gate,gate.qubits,'gate_error')[0]
                if gate.gate not in self.single_gate_reliability:
                    self.single_gate_reliability[gate.gate]=np.zeros(nqbits)
                try:
                    self.single_gate_reliability[gate.gate][qubit]=-math.log(1-gate_error_rate)
                except ValueError:
                    self.single_gate_reliability[gate.gate][qubit]=50
                if gate.gate not in self.gate_durations:
                    self.gate_durations[gate.gate]={}
                if qubit not in self.gate_durations[gate.gate]:
                    self.gate_durations[gate.gate][qubit]=self.backend_prop.gate_length(gate.gate,qubit)
                    
        #get readout reliability values
        self.readout_reliability=np.zeros(nqbits)
        min_T1=10000000
        min_T2=10000000
        for q_index in range(nqbits):
            readout_error=self.backend_prop.qubit_property(q_index,"readout_error")[0]
            try:
                self.readout_reliability[q_index]=-math.log(1-readout_error)
            except ValueError:
                self.readout_reliability[q_index]=50#np.inf
            #get a minimum decoherence time to be conservative 
            if self.backend_prop.t1(q_index)>0 and self.backend_prop.t2(q_index)>0:
                min_T1=min(self.backend_prop.t1(q_index),min_T1)
                min_T2=min(self.backend_prop.t2(q_index),min_T2)
        self.T1=min_T1
        self.T2=min_T2
        self.swap_graph.extend_from_weighted_edge_list(swap_graph_edge_list)
        self.swap_reliability_matrix=rx.digraph_floyd_warshall_numpy(self.swap_graph,lambda w: w)
        self.swap_graph=nx.DiGraph()
        self.swap_graph.add_weighted_edges_from(swap_graph_edge_list) #TODO: Only need this to more easily get shortest paths
        for source in range(nqbits): #get a reference of paths so that we can use this information later
            for dest in range(nqbits):
                self.path_matrix[source][dest]= nx.dijkstra_path(self.swap_graph,source,dest)
        #done with initialization

    def _score_heuristic(self,heuristic,front_layer,extended_set,layout,swap_qubits=None):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on the probability of success
        for the remaining paths between the logical qubits that must be mapped in front_layer (swap_path).
        The other reliability heuristic options
        """
        if heuristic=="swap_path": #evaluate the trial layouts resulting paths based on reliability
            _sum=0#this_swap_success
            for node in front_layer:
                log_qubits=node.qargs
                phys_qubits=[layout[_] for _ in node.qargs]
                assert len(phys_qubits)==2
                path_cost=self.swap_reliability_matrix[phys_qubits[0],phys_qubits[1]]
                _sum+=path_cost
            return _sum
        elif heuristic=="la_swap":
            """
            Swap reliability score heuristic with lookahead identical to Sabre
            """
            first_score=self._score_heuristic("swap_path",front_layer,[],layout,swap_qubits=swap_qubits)
            first_score/=len(front_layer)
            if not extended_set:
                second_score=0.0
            else:
                second_score=self._score_heuristic("swap_path",extended_set,[],layout,swap_qubits=swap_qubits)
                second_score/=len(extended_set)
            return first_score+EXTENDED_SET_WEIGHT*second_score
        elif heuristic=="path_adjust":
            """ Modify the the cost of each front layer path by taking into account asymettry of probabilities on a path,
                penalizes the application of high error rate swaps to converge the same path in a more relaiable way"""
            swap_logical_set=[_ for _ in swap_qubits]
            swap_physical_qubits=[layout[_] for _ in swap_qubits]
            orig_layout=layout.copy()
            orig_layout.swap(*swap_qubits)
            _sum=0
            for node in front_layer:
                log_qubits=node.qargs
                phys_qubits=[layout[_] for _ in node.qargs]
                assert len(phys_qubits)==2
                start_cost=self.swap_reliability_matrix[phys_qubits[0],phys_qubits[1]] #typical path cost
                _sum+=start_cost
                for log_q in log_qubits:
                    if log_q not in swap_logical_set: continue
                    #we have one of the logical qubits from front layer that is in the swap set, adjust its score
                    #get path between target node physical qubits`
                    source=layout[log_q]
                    source_original=orig_layout[log_q]
                    dest=[_ for _ in log_qubits if _!=log_q]
                    dest=dest[0]
                    dest_original=orig_layout[dest]
                    dest=layout[dest]
                    path=self.path_matrix[source][dest]
                    original_cost=self.swap_reliability_matrix[source_original][dest_original]
                    if original_cost<start_cost: continue  #don't consider a degraded path for swap adjustment
                    unused_swap=[path[-1],path[-2]]
                    unused_swap_cost=self.swap_reliability_matrix[unused_swap[0]][unused_swap[1]]
                    adj_cost=-unused_swap_cost+self.swap_reliability_matrix[swap_physical_qubits[0]][swap_physical_qubits[1]]
                    _sum+=adj_cost
            score_original=self._score_heuristic("swap_path",front_layer,extended_set,layout,swap_qubits)
            #print("Score original {} Score adjust {}".format(score_original,_sum))
            return _sum
        elif heuristic=="path_adjust_decoh":
            physical_qubits=[layout[_] for _ in swap_qubits]
            physical_qubits=tuple(sorted(physical_qubits))
            swap_duration=self.gate_durations["cx"][physical_qubits]*3
            #get the score for and adjusted path
            temp_score=self._score_heuristic("path_adjust",front_layer,extended_set,layout,swap_qubits=swap_qubits)
            #now take into account decoherence parameters if the depth of the highest depth qubit is used
            if max(self.logical_qubit_depths.values())==max(self.logical_qubit_depths[swap_qubits[0]],self.logical_qubit_depths[swap_qubits[1]]):
                depth1=self.logical_qubit_depths[swap_qubits[0]]
                depth2=self.logical_qubit_depths[swap_qubits[1]]
                latency=self.logical_qubit_times[swap_qubits[0]] if depth1>depth2 else self.logical_qubit_times[swap_qubits[1]]
                T1_derate=min((latency+swap_duration)/self.T1,1)
                T2_derate=min((latency+swap_duration)/self.T2,1)
                temp_score+=T1_derate+T2_derate
            else:
                #add a base T1 T2 derating factor to non critical path choices to make a fair comparison
                max_lq=0
                lq_at_max=list(self.logical_qubit_depths.keys())[0]
                for lq in self.logical_qubit_depths:
                    if self.logical_qubit_depths[lq]>max_lq:
                        max_lq=self.logical_qubit_depths[lq]
                        lq_at_max=lq
                latency=self.logical_qubit_times[lq_at_max]
                T1_derate=min((latency)/self.T1,1)
                T2_derate=min((latency)/self.T2,1)
                temp_score+=T1_derate+T2_derate
            return temp_score

    def _tie_breaker(self,best_swaps,layout,heuristic):
        """ Extends the base tie breaker to account for differences with reliability paths"""
        if heuristic=="simple_swap":
            swap_success_scores={swap: np.inf for swap in best_swaps}
            for swap_candidate in best_swaps:
                phys_qubits=[layout[_] for _ in swap_candidate]
                phys_qubits=sorted(phys_qubits)
                assert len(phys_qubits)==2
                this_swap_success=3*self.cx_gate_reliability[phys_qubits[0],phys_qubits[1]]
                swap_success_scores[swap_candidate]=this_swap_success
            min_swap_value=min(swap_success_scores.values())
            swaps_with_min=[s for s,v in swap_success_scores.items() if v==min_swap_value] #take swap application with best success
            if len(swaps_with_min)>1:
                logger.debug('Breaking a Tie')
                logger.debug('Length of tie'.format(len(best_swaps)))
                return self.rng.choice(swaps_with_min)
            else:
                return swaps_with_min[0]
            
    def _second_pass(self,swap_scores,initial_layout,front_layer,dag,heuristic,coupling_map):
        """ Performs a second pass on a set of swap candidates that are all physically next to each other in order to take into account 
            single qubit errors"""
        if heuristic==None:
            return swap_scores
        elif heuristic=="simple_exec":
            filtered_swap_scores={}
            done=False
            swap_tuple_list=[(swap,swap_scores[swap]) for swap in swap_scores]
            swap_tuple_list=sorted(swap_tuple_list,key= lambda x: x[1])
            for swap,score in swap_tuple_list:
                trial_layout=initial_layout.copy()
                trial_layout.swap(*swap) #trial layout is after swap application
                for node in front_layer:
                    front_layer_physical_map=[trial_layout[_] for _ in node.qargs]
                    distance=coupling_map.distance(*front_layer_physical_map)
                    if distance==1: #physically co-located qubits, only take those at the top
                        if swap not in filtered_swap_scores:
                            filtered_swap_scores[swap]={"layout":trial_layout,"layout_score":score,"gates":[node]} #collect gates that will execute
                            #TODO: adjust the last SWAP score, if we route to physical neighbors the swap cost between the two nodes should not be considered
                        else:
                            filtered_swap_scores[swap]["gates"].append(node) 
                    else:
                        done=True
                        break
                if done: break
            if not filtered_swap_scores or len(filtered_swap_scores)==1:
                return swap_scores #unchanged swap scores
            else:
                logger.debug('SECOND PASS :: Filtering a set of swaps that co-locate qubits')
                logger.debug("SECOND PASS :: FILTER :: SWAP SET AND SCORES {}".format(filtered_swap_scores)) 
                #want to take into account the costs of executing instructions on this physical mappings
                for swap in filtered_swap_scores:
                    executed_list=[]
                    swap_layout=filtered_swap_scores[swap]["layout"]
                    temp_swap_score=filtered_swap_scores[swap]["layout_score"]
                    nodes_accounted=set()
                    for executeable_gate in filtered_swap_scores[swap]["gates"]:
                        #need to get the cost of this executable gate
                        dead_set=set()
                        #only those successors we have not accounted for yet
                        successor_gen=dag.bfs_successors(executeable_gate)
                        exec_set=set(executeable_gate.qargs)
                        while len(dead_set)!=len(exec_set):
                            try:
                                _,node_successors = next(successor_gen)
                                node_successors=filter(lambda node: (node not in nodes_accounted) and node.type=='op' and node.name!="barrier",node_successors)
                            except StopIteration:
                                break
                            while len(dead_set)!=len(exec_set):
                                try:
                                    successor = next(node_successors)
                                    succ_set=set(successor.qargs)
                                    if bool(succ_set.intersection(dead_set)):
                                        dead_set=dead_set.union(exec_set.intersection(succ_set))
                                        break
                                    if len(successor.qargs)==1 or _is_executable(swap_layout,coupling_map,*successor.qargs):
                                        logger.debug("SECOND PASS :: FOUND EXECUTABLE SUCCESSOR") 
                                        #take this node into account for the cost function
                                        nodes_accounted.add(successor)
                                        exec_cost=self._get_execution_cost(successor,swap_layout)
                                        filtered_swap_scores[swap]["layout_score"]+=exec_cost
                                        continue
                                    else:
                                        #unexecutable gate, select the qubits that die  
                                        dead_set=dead_set.union(exec_set.intersection(succ_set))
                                        continue
                                except StopIteration:
                                    break
                        #lastly, take into account the executeable_gate
                        if executeable_gate not in nodes_accounted:
                            nodes_accounted.add(executeable_gate)
                            exec_cost=self._get_execution_cost(executeable_gate,swap_layout)
                            filtered_swap_scores[swap]["layout_score"]+=exec_cost
                #return the new swap scores
                #print(swap_tuple_list)
                ret={swap:filtered_swap_scores[swap]["layout_score"] for swap in filtered_swap_scores}
                #print(ret)
                return ret

    def _get_execution_cost(self,node,layout):
        """ Using the reliability matrices and data structures, look up the reliability for a DAGNode"""
        assert node.type=="op"
        if node.name=="cx":
            #print("cx data")
            logger.debug('Getting additional execution cost for cx gate')
            p_q_1=layout[node.qargs[0]]
            p_q_2=layout[node.qargs[1]]
            #print(self.cx_gate_reliability[p_q_1][p_q_2])
            return self.cx_gate_reliability[p_q_1][p_q_2]
        elif node.name=="measure":
            #print("measure data")
            logger.debug('Getting additional execution cost for measurement')
            assert len(node.qargs)==1
            phys_node=layout[node.qargs[0]]
            #print(self.readout_reliability[phys_node])
            return self.readout_reliability[phys_node]
        else:
            #print("single gate data")
            logger.debug('Getting additional execution cost for single qubit gate')
            assert len(node.qargs)==1
            phys_node=layout[node.qargs[0]]
            #print(self.single_gate_reliability[node.name][phys_node])
            return self.single_gate_reliability[node.name][phys_node]            
            
                
def _is_executable(layout,coupling_map,logical_q_0,logical_q_1):
    return coupling_map.distance(layout[logical_q_0],layout[logical_q_1])==1
