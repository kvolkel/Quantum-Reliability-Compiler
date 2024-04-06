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

"""Routing via SWAP insertion using the SABRE method from Li et al."""

import logging
from copy import deepcopy
from itertools import cycle
import numpy as np

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode

logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = 20     # Size of lookahead window. TODO: set dynamically to len(current_layout)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001         # Decay cooefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5   # How often to reset all decay rates to 1.


class SabreSwap(TransformationPass):
    r"""Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of the SWAP-based heuristic search from the SABRE qubit
    mapping paper [1] (Algorithm 1). The hueristic aims to minimize the number
    of lossy SWAPs inserted and the depth of the circuit.

    This algorithm starts from an initial layout of virtual qubits onto physical
    qubits, and iterates over the circuit DAG until all gates are exhausted,
    inserting SWAPs along the way. It only considers 2-qubit gates as only those
    are germane for the mapping problem (it is assumed that 3+ qubit gates are
    already decomposed).

    In each iteration, it will first check if there are any gates in the
    ``front_layer`` that can be directly applied. If so, it will apply them and
    remove them from ``front_layer``, and replenish that layer with new gates
    if possible. Otherwise, it will try to search for SWAPs, insert the SWAPs,
    and update the mapping.

    The search for SWAPs is restricted, in the sense that we only consider
    physical qubits in the neighoborhood of those qubits involved in
    ``front_layer``. These give rise to a ``swap_candidate_list`` which is
    scored according to some heuristic cost function. The best SWAP is
    implemented and ``current_layout`` updated.

    **References:**

    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(self, coupling_map, heuristic='basic', tie_heuristic=None,second_pass=None,seed=None):
        r"""SabreSwap initializer.

        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            heuristic (str): The type of heuristic to use when deciding best
                swap strategy ('basic' or 'lookahead' or 'decay').
            seed (int): random seed used to tie-break among candidate swaps.

        Additional Information:

            The search space of possible SWAPs on physical qubits is explored
            by assigning a score to the layout that would result from each SWAP.
            The goodness of a layout is evaluated based on how viable it makes
            the remaining virtual gates that must be applied. A few heuristic
            cost functions are supported

            - 'basic':

            The sum of distances for corresponding physical qubits of
            interacting virtual qubits in the front_layer.

            .. math::

                H_{basic} = \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]

            - 'lookahead':

            This is the sum of two costs: first is the same as the basic cost.
            Second is the basic cost but now evaluated for the
            extended set as well (i.e. :math:`|E|` number of upcoming successors to gates in
            front_layer F). This is weighted by some amount EXTENDED_SET_WEIGHT (W) to
            signify that upcoming gates are less important that the front_layer.

            .. math::

                H_{decay}=\frac{1}{\left|{F}\right|}\sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]
                    + W*\frac{1}{\left|{E}\right|} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]

            - 'decay':

            This is the same as 'lookahead', but the whole cost is multiplied by a
            decay factor. This increases the cost if the SWAP that generated the
            trial layout was recently used (i.e. it penalizes increase in depth).

            .. math::

                H_{decay} = max(decay(SWAP.q_1), decay(SWAP.q_2)) {
                    \frac{1}{\left|{F}\right|} \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]\\
                    + W *\frac{1}{\left|{E}\right|} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]
                    }
        """

        super().__init__()
        self.coupling_map = coupling_map
        self.heuristic = heuristic
        self.seed = seed
        self.applied_gates = None
        self.qubits_decay = None
        #extra heuristic parameters

        #KV: added data structures for reliability aware routing
        self.tie_heuristic=tie_heuristic
        self.second_pass=second_pass
        self.logical_qubit_depths={}
        self.logical_qubit_times={} #tracks the propagation time through the gates of the qubit to get a time estimate to derate decoherence
        self.gate_durations={}


    def _add_swap_times(self,best_swap,layout): #KV: Add to the logical qubit depth values, and add swap latencies
        """Using the layout before best_swap is applied to its logical qubits, add the qubit depth values and the swap latency """
        physical_qubits=[layout[_] for _ in best_swap]
        physical_qubits=tuple(sorted(physical_qubits))
        swap_duration=self.gate_durations["cx"][physical_qubits]*3
        #print(self.logical_qubit_depths)
        for lq in best_swap:
            self.logical_qubit_depths[lq]+=3 #3 gates on critical path now
            self.logical_qubit_times[lq]+=swap_duration
        
    def _reliability_swap_init(self,dag): #KV:Initialization for reliability
        """Initialization for reliablity aware routing that depends on state that is only visible to the base class"""
        canonical_register = dag.qregs['q']
        for node in dag.gate_nodes():
            for logical_q in node.qargs:
                if logical_q not in self.logical_qubit_depths:
                    self.logical_qubit_depths[logical_q]=1
                else:
                    self.logical_qubit_depths[logical_q]+=1
        for q in canonical_register:
            if q not in self.logical_qubit_depths:
                self.logical_qubit_depths[q]=0
        self.logical_qubit_times={qubit:0.0 for qubit in canonical_register}
        
    def run(self, dag):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('Sabre swap runs on physical circuits only.')

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError('More virtual qubits exist than physical.')

        self.rng = np.random.default_rng(self.seed)

        #KV: Added some initialization that needs dag
        self._reliability_swap_init(dag)

        
        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = dag._copy_circuit_metadata()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        self.coupling_map.make_symmetric()

        canonical_register = dag.qregs['q']
        current_layout = Layout.generate_trivial_layout(canonical_register)

        # A decay factor for each qubit used to heuristically penalize recently
        # used qubits (to encourage parallelism).
        self.qubits_decay = {qubit: 1 for qubit in dag.qubits}

        # Start algorithm from the front layer and iterate until all gates done.
        num_search_steps = 0
        front_layer = dag.front_layer()
        self.applied_gates = set()
        while front_layer:
            execute_gate_list = []
            # Remove as many immediately applicable gates as possible
            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    physical_qubits = (current_layout[v0], current_layout[v1])
                    if physical_qubits in self.coupling_map.get_edges():
                        execute_gate_list.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)

            if execute_gate_list:
                for node in execute_gate_list:
                    new_node = _transform_gate_for_layout(node, current_layout)
                    mapped_dag.apply_operation_back(new_node.op,
                                                    new_node.qargs,
                                                    new_node.cargs,
                                                    new_node.condition)
                    front_layer.remove(node)
                    self.applied_gates.add(node)
                    for successor in dag.quantum_successors(node):
                        if successor.type != 'op':
                            continue
                        if self._is_resolved(successor, dag):
                            front_layer.append(successor)

                    if node.qargs:
                        self._reset_qubits_decay()
                        #KV: Add latency accumulations
                        if len(self.gate_durations)>0 and node.name!="barrier" and node.name!="measure":
                            gate_type=node.name
                            if len(node.qargs)>1:
                                physical_qubit=[current_layout[_] for _ in node.qargs]
                                physical_qubit=tuple(sorted(physical_qubit))
                            else:
                                physical_qubit=current_layout[node.qargs[0]]
                            for lq in node.qargs:
                                self.logical_qubit_times[lq]+=self.gate_durations[gate_type][physical_qubit]

                # Diagnostics
                logger.debug('free! %s',
                             [(n.name, n.qargs) for n in execute_gate_list])
                logger.debug('front_layer: %s',
                             [(n.name, n.qargs) for n in front_layer])

                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            extended_set = self._obtain_extended_set(dag, front_layer)
            swap_candidates = self._obtain_swaps(front_layer, current_layout)
            swap_scores = dict.fromkeys(swap_candidates, 0)
            filtered_swap_list=[]
            for swap_qubits in swap_scores:
                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)
                
		#KV:Decoherence adjustments
                if self.heuristic=="path_adjust_decoh": #Try to ensure that we move forward with the decoherence strategy
                    leave=False
                    for node in front_layer:
                        log_qubits=node.qargs
                        phys_qubits=[trial_layout[_] for _ in node.qargs]
                        start_cost=self.swap_reliability_matrix[phys_qubits[0],phys_qubits[1]] #typical path cost
                        for log_q in log_qubits:
                            if log_q not in swap_qubits: continue
                            source=trial_layout[log_q]
                            source_original=current_layout[log_q]
                            dest=[_ for _ in log_qubits if _!=log_q]
                            dest=dest[0]
                            dest_original=current_layout[dest]
                            dest=trial_layout[dest]
                            original_cost=self.swap_reliability_matrix[source_original][dest_original]
                            if original_cost<start_cost: continue  #don't consider a degraded path for swap adjustment
                            filtered_swap_list.append(swap_qubits) #save this for later, take all that make forward progress
                            leave=True
                            break
                        if leave: break
                else:
                    score = self._score_heuristic(self.heuristic,
                                                  front_layer,
                                                  extended_set,
                                                  trial_layout,
                                                  swap_qubits)
                    swap_scores[swap_qubits] = score
                    
            if self.heuristic=="path_adjust_decoh": #KV:calculate swap scores on filtered swaps
                swap_scores={swap_qubits:0 for swap_qubits in filtered_swap_list} 
                for swap_qubits in filtered_swap_list:
                    trial_layout = current_layout.copy()
                    trial_layout.swap(*swap_qubits)
                    score = self._score_heuristic(self.heuristic,
                                                  front_layer,
                                                  extended_set,
                                                  trial_layout,
                                                  swap_qubits)
                    swap_scores[swap_qubits] = score
            swap_scores=self._second_pass(swap_scores,current_layout,front_layer,dag,self.second_pass,self.coupling_map) #KV: Second pass 
            min_score = min(swap_scores.values())
            #print(min_score)
            best_swaps = [k for k, v in swap_scores.items() if v == min_score]
            best_swaps.sort(key=lambda x: (x[0].index, x[1].index))
            #allow for a tie breaker to be done instead of random selection
            best_swap = self._tie_breaker(best_swaps,current_layout,self.tie_heuristic)
            
            swap_node = DAGNode(op=SwapGate(), qargs=best_swap, type='op')
            swap_node = _transform_gate_for_layout(swap_node, current_layout)
            mapped_dag.apply_operation_back(swap_node.op, swap_node.qargs)
            
            if len(self.gate_durations)>0: #KV: add swap times for the best swap
                self._add_swap_times(best_swap,current_layout)
                
            current_layout.swap(*best_swap)


            num_search_steps += 1
            if num_search_steps % DECAY_RESET_INTERVAL == 0:
                self._reset_qubits_decay()
            else:
                self.qubits_decay[best_swap[0]] += DECAY_RATE
                self.qubits_decay[best_swap[1]] += DECAY_RATE

            # Diagnostics
            logger.debug('SWAP Selection...')
            logger.debug('extended_set: %s',
                         [(n.name, n.qargs) for n in extended_set])
            logger.debug('swap scores: %s', swap_scores)
            logger.debug('best swap: %s', best_swap)
            logger.debug('qubits decay: %s', self.qubits_decay)
        self.property_set['final_layout'] = current_layout
        return mapped_dag

    def _reset_qubits_decay(self):
        """Reset all qubit decay factors to 1 upon request (to forget about
        past penalizations).
        """
        self.qubits_decay = {k: 1 for k in self.qubits_decay.keys()}

    def _is_resolved(self, node, dag):
        """Return True if all of a node's predecessors in dag are applied.
        """
        predecessors = dag.quantum_predecessors(node)
        predecessors = filter(lambda x: x.type == 'op', predecessors)
        return all([n in self.applied_gates for n in predecessors])

    def _obtain_extended_set(self, dag, front_layer):
        """Populate extended_set by looking ahead a fixed number of gates.
        For each existing element add a successor until reaching limit.
        """
        # TODO: use layers instead of bfs_successors so long range successors aren't included.
        extended_set = set()
        bfs_successors_pernode = [dag.bfs_successors(n) for n in front_layer]
        node_lookahead_exhausted = [False] * len(front_layer)
        for i, node_successor_generator in cycle(enumerate(bfs_successors_pernode)):
            if all(node_lookahead_exhausted) or len(extended_set) >= EXTENDED_SET_SIZE:
                break

            try:
                _, successors = next(node_successor_generator)
                successors = list(filter(lambda x: x.type == 'op' and len(x.qargs) == 2,
                                         successors))
            except StopIteration:
                node_lookahead_exhausted[i] = True
                continue

            successors = iter(successors)
            while len(extended_set) < EXTENDED_SET_SIZE:
                try:
                    extended_set.add(next(successors))
                except StopIteration:
                    break

        return extended_set

    def _obtain_swaps(self, front_layer, current_layout):
        """Return a set of candidate swaps that affect qubits in front_layer.

        For each virtual qubit in front_layer, find its current location
        on hardware and the physical qubits in that neighborhood. Every SWAP
        on virtual qubits that corresponds to one of those physical couplings
        is a candidate SWAP.

        Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not duplicated.
        """
        candidate_swaps = set()
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_layout[virtual]
                for neighbor in self.coupling_map.neighbors(physical):
                    virtual_neighbor = current_layout[neighbor]
                    swap = sorted([virtual, virtual_neighbor],
                                  key=lambda q: (q.register.name, q.index))
                    candidate_swaps.add(tuple(swap))

        return candidate_swaps

    def _score_heuristic(self, heuristic, front_layer, extended_set, layout, swap_qubits=None):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        if heuristic == 'basic':
            return sum(self.coupling_map.distance(*[layout[q] for q in node.qargs])
                       for node in front_layer)

        elif heuristic == 'lookahead':
            first_cost = self._score_heuristic('basic', front_layer, [], layout)
            first_cost /= len(front_layer)

            second_cost = self._score_heuristic('basic', extended_set, [], layout)
            second_cost = 0.0 if not extended_set else second_cost / len(extended_set)

            return first_cost + EXTENDED_SET_WEIGHT * second_cost

        elif heuristic == 'decay':
            return max(self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]]) * \
                   self._score_heuristic('lookahead', front_layer, extended_set, layout)

        else:
            raise TranspilerError('Heuristic %s not recognized.' % heuristic)

    def _tie_breaker(self,best_swaps,layout,heuristic):
        """ Tie breaker function so we can overide it"""
        if len(best_swaps)==1:
            return best_swaps[0]
        else:
            return self.rng.choice(best_swaps)

    def _second_pass(self,swap_scores,initial_layout,front_layer,dag,heuristic,coupling_map):
        """ Performs a second pass on a set of sorted swap scores"""
        return swap_scores

    
def _transform_gate_for_layout(op_node, layout):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = deepcopy(op_node)

    device_qreg = op_node.qargs[0].register
    premap_qargs = op_node.qargs
    mapped_qargs = map(lambda x: device_qreg[layout[x]], premap_qargs)
    mapped_op_node.qargs = mapped_op_node.op.qargs = list(mapped_qargs)

    return mapped_op_node


