
#specific passes needed
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import ASAPSchedule
from qiskit.transpiler.passes import CheckCXDirection
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import TimeUnitAnalysis
#import Layouts
from qiskit.transpiler.passes.layout.noise_adaptive_layout import NoiseAdaptiveLayout
from qiskit.transpiler.passes.layout.sabre_layout import SabreLayout
from qiskit.transpiler.passes.layout.trivial_layout import TrivialLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla

#import Routers 
from qiskit.transpiler.passes.routing.basic_swap import BasicSwap
from qiskit.transpiler.passes.routing.sabre_swap import SabreSwap
from qiskit.transpiler.passes.routing.noise_adaptive_sabre_swap import NoiseAdaptiveSabreSwap


#import basis changers
from qiskit.transpiler.passes.basis.unroller import Unroller
    
    
PASS_NAMES=[
    "base+base",
    "noise+base",
    "base+sabre",
    "noise+sabre",
    "sabre+sabre",
    "base+noise",
    "sabre+noise",
    "noise+noise",
    "noise+noise_la",
    "noise+adjust",
    "noise+noise+exec",
    "noise+adj+exec",
    "noise+adj+dec+exec",
    "noise+adj+dec"
]

def get_passes(backend_basis,backend_coupling_map,backend_properties,backend_durations):
  
    pass_sets={}
    for pass_name in PASS_NAMES:
        pass_sets[pass_name]=[]
   
    for pass_name in pass_sets:
        start_passes=[TrivialLayout(backend_coupling_map),FullAncillaAllocation(backend_coupling_map),EnlargeWithAncilla(),ApplyLayout()]
        decompose_after_layout=[ApplyLayout(),Unroll3qOrMore(),Unroller(backend_basis),CheckMap(backend_coupling_map),BarrierBeforeFinalMeasurements()]
        end_passes=[CheckCXDirection(backend_coupling_map),CXDirection(backend_coupling_map),Unroller(backend_basis)]#TimeUnitAnalysis(backend_durations),ASAPSchedule(backend_durations),Unroller(backend_basis)]
        end_passes=[Unroller(backend_basis)]
        pass_specific_layout_swap=[]
        if pass_name=="base+base":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[TrivialLayout(backend_coupling_map)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[BasicSwap(backend_coupling_map)]
            pass_sets[pass_name]+=end_passes
        elif pass_name=="noise+base":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[NoiseAdaptiveLayout(backend_properties)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[BasicSwap(backend_coupling_map)]
            pass_sets[pass_name]+=end_passes
        elif pass_name=="base+sabre":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[TrivialLayout(backend_coupling_map)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[SabreSwap(backend_coupling_map,heuristic='decay',seed=11)]
            pass_sets[pass_name]+=end_passes
        elif pass_name=="noise+sabre":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[NoiseAdaptiveLayout(backend_properties)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[SabreSwap(backend_coupling_map,heuristic='decay',seed=11)]
            pass_sets[pass_name]+=end_passes
        elif pass_name == "sabre+sabre":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[SabreLayout(backend_coupling_map,routing_pass=SabreSwap(backend_coupling_map,heuristic='decay',seed=11),seed=11)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[SabreSwap(backend_coupling_map,heuristic='decay',seed=11)]
            pass_sets[pass_name]+=end_passes
        elif pass_name=="base+noise":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[TrivialLayout(backend_coupling_map)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[NoiseAdaptiveSabreSwap(backend_coupling_map,backend_properties,heuristic='swap_path',seed=11)]
            pass_sets[pass_name]+=end_passes
        elif pass_name =="sabre+noise":
            pass_sets[pass_name]+=start_passes+[Unroller(backend_basis)] #add unroller pass in beginning too to make sure noise just sees the basis gates
            pass_sets[pass_name]+=[SabreLayout(backend_coupling_map,routing_pass=NoiseAdaptiveSabreSwap(backend_coupling_map,backend_properties,heuristic='swap_path',seed=11),seed=11)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[NoiseAdaptiveSabreSwap(backend_coupling_map,backend_properties,heuristic='swap_path',seed=11)]
            pass_sets[pass_name]+=end_passes
        elif pass_name=="noise+noise":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[NoiseAdaptiveLayout(backend_properties)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[NoiseAdaptiveSabreSwap(backend_coupling_map,backend_properties,heuristic='swap_path',seed=11)]
            pass_sets[pass_name]+=end_passes
        elif pass_name=="noise+noise_la":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[NoiseAdaptiveLayout(backend_properties)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[NoiseAdaptiveSabreSwap(backend_coupling_map,backend_properties,heuristic='la_swap',seed=11)]
            pass_sets[pass_name]+=end_passes
        elif pass_name=="noise+adjust":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[NoiseAdaptiveLayout(backend_properties)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[NoiseAdaptiveSabreSwap(backend_coupling_map,backend_properties,heuristic='path_adjust',seed=11)]
            pass_sets[pass_name]+=end_passes
        elif pass_name=="noise+noise+exec":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[NoiseAdaptiveLayout(backend_properties)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[NoiseAdaptiveSabreSwap(backend_coupling_map,backend_properties,heuristic='swap_path',second_pass="simple_exec",seed=11)]
            pass_sets[pass_name]+=end_passes
        elif pass_name=="noise+adj+exec":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[NoiseAdaptiveLayout(backend_properties)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[NoiseAdaptiveSabreSwap(backend_coupling_map,backend_properties,heuristic='path_adjust',second_pass="simple_exec",seed=11)]
            pass_sets[pass_name]+=end_passes
        
        elif pass_name=="noise+adj+dec+exec":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[NoiseAdaptiveLayout(backend_properties)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[NoiseAdaptiveSabreSwap(backend_coupling_map,backend_properties,heuristic='path_adjust_decoh',second_pass="simple_exec",seed=11)]
            pass_sets[pass_name]+=end_passes

        elif pass_name=="noise+adj+dec":
            pass_sets[pass_name]+=start_passes
            pass_sets[pass_name]+=[NoiseAdaptiveLayout(backend_properties)]
            pass_sets[pass_name]+=decompose_after_layout
            pass_sets[pass_name]+=[NoiseAdaptiveSabreSwap(backend_coupling_map,backend_properties,heuristic='path_adjust_decoh',seed=11)]
            pass_sets[pass_name]+=end_passes
            
    return pass_sets

def pass_names():
    return PASS_NAMES
