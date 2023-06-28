# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:26:37 2022

@author: EsteJalovecJ

@co-author: Ivan Alsina Ferrer
@co-author: Leonhard Richter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyqubo import Spin
import itertools

from scipy.optimize import minimize
from qiskit.visualization import plot_histogram

from qiskit.algorithms.optimizers import COBYLA
from pyqubo import Array, Binary

from colorama import Fore, Back, Style

from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA

from qiskit.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute, assemble, IBMQ, transpile
from qiskit.circuit import ParameterVector, Parameter

from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, RuntimeDecoder
from qiskit.providers.ibmq import least_busy
from qiskit_ibm_runtime.program import UserMessenger

from qiskit.utils.mitigation import complete_meas_cal, CompleteMeasFitter


from qiskit.tools.monitor import job_monitor

import numpy as np
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram

import numpy as np
from scipy.optimize import OptimizeResult

import os
import json
from json import JSONEncoder

import plotly.express as px
import plotly.graph_objects as go

def plot_schedule_top_counts(counts, solutions, experiment_series): 
    hobo_counts = hobo_to_unpruned_qubo_counts(counts, experiment_series)
    schedules = [s for s,v in get_top_counts(hobo_counts, solutions).items()]
    
    for schedule in schedules: 
        print('\nSchedule: ',schedule)
        print('Feasible:', bool(is_feasible(schedule, experiment_series)))
        print('Energy:', evaluate_wrapper(schedule,experiment_series.active_experiment.Q, experiment_series.active_experiment.offset ))
        print('Probility of sampling this schedule: ', hobo_counts[schedule]/sum(hobo_counts.values()))
        plot_bs(schedule, experiment_series)
        

def apply_one_qubit_gate(circ,qubit, gate, parameter):
    if gate == 'rx':
        circ.rx(parameter, qubit)
    elif gate == 'ry':
        circ.ry(parameter, qubit)
    elif gate == 'rz':
        circ.rz(parameter, qubit)

    return circ

def apply_two_qubit_gate(circ, q1, q2, gate, parameter = None):
    
    if gate == 'cx':
        circ.cx(q1,q2)
    elif gate == 'rzz':
        circ.rzz(parameter, q1,q2)
    elif gate == 'rzx':
        circ.rzx(parameter, q1,q2)
    elif gate == 'rxx':
        circ.rxx(parameter, q1,q2)

    return circ

def get_nparams(nqubits, vqe_layers, twoqubit_gate, entanglement): 
    
    if twoqubit_gate =='cx': 
        parametrised_2qubit_gate = False
    else: 
        parametrised_2qubit_gate=True
        
    
    if entanglement == 'linear0':
        if parametrised_2qubit_gate: 
            nparams = nqubits +2*nqubits*vqe_layers - vqe_layers
        else: 
            nparams = (vqe_layers+1)*nqubits
            
    elif entanglement == 'linear1':
        if parametrised_2qubit_gate: 
            nparams = 3*(nqubits-1)*vqe_layers
        else: 
            nparams = 2*(nqubits-1)*vqe_layers
        
    elif entanglement == 'linear2':
        if parametrised_2qubit_gate: 
            nparams = (2*(nqubits-1)+1)*vqe_layers
        else: 
            nparams = nqubits*vqe_layers
            
    else: 
        nparams = nqubits
    
    return nparams 


def plot_simulated_convergence(res, experiment_series,ansatz, e0=None, title = ''):

    local_backend = Aer.get_backend('qasm_simulator')

    colors = []
    plt.figure(figsize=(7,5))    

    for learning_rate in res: 
        iterations = res[learning_rate]['total_iters']
        intermidiate_energies = []
        intermidiate_variances = []
        for intermidiate_params in res[learning_rate]['all_params']: 
            e, v = evaluate_params(intermidiate_params, ansatz, local_backend, experiment_series)
            intermidiate_energies.append(e)
            intermidiate_variances.append(v)
            

        p = plt.plot(range(1,iterations+1), intermidiate_energies, label='$\eta$ = ' + str(learning_rate))

        plt.fill_between(range(1,iterations+1),
                             np.array(intermidiate_energies) + np.sqrt(np.array(v)),
                             np.array(intermidiate_energies) - np.sqrt(np.array(v)), alpha = 0.3)
        
        colors.append(p[0].get_color())
        plt.scatter(range(1,iterations+1), intermidiate_energies, color = colors[-1] )
            

    #plt.axhline(y=0, color='k', ls = '--', )
    if e0 != None: 
        plt.axhline(y=e0, color='grey', ls = '--', )
    plt.xlim(0,iterations)
    #plt.xlim(10,25)
    #plt.ylim(-1, 200)
    plt.xlabel('iteration')
    plt.ylabel('energy')
    plt.title(title)
    plt.legend(loc=1)
    plt.savefig('convergence')
    plt.show()

    



                                    
    
def plot_convergence(results, e0 = None,labels=[], title = '', path_save = None):
    colors = []
    plt.figure(figsize=(7,5))    

    iteration =0
    
    counter = 1
    for result in results:
        
        iterations = result['total_iters']
        if iterations > iteration: iteration=iterations

        convergence_energy = result['energies'] 
        error = result['errors']
        label=labels[counter-1]
        p = plt.plot(range(1,iterations+1), convergence_energy, label=label)

        plt.fill_between(range(1,iterations+1),
                            np.array(convergence_energy) + np.array(error),
                            np.array(convergence_energy) - np.array(error), alpha = 0.3)
        colors.append(p[0].get_color())
        plt.scatter(range(1,iterations+1), convergence_energy, color = colors[-1] )
        counter+=1

    #plt.axhline(y=0, color='k', ls = '--', )
    if e0 != None: 
        plt.axhline(y=e0, color='grey', ls = '--', )
    plt.xlim(0,iteration)
    #plt.xlim(10,25)
    #plt.ylim(-1, 3)
    plt.xlabel('iteration')
    plt.ylabel('energy')
    plt.title(title)
    plt.legend(loc=1)
    if path_save != None : plt.savefig(path_save)
    plt.show()


def p(x,y):  ### runtime ready
    if x==y: 
        return 1
    else: 
        return 0

def qkt(x,y): ### runtime ready
    x_arr = bstr_to_arr(x)
    y_arr = bstr_to_arr(y)
    xy = list(zip(x_arr,y_arr))
    q=1
    for (xi,yi) in xy: 
        q *= p(xi,yi)
    return q


def get_Ry_ansatz(params,nqubits, vqe_layers, entanglement): 
    """
    Creates the qiskit circuit corresponding to the ansatz with the apropiate measurement

    Arguments: 

        nqubits: (int) , number of qubits of the circuit
        vqe_layers: (int)  , number of layers of depth of the VQE circuit
        params: (np.ndarray) of shape (p+1)*nqubits, parameters for the single-qubits Ry rotations
        entanglement: (str), type of nqubit unitary that will be used to generate entanglement on the circuit.
                      It can be: 
                      - 'linear': 

                                    --[Ry]--o---------[Ry]--
                                            |
                                    --[Ry]--x--o------[Ry]--
                                               |
                                    --[Ry]-----x--o---[Ry]--
                                                  |
                                    --[Ry]--------x---[Ry]--

                     - 'circular': 

                                    --[Ry]--o--------x---[Ry]--
                                            |        |
                                    --[Ry]--x--o-----|---[Ry]--
                                               |     |
                                    --[Ry]-----x--o--|---[Ry]--
                                                  |  |
                                    --[Ry]--------x--o---[Ry]--

                     - 'full' : 

                                    --[Ry]--o--o--o------------[Ry]--
                                            |  |  |
                                    --[Ry]--x--|--|--o--o------[Ry]--
                                               |  |  |  |
                                    --[Ry]-----x--|--x--|--o---[Ry]-- 
                                                  |     |  |
                                    --[Ry]--------x-----x--x---[Ry]-- 

                    - 'linear_crossed' : 

                                    --[Ry]--o------[Ry]--
                                            |
                                    --[Ry]--x--o---[Ry]--
                                               |
                                    --[Ry]--o--x---[Ry]--
                                            |     
                                    --[Ry]--x------[Ry]--

                    - 'circular_crossed' : 

                                    --[Ry]--o-----x---[Ry]--
                                            |     |
                                    --[Ry]--x--o--|---[Ry]--
                                               |  |
                                    --[Ry]--o--x--|---[Ry]--
                                            |     |
                                    --[Ry]--x-----o---[Ry]--

    (the circuits shown represent the whole circuit generated by each type of entanglement, not just the entangling gates, [Ry] is a 1-qubit Ry parametrised rotation,  o  is the control and   x   is the not ). 
    """
    qc_ansatz = QuantumCircuit(nqubits)
    #qc_ansatz.h(range(nqubits))
    for i in range(nqubits):
        qc_ansatz.ry(params[i],i)

    qc_ansatz.barrier()
    for layer in range(vqe_layers):
        if entanglement == 'linear': 
            for i in range(nqubits-1):
                qc_ansatz.cx(i,i+1)
        if entanglement == 'full': 
            for i in range(nqubits-1): 
                for j in range(nqubits): 
                    if i<j: qc_ansatz.cx(i,j)

        if entanglement == 'circular':
            for i in range(nqubits):
                qc_ansatz.cx(i,(i+1)%nqubits)

        if entanglement == 'linear_crossed': 
            even_qubits = [q for q in range(nqubits) if q%2==0]
            odd_qubits = [q for q in range(nqubits) if q%2==1]

            if nqubits%2==0: 
                odd_qubits.pop(-1)
            else: 
                even_qubits.pop(-1)

            for q in even_qubits: 
                qc_ansatz.cx(q,q+1)
            for q in odd_qubits:
                qc_ansatz.cx(q,q+1)

        if entanglement == 'circular_crossed': 
            even_qubits = [q for q in range(nqubits) if q%2==0]
            odd_qubits = [q for q in range(nqubits) if q%2==1]

            for q in even_qubits: 
                qc_ansatz.cx(q,(q+1)%nqubits)
            for q in odd_qubits:
                qc_ansatz.cx(q,(q+1)%nqubits)


        qc_ansatz.barrier()
        for i in range(nqubits):
            qc_ansatz.ry(params[int((layer+1)*nqubits+i)],i)

    qc_ansatz.measure_all()
    return qc_ansatz
    

def direct_parameter_visualisation(all_params_list, rel_bits = None): 

    all_params_arr = np.array(all_params_list)
    nb_params = len(all_params_list[0])
    total_iters = len(all_params_list)
    
    plt.figure(figsize=(12, 7))
    for param in range(nb_params):
        if rel_bits !=None:
            if param in rel_bits:
                rel_color = 'r'
            else: 
                rel_color = 'grey'
            plt.plot(range(total_iters), all_params_arr[:,param], label=f'param:{param}', color = rel_color)
        else: 
            plt.plot(range(total_iters), all_params_arr[:,param], label=f'param:{param}')
        
    plt.axhline(y=2*np.pi, color='grey', ls = '--', )
    plt.axhline(y=np.pi, color='grey', ls = '--', )
    plt.axhline(y=np.pi/2, color='grey', ls = '--', )
    plt.axhline(y=0., color='grey', ls = '--', )
    plt.legend(loc='best')
    plt.xlim(0, total_iters+5)
    plt.xlabel('iteration')
    plt.ylabel('param. value')
    


def hobo_to_qubo_bs_wrapper(hobo_bs, experiment_series): ### runtime ready
    """
    Converts bitstring from memory efficient encoding into QUBO problem encoding
    
    Arguments: 
    - hobo_bs : bitstring in the memory efficient encoding
    - Q : 
    - offset : 
    """

    id1 = 0
    id2 = 0
    g = ''
    for j in experiment_series.data.O_j:
        for o in experiment_series.data.O_j[j]:
            MT = experiment_series.active_experiment.pruning_data.MT_post_pruning[(j, o)]
            log = np.log2(MT)
            if int(log) == 0: k = 1
            elif  (log%int(log))==0 : k = int(log)
            else: k= int(log+1) 

            id2 += int(k)
            relevant_bs = hobo_bs[id1:id2]        
            id1 = id2
            gio=''
            for mt in range(MT):
                binkt = format( mt,'b').zfill(k)            
                gio+=str(qkt(   binkt  , relevant_bs ))
            g+=gio
    
    pruned_list = list(bstr_to_arr(g))
    indices = experiment_series.active_experiment.pruning_data.orig_index.values()
    all_bits = list(np.ones(experiment_series.data.nb_qubits)*(-1))

    for index in  indices: 
        all_bits[index] = 0

    for index in range(len(all_bits)):
        bit = all_bits[index]
        if bit==-1:
            all_bits[index] = int(pruned_list.pop(0))
            
    final_bs = ''
    for bit in all_bits:    
        final_bs+=str(bit)
    
    return final_bs
    
    

def hobo_to_unpruned_qubo_counts(counts, experiment_series):
    """
    Converts counts from memory efficient encoding into QUBO problem encoding
    """
    
    qubo_counts = dict()
    for (hobo_bs, count) in counts.items(): 
        qubo_bs =  hobo_to_qubo_bs_wrapper(hobo_bs, experiment_series)
        qubo_counts[qubo_bs] = count
            
    return qubo_counts


def get_counts(backend, circuit):
    """
    Runs a circuit on a given backend and returns counts
    
    """
    #trans_qc = transpile(circuit, backend)
    job =  backend.run(circuit)
    counts =job.result().get_counts()
    
    return counts

def sample_mean_estimate(function, counts,  Q, offset): 
    """
    Evaluates the estimate of the mean from a sample 
    
    Arguments: 
    - function : 
    - counts : 
    -  Q  : 
    - offset: 
    
    """
    return sum([function(evaluate_wrapper(key, Q, offset))*value 
                for key,value in list(counts.items())])/sum(counts.values())



def evaluate_params(params, ansatz, backend, experiment_series): 
    qc_binded = ansatz.bind_parameters(params)
    counts= get_counts(backend, qc_binded)
    
    counts_qubo = hobo_to_unpruned_qubo_counts(counts, experiment_series)
    Q = experiment_series.active_experiment.Q 
    offset = experiment_series.active_experiment.offset
#     energies = []    

#     shots = 0
#     for bs ,count in counts_qubo.items(): 
#         e = evaluate_wrapper(bs, experiment_series.active_experiment.Q , experiment_series.active_experiment.offset)
#         energies.append(e*count)
#         shots+= count
    energies = sample_mean_estimate(lambda x : x , counts_qubo, Q, offset)
    variances = sample_mean_estimate(lambda x : pow(x - energies,2.),counts_qubo, Q,offset)
    
    return energies, variances

def evaluate_wrapper(x,  Q, offset): ### runtime ready
    """
    Given a bitstring, an n x n QUBO matrix and the offset,
    evaluates the cost function x^T Q x + offset.

    Arguments:
    - x: (str) bitstring, as a python string of '0' and '1',
        of length n
    - Q: (np.ndarray) of shape (n, n)
    - offset : (float) et
    """
    
    x = np.array(list(x), dtype=np.uint8)
    return float(x @ Q @ x + offset)


def sample_mean_estimate(function, counts,  Q, offset): 
    """
    Evaluates the estimate of the mean from a sample 
    
    Arguments: 
    - function : 
    - counts : 
    -  Q  : 
    - offset: 
    
    """
    return sum([function(evaluate_wrapper(key, Q, offset))*value 
                for key,value in list(counts.items())])/sum(counts.values())


def sort_counts(counts):
    return sorted([(k,v) for k,v in counts.items()], key=lambda x: x[1], reverse=True)

def get_top_counts(counts, number_of_keys):
    sorted_counts = sort_counts(counts)
    return {k: v for k,v in sorted_counts[0:min(len(sorted_counts),number_of_keys)]}

def top_counts(number_of_bitstrings, results, experiment_series):
    """
    Returns the counts for the top <number_of_bitstrings> obtained from running the final parameters from the results.
    """
    
    all_top_counts = dict()
    for learning_rate in results: 
        if learning_rate != 'Hyper_params':
            result = results[learning_rate]
            iterations = result['total_iters']
            #convergence_energy = result['energies'] 
            #error = result['errors']
            #final_params = result['final_params']

            #qc_res = ansatz.bind_parameters(final_params)

            #ansatz(final_params, nqubits=nqubits, vqe_layers=vqe_layers,entanglement= entanglement)
            #counts_hobo = get_counts(backend, qc_res)
            counts_hobo = result['all_counts'][-1]
            counts = hobo_to_unpruned_qubo_counts(counts_hobo,experiment_series)

            #result['final_counts'] = counts
            #E_estimate = sample_mean_estimate(lambda x: x, counts, experiment_series.active_experiment.Q ,experiment_series.active_experiment.offset)
            all_top_counts[learning_rate]=get_top_counts(counts,number_of_bitstrings)

    return all_top_counts





# def get_results(backend,
#                 optimal_params,
#                 nqubits,
#                 iterations, 
#                 tolerance,
#                 p, 
#                 entanglement,
#                 shots): 

#     backend.shots = shots
#     qc_res = get_Ry_ansatz(optimal_params, nqubits, p, entanglement= entanglement)
#     trans_qc = transpile(qc_res, backend)
#     job =  backend.run(trans_qc)
#     counts =job.result().get_counts()
    
#     print('Backend name :  ',backend.name())
#     print('Entanglement:    ', entanglement)
#     print('\n')
        
    
#     print('Designed Circuit')
#     print(qc_res)
        
#     print('Transpiled Circuit')
#     print(trans_qc)

    
#     return {'job':job, 'counts':counts ,
#             'histogram' :plot_histogram(counts, title='Histogram of counts with optimal params', color='#648FFF'),
#             'circuit_plot':qc_res.draw('mpl'),
#             'trans_circuit_plot':trans_qc.draw('mpl')}


def bstr_to_arr(bstr):
    arr = np.zeros(len(bstr))
    for _ in range(len(bstr)): 
        arr[_] = int(bstr[_])
    return arr

def is_feasible(bs,experiment_series , constraints={'assignment':True, 'machine': True, 'order': True}):
    
    data = experiment_series.data

    
    index_to_JOMT = {v: k for k, v in data.jomt_to_index.items()}
    JOMT_to_bool = {k: int(bs[data.jomt_to_index[k]])
                    for k in data.jomt_to_index.keys()}

    VALS = np.array([val for val in JOMT_to_bool.values()])
    relevant_indices =  np.where(VALS==1)[0]
    bits_to_indices = np.array([(j,o,m,t) for (j,o,m,t) in JOMT_to_bool.keys()])
    relevant_JOMT = bits_to_indices[relevant_indices]
    relevant_set = set((j,o,m,t) for j,o,m,t in relevant_JOMT)

    relevant_relevant = set(itertools.product(relevant_set, relevant_set))
    
    feas = True
    
    #assignment constraint
    if constraints['assignment']: 
        for j, o in data.JO:
            feas *= (sum(JOMT_to_bool[j, o, m, t]
                     for m in data.M for t in data.T) == 1)
            
        assignment_feas = feas
            
    feas = True
    #machine constraint
    if constraints['machine']: 
        for ((j,o,m,t),(jj,oo,mm,tt)) in relevant_relevant: 
            if (m == mm and (j,o)!=(jj,oo)):
                if (t<=tt and tt<t+data.djom[j][o][m]): 
                    feas *= 0
        machine_feas = feas
        
    
    feas = True
    #order constraint
    if constraints['order']: 
        for ((j,o,m,t),(jj,oo,mm,tt)) in relevant_relevant: 
            if j==jj and o<oo: 
                feas *= (t+data.djom[j][o][m] <= tt )
                
        order_feas = feas
    
    #print(int(assignment_feas), int(machine_feas),int(order_feas))
    
    return assignment_feas*machine_feas*order_feas



def show_count_feasability(counts, data_dict): 
    sorted_counts = sort_counts(counts)
    for bs,count in sorted_counts:
    
        feas = is_feasible(bs, data_dict)
        if not feas:
            print(bs, count, '   ', Fore.RED + str(bool(feas)))
            print(Style.RESET_ALL)
        else: 
            print(bs,count, '   ', str(bool(feas)))
    
    return 1


def histogram_wrapper(figure, experiment_series, E0):
    axes = figure.get_axes()[0]
    labels = axes.xaxis.get_ticklabels()
    patches = axes.patches
    
    new_labels = []
    counter = 0
    for label, patch in zip(labels, patches):
        
        text = label.get_text()
        color = color_picker(text, experiment_series, E0)
        
        if len(str(text))>10: 
            new_labels.append(str(int(text, 2)))
        patch.set_facecolor(color)
        
        if color=='blue' or color=='green': 
            label.set_color(color)
            axes.annotate(str(np.round(evaluate_wrapper(str(text),experiment_series.active_experiment.Q,experiment_series.active_experiment.offset),3)), (counter,0.4), color='w', rotation=85)
        else: 
            label.set_color(color)
        counter+=1
    axes.tick_params(axis='x', labelrotation=88)
    if len(str(labels[0]))>10: 
        axes.set_xticklabels(new_labels)
    #axes.set_xticks(labels, rotation = 90) 
    
    return figure

def color_picker(text, experiment_series, E0):
    feas = is_feasible(text, experiment_series)
    if not feas: 
        return 'red'
    else: 
        if evaluate_wrapper(text, experiment_series.active_experiment.Q,experiment_series.active_experiment.offset)>E0: 
            return 'blue'
    return 'green'



    
def data_frame_bs(bs: str, experiment_series, save_as: str = None):
    data = experiment_series.data
    
    schedule = []
    JOMT_usage = {jomt: bool(int(bs[index]))
                  for index, jomt in enumerate(data.jomt)}
    for j in data.J:
        job_jobs = []
        job_starts = []
        job_ends = []
        job_durations = []
        job_operations = []
        job_machines = []

        for o in data.O_j[j]:
            for m in data.M:
                for t in data.T:
                    if JOMT_usage[(j, o, m, t)]:
                        job_operations.append(o)
                        job_machines.append(m)
                        job_starts.append(float(t))
                        job_ends.append(t + data.duration[j, o, m])
                        job_durations.append(data.duration[j, o, m])
                        job_jobs.append(j)

        job_schedule = pd.DataFrame()
        job_schedule['job'] = job_jobs
        job_schedule['operation'] = job_operations
        job_schedule['machine'] = job_machines
        job_schedule['start'] = job_starts
        job_schedule['end'] = job_ends
        job_schedule['duration'] = job_durations

        schedule.append(job_schedule)

        if save_as is not None:
            with open(save_as, 'a') as file:
                for df in schedule:
                    df.to_csv(file, sep=', ')

    return schedule

def plot_bs(bs: str, experiment_series, save_as: str = None):
    """
    plots the gant chart corresponding to the given binary string belonging to data
    """
    data = experiment_series.data
    df_list = data_frame_bs(bs, experiment_series)
    fig = go.Figure()
    for j in range(len(df_list)):
        df = df_list[j]
        fig.add_bar(
            x=df['duration'], base=df['start'], y=df['machine'],
            orientation='h', name=f'Job {j + 1}',
            hovertemplate=f'Job {j}' + '<br>' +
            '(%{base},%{x})<extra>%{y}</extra>'
        )
    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            tickmode='array',
            tickvals=[t+1 for t in data.T],
            ticktext=[x+1 if (x != max(data.T)) else f'{x+1}=T_max' for x in data.T]),
        xaxis_range=[0, max(data.T)+1],
        yaxis=dict(
            tickmode='array',
            tickvals=[m for m in data.M],
            ticktext=[f'machine {m}' for m in data.M])
    )
    fig.show()

    if not(save_as == None):
        fig.write_image(f"images/{save_as}.svg")
        fig.write_html(f"images/{save_as}.svg")

        
def build_spectrum(energy_set): 
    h_counts, bins, bars = plt.hist(np.array([v for (k,v) in energy_set]), bins=30) 

    return bins, h_counts

def low_energy_spectrum(energy_set, nb_energies):
    all_sorted_energies= sorted(energy_set , key= lambda x : x[1], reverse=False)
    
    low_energies = dict()
    counter =0
    for bs, e in all_sorted_energies:     
        if not str(e) in low_energies:
            if counter< nb_energies: 
                low_energies[str(e)] = 1
                counter +=1
            else: 
                break
        else: 
            low_energies[str(e)] +=1
    
    small_energy_bitstrings = sum(low_energies.values())
    

    small_sorted_energies = []
    for _ in range(small_energy_bitstrings):
        small_sorted_energies.append(all_sorted_energies[_])
    
    
    
    return build_spectrum(small_sorted_energies)
    
    
def plot_feasible_schedules(res, experiment_series, res_top_counts, save_as=None): 
    for lr in res:
        if lr != 'Hyper_params':
            print(lr)
            for bs in res_top_counts[lr]:
                if is_feasible(bs, experiment_series):
                    print(np.round(evaluate_wrapper(bs, experiment_series.active_experiment.Q, experiment_series.active_experiment.offset),3), bs,res_top_counts[lr][bs], int(bs,2) )
                    plot_bs(bs, experiment_series, save_as = save_as)


def get_linear_ansatz(nqubits, vqe_layers, entanglement, onequbit_gate = 'ry', twoqubit_gate ='cx', constant_depth = True , measure = True): 
    
    nparams = get_nparams(nqubits, vqe_layers, twoqubit_gate, entanglement )
    
    params = ParameterVector('θ', nparams )
    
    qc_ansatz = QuantumCircuit(nqubits)
    
    p=0
    
    if entanglement == None: 
        for i in range(nqubits):
            qc_ansatz = apply_one_qubit_gate(qc_ansatz,i, onequbit_gate, params[p])
            p+=1
        if measure: 
            qc_ansatz.measure_all()
            
        return qc_ansatz
        
    
    if entanglement == 'linear0': 
        for i in range(nqubits):
            qc_ansatz = apply_one_qubit_gate(qc_ansatz,i, onequbit_gate, params[p])
            p+=1
        qc_ansatz.barrier()
        
    
    for layer in range(vqe_layers):
        if entanglement == 'linear0': 
            if not constant_depth: 
                for i in range(nqubits-1):
                    if twoqubit_gate=='cx':
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate)
                    else: 
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate, params[p])
                        p+=1
                for i in range(nqubits):
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i, onequbit_gate, params[p])
                    p+=1
                        
            else: 
                even_qubits = [q for q in range(nqubits) if q%2==0]
                odd_qubits = [q for q in range(nqubits) if q%2==1]

                if nqubits%2==0: 
                    odd_qubits.pop(-1)
                else: 
                    even_qubits.pop(-1)

                for i in even_qubits:
                    if twoqubit_gate=='cx':
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate)
                    else: 
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate, params[p])
                        p+=1
                        
                for i in odd_qubits:
                    if twoqubit_gate=='cx':
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate)
                    else: 
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate, params[p])
                        p+=1
                for i in range(nqubits):
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i, onequbit_gate, params[p])
                    p+=1
                        
        if entanglement == 'linear1': 
            if not constant_depth: 
                for i in range(nqubits-1):
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i, onequbit_gate, params[p])
                    p+=1
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i+1, onequbit_gate, params[p])
                    p+=1
                    if twoqubit_gate=='cx':
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate)
                    else: 
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate, params[p])
                        p+=1
                        
            else: 
                even_qubits = [q for q in range(nqubits) if q%2==0]
                odd_qubits = [q for q in range(nqubits) if q%2==1]

                if nqubits%2==0: 
                    odd_qubits.pop(-1)
                else: 
                    even_qubits.pop(-1)

                for i in even_qubits:
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i, onequbit_gate, params[p])
                    p+=1
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i+1, onequbit_gate, params[p])
                    p+=1
                    if twoqubit_gate=='cx':
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate)
                    else: 
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate, params[p])
                        p+=1
                for i in odd_qubits:
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i, onequbit_gate, params[p])
                    p+=1
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i+1, onequbit_gate, params[p])
                    p+=1
                    if twoqubit_gate=='cx':
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate)
                    else: 
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate, params[p])
                        p+=1
                        
        if entanglement == 'linear2':
            if not constant_depth: 
                for i in range(nqubits-1):
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i, onequbit_gate, params[p])
                    p+=1
                    if twoqubit_gate=='cx':
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate)
                    else: 
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate, params[p])
                        p+=1
                        
            else: 
                even_qubits = [q for q in range(nqubits) if q%2==0]
                odd_qubits = [q for q in range(nqubits) if q%2==1]

                if nqubits%2==0: 
                    odd_qubits.pop(-1)
                else: 
                    even_qubits.pop(-1)

                for i in even_qubits:
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i, onequbit_gate, params[p])
                    p+=1
                    if twoqubit_gate=='cx':
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate)
                    else: 
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate, params[p])
                        p+=1
                for i in odd_qubits:
                    qc_ansatz = apply_one_qubit_gate(qc_ansatz,i, onequbit_gate, params[p])
                    p+=1
                    if twoqubit_gate=='cx':
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate)
                    else: 
                        qc_ansatz = apply_two_qubit_gate(qc_ansatz, i, i+1, twoqubit_gate, params[p])
                        p+=1
            qc_ansatz = apply_one_qubit_gate(qc_ansatz,nqubits-1, onequbit_gate, params[p])
            p+=1
        qc_ansatz.barrier()

    
    if measure : 
        qc_ansatz.measure_all()
        
    return qc_ansatz
        
    
    
def makespan(bs, experiment_series): 

    df_list = data_frame_bs(bs, experiment_series)

    job_finish_time = []
    for j in range(len(df_list)):
        job_finish_time.append(df_list[j].tail(1)['end'].values[0])

    return max(job_finish_time)

def get_plots(all_res, experiment_series, nqubits, file = 'results'):
    res_dir = os.path.join(os.getcwd(), file)

    for entanglement in all_res: 
        for twoqubit_gate in all_res[entanglement]: 
            for constant_depth in all_res[entanglement][twoqubit_gate]:

                figs_dir = os.path.join(res_dir, str(entanglement)+'_'+twoqubit_gate+'_'+str(constant_depth))
                
                for vqe_layers in all_res[entanglement][twoqubit_gate][constant_depth]:
                    
                    if constant_depth == 'True': cts_d = True
                    else: cts_d = False
                    vqe_layers = int(vqe_layers)

                    ansatz_res = get_linear_ansatz(nqubits, vqe_layers, entanglement, onequbit_gate='ry', twoqubit_gate=twoqubit_gate, constant_depth=cts_d, measure=True)
                    ansatz_res.draw('mpl').savefig('results/circuit_'+twoqubit_gate+'_'+str(cts_d)+'_L'+str(vqe_layers))


                    res = all_res[entanglement][twoqubit_gate][str(constant_depth)][vqe_layers]
                    
                    
                    con_plot_name = figs_dir+'_convergence_L='+str(vqe_layers)
                    plot_convergence(res, e0=0.5, title = 'QASM simulator L = '+str(vqe_layers), path_save = con_plot_name)
                    res_top_counts = top_counts(10, res, expser)

                    for k in range(len(learning_rates)):
                        lr = str(learning_rates[k])
                        hist_name = figs_dir+'_hist_L='+str(vqe_layers)+'_lr='+str(k)
                        histogram_wrapper(plot_histogram(res_top_counts[learning_rates[k]],bar_labels=True,title='lr = '+lr+'  L= '+str(vqe_layers)),  expser, 0.5).savefig(hist_name)
                        
                    feas_ratio(res, experiment_series,nqubits, save_as=figs_dir+'_feas_ratio_L='+str(vqe_layers))
                    plot_makespan(res, expser, save_as = figs_dir+'_makespan_L='+str(vqe_layers))
                    #plot_feasible_schedules(res, expser, res_top_counts, save_as=None)


def plot_makespan(res, experiment_series, save_as = None):
    
    colors = []
    for lr in res:
        if lr != 'Hyper_params':
            average_makespan = []
            best_makespan = []
            iters=0
            for counts in res[lr]['all_counts']: 
                counts = hobo_to_unpruned_qubo_counts(counts, experiment_series)
                ave = 0. 
                best = np.inf
                counted = 0
                for (bs,count) in counts.items(): 
                    if is_feasible(bs, experiment_series): 
                        t = makespan(bs, experiment_series)
                        ave += count * t

                        if t < best : 
                            best = t
                        #print(makespan(bs, expser))
                        #plot_bs(bs, expser)
                        counted += count
                
                if counted !=0: 
                    iters +=1
                    ave = ave / counted
                    average_makespan.append(ave)
                    best_makespan.append(best)

            p = plt.plot(range(31-iters, 31), average_makespan )
            colors.append(p[0].get_color())
            
            plt.scatter(range(31-iters,31), average_makespan, label = 'lr'+str(lr), color = colors[-1] )
            plt.plot(range(31-iters,31), best_makespan,ls='--', label = 'lr'+str(lr), color = colors[-1])
    
    plt.xlabel('iteration')
    plt.ylabel('Makespan')
    plt.axhline(y=3, color='grey', ls = '--', )
    plt.legend()
    if save_as != None: plt.savefig(save_as)
    plt.show()


def feas_ratio(res, experiment_series,nqubits, save_as=None):
               
    for lr in res: 
        
        if lr != 'Hyper_params':
            probabilities_sampling_feas = []
            for counts in res[lr]['all_counts']:
                
                #number_sampled_feas = []
                #probs_bin =[]


                #binded_circ = ansatz.bind_parameters(params)
                #counts = get_counts(backend_local, binded_circ)
                counts = hobo_to_unpruned_qubo_counts(counts,experiment_series)
                feas_ratio = feasible_solutions_sampled(counts, experiment_series)

                probabilities_sampling_feas.append(feas_ratio)
                #number_sampled_feas.append(number_sampled)


            res[lr]['Prob sampling feas'] = probabilities_sampling_feas
            #res[lr]['Number feas sampled'] = number_sampled_feas
            #res[lr]['Binomial prob sampling'] = probs_bin
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    
    
    #bound_prob_random_sampling = shots / pow(2,nqubits)
    
   
    c=0
    for lr in res: 
        if lr != 'Hyper_params':
            color = colors[c]
            c+=1
            plt.scatter(range(res[lr]['total_iters']),res[lr]['Prob sampling feas'], color=color, label ='lr='+str(lr))
            plt.plot(range(res[lr]['total_iters']), res[lr]['Prob sampling feas'], lw = 1,  color=color)
            #plt.plot(range(res[lr]['total_iters']), res[lr]['Binomial prob sampling'], lw = 1,ls = '--' , color=color)
    plt.axhline(y=1, color='grey', ls = '--', )
    plt.xlabel('iteration')
    plt.ylabel('Feas. ratio')
    plt.legend()
    
    if save_as != None : plt.savefig(save_as)
    
    plt.show()
    