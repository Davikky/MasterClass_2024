# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:26:37 2022

@author: EsteJalovecJ

"""


import numpy as np
#import time as time
import json
from json import JSONEncoder
import itertools


from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.circuit import ParameterVector, Parameter


def p(x,y):   
    if x==y: 
        return 1
    else: 
        return 0

def qkt(x,y):  
    x_arr = bstr_to_arr(x)
    y_arr = bstr_to_arr(y)
    xy = list(zip(x_arr,y_arr))
    q=1
    for (xi,yi) in xy: 
        q *= p(xi,yi)
    return q



def hobo_to_qubo_bs_wrapper(hobo_bs,  MT_postpruning, O_j):  
    """
    Converts bitstring from memory efficient encoding into QUBO problem encoding
    
    Arguments: 
    - hobo_bs : bitstring in the memoryx efficient encoding
    - Q : 
    - offset : 
    """

    id1 = 0
    id2 = 0
    g = ''
    for j in O_j:
        for o in O_j[j]:
            MT = MT_postpruning[(j, o)]
            k = int(np.ceil(np.log2(MT)))
            id2 += int(k)
            relevant_bs = hobo_bs[id1:id2]        
            id1 = id2
            gio=''
            for mt in range(MT):
                binkt = format( mt,'b').zfill(k)            
                gio+=str(qkt(   binkt  , relevant_bs ))
            g+=gio
    return g
    

def hobo_to_qubo_counts(counts, MT_postpruning, O_j):
    """
    Converts counts from memory efficient encoding into QUBO problem encoding
    """
    
    qubo_counts = dict()
    for (hobo_bs, count) in counts.items(): 
        qubo_bs =  hobo_to_qubo_bs_wrapper(hobo_bs, MT_postpruning, O_j)
        qubo_counts[qubo_bs] = count
            
    return qubo_counts

def get_counts(backend, circuit):
    """
    Runs a transpiled circuit on a given backend and returns counts
    
    """

    return backend.run(circuit).result().get_counts()



def inverse_exponential(E,tau=1.55):  
    return np.exp(-E*tau) 

def square_inverse_exponential(E,tau=1.55):  
    return np.exp(-2*E*tau) 

def inverse(E,tau=0.45):  
    return 1/pow(E,tau)


def evaluate_wrapper(x,  Q, offset):  
    """
    Given a bitstring, an n x n QUBO matrix and the offset,
    evaluates the cost function x^T Q x + offset.

    Arguments:
    - x: (str) bitstring, as a python string of '0' and '1',
        of length n
    - Q: (np.ndarray) of shape (n, n)
    - offset : (float) 
    """
    
    x = np.array(list(x), dtype=np.uint8)
    return float(x @ Q @ x) + offset

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

def bstr_to_arr(bstr):
    arr = np.zeros(len(bstr))
    for _ in range(len(bstr)): 
        arr[_] = int(bstr[_])
    return arr


def create_qubits(jomt):  
    qubit_dict = dict()
    for k in jomt:
        qubit_dict[k] = Binary('x_%i_%i_%i_%i' % k)
    return qubit_dict


# def get_nqubits_hobo(experiment_series):  
    
#     nb_tasks = experiment_series.data.nb_tasks
#     nb_machines = experiment_series.data.nb_machines
#     tmax =  experiment_series.data.tmax
    
#     log = int(np.ceil(np.log2(nb_machines*tmax)))

#     return nb_tasks*log

def get_nqubits_hobo_wrapper(experiment_series):  

    MT_post_pruning = experiment_series.active_experiment.pruning_data.MT_post_pruning
    new_nb_qubits = 0

 
    for (j,o) in MT_post_pruning.keys():
        log = int(np.ceil(np.log2(MT_post_pruning[(j, o)])))
        new_nb_qubits += log
        
    return new_nb_qubits


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
        #qc_ansatz.barrier()
        
    
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
        #qc_ansatz.barrier()

    
    if measure : 
        qc_ansatz.measure_all()
        
    return qc_ansatz
    



def one_optimisation_step(Q, offset, MT_postpruning, O_j ,old_params, backend, ansatz, learning_rate , nqubits, entanglement='linear', vqe_layers = 1):
    '''
    One update of all the parameters of the VQE ansatz by gradient descent in the direction that minimises the euclidean distance between the parametrised sampled state and the 'Filtered'
    state (see notebookon FVQE algorithm and references therein  for details)
    
    Arguments: 
    
        learning_rate: (float) , step size for parameter update  (usual values between 2 and 4). 
        old_params: (np.ndarray) of size number of paramters  
        ansatz: (qiskit.circuit) , parametrised circuit VQE ansatz
        backend: (qiksit.backend) 

        entanglement : (str)  ,  type of strucutre for 2-qubit unitaries that will be used in the ansatz
        vqe_layers : (int)  , number of layers of VQE circuit
        
    Returns : 
    
        new_params : ()  ,  new set of parameters for the VQE circuit
        H  : (float)  , estimate of energy with olf_params
        variance : (float) , variance of the energy estimate
    
    '''

    m = len(old_params)
    circuit_calls = 2*m+1
    
    circuit_theta = ansatz.bind_parameters(old_params)
    
    circuit_counts = get_counts(backend, circuit_theta)
    iter_counts = circuit_counts
    counts = hobo_to_qubo_counts(circuit_counts,MT_postpruning, O_j)

    f_function = inverse_exponential
    fsquare = square_inverse_exponential
    H = lambda x : x
    
    F2 = sample_mean_estimate(fsquare, counts, Q, offset)
    H = sample_mean_estimate(H, counts, Q, offset)
    variance = sample_mean_estimate(lambda x : pow(x - H,2.), counts, Q, offset)
    
    gradient = []
    for j in range(int((circuit_calls-1)/2)):
        ej = np.zeros(m)
        ej[j] = 1.
        theta_minus = old_params - np.pi/2*ej
        theta_plus = old_params + np.pi/2*ej
        
        circuit_theta = ansatz.bind_parameters(theta_plus)
        circuit_counts = get_counts(backend, circuit_theta)
        counts = hobo_to_qubo_counts(circuit_counts, MT_postpruning, O_j)
        
        C_j  = sample_mean_estimate(f_function, counts,  Q, offset)
        
        circuit_theta = ansatz.bind_parameters(theta_minus)
        circuit_counts = get_counts(backend, circuit_theta )

        counts = hobo_to_qubo_counts(circuit_counts, MT_postpruning, O_j)

        C_j -= sample_mean_estimate(f_function, counts, Q, offset)
        
        gradient.append(C_j)

    gradient = np.array(gradient)/(4*np.sqrt(F2))
    new_params = (old_params +learning_rate*gradient)

    return new_params, H , variance, iter_counts



def run_fvqe(nqubits, Q, offset, MT_postpruning, O_j, initial_params, backend, iterations, learning_rate, ansatz , tol= 0.001): 
    """
    Runs fvqe algorithms with specified hyper-parameters
    
    """
    
    total_iters=0
    params = np.copy(initial_params)

    convergence_energy = []
    variances = []
    errors = []
    E_current = 10000.
    
    all_params = []
    all_counts = []
    
    for t in range(iterations):
        total_iters +=1
        all_params.append(params)
        
        
        params, Et, s2, iter_counts = one_optimisation_step(Q, offset, MT_postpruning, O_j, params, backend, ansatz, learning_rate , nqubits)
        
        convergence_energy.append(Et)
        variances.append(s2)
        errors.append(np.sqrt(s2))
        all_counts.append(iter_counts)
        

        if abs(Et-E_current)< tol: 
            break
        E_current = Et
        # if (t)%5==0: 
        #     msg.publish('    iter:   ', t, '   ,   current E  :  ',  E_current)
    
    return convergence_energy, variances, errors, params,all_params, total_iters, all_counts




    

def run_vqe(backend,ansatz, experiment_series, initial_params, iterations = 30, shots = 500, tol = 0): 
    
    Q = experiment_series.active_experiment.pruning_data.Qpruned
    offset= experiment_series.active_experiment.offset
    MT_postpruning = experiment_series.active_experiment.pruning_data.MT_post_pruning

    O_j = experiment_series.data.O_j
    nqubits = Q.shape[0]

     
    backend.set_options(shots=shots)

    
    
#     ansatz = get_linear_ansatz(nqubits, vqe_layers, entanglement, onequbit_gate = onequbit_gate, twoqubit_gate =twoqubit_gate, constant_depth = constant_depth , measure = True)
#     ansatz = transpile(ansatz, backend,optimization_level=1)
    initial_params = np.ones(ansatz.num_parameters)*np.pi/2
    results = dict()

    convergence_energy, variances, errors, params,all_params, total_iters, all_counts = run_fvqe(nqubits, Q, offset, MT_postpruning,
                                                                                         O_j,initial_params, backend,
                                                                                         iterations,
                                                                                         2.5, 
                                                                                         ansatz = ansatz,
                                                                                         tol= tol)
        
    #t = time.time() - t0
    result = dict()
    result['energies'] = convergence_energy
    result['variances'] = variances
    result['errors'] = errors
    result['final_params'] = params
    result['all_params'] = all_params
    result['total_iters'] = total_iters
    #result['elapsed_time'] = t
    result['all_counts'] = all_counts

    

    #hyper_params['backend'] = backend.name
    #results['Hyper_params'] = hyper_params
        
    return result
