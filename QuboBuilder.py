# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:26:37 2022

@author: EsteJalovecJ
"""

import numpy as np
import itertools
from pyqubo import Binary
from qiskit_optimization import QuadraticProgram
from FlexibleJobShop.schedulers import TimePruner, MachinePruner

def goal_function_energy2(bs, experiment_series): ### runtime ready
        """the goal function is to minimize the last finishing time. An additional weighting is used"""

        S = experiment_series.data.minimum_posterior_times
        P = experiment_series.data.minimum_anterior_times

        def Gamma(j, o, m):
            return experiment_series.data.tmax - S[(j, o, m)] - P[(j, o)]

        def penalty(j, o, m, t):
            
            if experiment_series.data.djom[j][o][m] == np.inf: 
                return 100000.
            else: 
                return (t + experiment_series.data.djom[j][o][m] - P[(j, o)]) / (Gamma(j, o, m)+1)

        p = penalty
        
        value =0.
        for index in range(experiment_series.data.nb_qubits): 
            b = bs[index]
            j,o,m,t = experiment_series.data.index_to_jomt[index]
            value += p(j,o,m,t)*int(b)
        
        return value

def goal_function_energy(bs, experiment_series): ### runtime ready
        """the goal function is to minimize the last finishing time. An additional weighting is used"""
    
        def simple_penalty(j,o,m,t): 
            if experiment_series.data.djom[j][o][m] == np.inf: 
                return 100000.
            else: 
                return (t + experiment_series.data.djom[j][o][m]) / experiment_series.data.tmax
    
        p=simple_penalty


        value =0.
        for index in range(experiment_series.data.nb_qubits): 
            b = bs[index]
            j,o,m,t = experiment_series.data.index_to_jomt[index]
            value += p(j,o,m,t)*int(b)

        return value

def build_dummy_schedule(experiment_series): 
    dummy_schedule = np.zeros(experiment_series.data.nb_qubits)
    tstep=0
    for j in experiment_series.data.J: 
        
        for o in experiment_series.data.O_j[j]:
            aux_dict= {k:v for k,v in experiment_series.data.djom[j][o].items() if v!= np.inf}
            machine = max(aux_dict, key=aux_dict.get)

            index = experiment_series.data.jomt_to_index[j,o,machine, tstep]
            dummy_schedule[index] = 1.
            tstep += experiment_series.data.djom[j][o][machine]
            
    dummy_bitstring_schedule = ''
    for b in dummy_schedule:
        dummy_bitstring_schedule+= str(int(b))
    
    return dummy_bitstring_schedule   

def dummy_schedule_goal_penalty(experiment_series):
    dummy_schedule = np.zeros(experiment_series.data.nb_qubits)
    tstep=0
    for j in experiment_series.data.J: 
        
        for o in experiment_series.data.O_j[j]:
            aux_dict= {k:v for k,v in experiment_series.data.djom[j][o].items() if v!= np.inf}
            machine = max(aux_dict, key=aux_dict.get)

            index = experiment_series.data.jomt_to_index[j,o,machine, tstep]
            dummy_schedule[index] = 1.
            tstep += experiment_series.data.djom[j][o][machine]
            
    dummy_bitstring_schedule = ''
    for b in dummy_schedule:
        dummy_bitstring_schedule+= str(int(b))
        
    
    
    gf_e = goal_function_energy(dummy_bitstring_schedule, experiment_series)
        
    return gf_e




def assignment_constraint(experiment_series): ### runtime ready
    return sum([(1-sum([experiment_series.active_experiment.qubits[j, o, m, t]
             for m in experiment_series.data.M for t in experiment_series.data.T]))**2
            for j, o in experiment_series.data.JO])

def order_constraint(experiment_series): ##runtime ready
        """the operations of each job need to be done in the right order"""
        # here and in the folowing I decied to not use many for loops but to define the index sets
        # in the end it does not make a difference I guess
        V_j = {j: set() for j in experiment_series.data.J}
        for j in V_j:
            # cartesian product of O M and T
            OMT = set(itertools.product(experiment_series.data.O_j[j], experiment_series.data.M, experiment_series.data.T))
            OMT_OMT = set(itertools.product(OMT, OMT))
            V_j[j] = {((o, m, t), (oo, mm, tt))
                      for ((o, m, t), (oo, mm, tt)) in OMT_OMT
                      if (o < oo and tt < t+experiment_series.data.djom[j][o][m])
                      }

        return sum([
            sum([experiment_series.active_experiment.qubits[j, o, m, t]*experiment_series.active_experiment.qubits[j, oo, mm, tt]
                 for ((o, m, t), (oo, mm, tt)) in V_j[j]
                 ])
            for j in experiment_series.data.J])

def machine_constraint(experiment_series): ##runtime ready
        """on each machine only one task can run at the time"""
        W_m = {m: set() for m in experiment_series.data.M}
        JOT = set(itertools.product(experiment_series.data.JO, experiment_series.data.T))
        # this is ugly and a work around, I needed the indices all on the same level
        JOT_new = set()
        for ((j, o), t) in JOT:
            JOT_new.add((j, o, t))
        JOT = JOT_new
        JOT_JOT = set(itertools.product(JOT, JOT))
            
        for m in experiment_series.data.M:
            W_m[m] = {((j, o, m, t), (jj, oo, m,tt))
                        for ((j, o, t), (jj, oo, tt)) in JOT_JOT
                        if (t <= tt and tt <t+experiment_series.data.djom[j][o][m] and (j,o)!= (jj,oo)) }

        return sum([experiment_series.active_experiment.qubits[j, o, m, t]*experiment_series.active_experiment.qubits[jj, oo, m, tt] 
                    # note that mm not being used is on purpose. They are just defined for better readability
                    for mmm in experiment_series.data.M
                    for ((j, o, m, t), (jj, oo, mm, tt)) in W_m[mmm]
                    ])
    
def goal_function(experiment_series):
    
    def simple_penalty(j,o,m,t): 
        if experiment_series.data.djom[j][o][m] == np.inf: 
            return 100000.
        else: 
            return (t + experiment_series.data.djom[j][o][m]) / experiment_series.data.tmax
    p=simple_penalty
    
    return sum([
            sum([
                sum([
                    sum([
                        p(j, o, m, t)*experiment_series.active_experiment.qubits[j, o, m, t]
                        for t in experiment_series.data.T
                    ])
                    for m in experiment_series.data.M ])  
                for o in experiment_series.data.O_j[j]])
            for j in experiment_series.data.J])
    
def goal_function2(experiment_series): ### runtime ready
        """the goal function is to minimize the last finishing time. An additional weighting is used"""

        S = experiment_series.data.minimum_posterior_times
        P = experiment_series.data.minimum_anterior_times

        def Gamma(j, o, m):
            g = experiment_series.data.tmax - S[(j, o, m)] - P[(j, o)]
            if g > 0 : 
                return g
            else: 
                return 
            return experiment_series.data.tmax - S[(j, o, m)] - P[(j, o)]

        def penalty(j, o, m, t):
            
            if experiment_series.data.djom[j][o][m] == np.inf: 
                return 100000.
            else: 
                return (t + experiment_series.data.djom[j][o][m] - P[(j, o)]) / (Gamma(j, o, m)+1)

        p = penalty

        goal_func = []
        for j in experiment_series.data.J: 
            for o in experiment_series.data.O_j[j]: 
                for m in experiment_series.data.M : 
                    if experiment_series.data.djom[j][o][m] != np.inf: 
                        
                        for t in range(int(P[(j,o)]), int(experiment_series.data.tmax -S[(j, o, m)]) ):
                            goal_func.append(p(j, o, m, t)*experiment_series.active_experiment.qubits[j, o, m, t])
                    else: 
                        for t in experiment_series.data.T: 
                            goal_func.append(10000.*experiment_series.active_experiment.qubits[j, o, m, t])
                
        
        return sum(goal_func)
        # return sum([
        #     sum([
        #         sum([
        #             sum([
        #                 p(j, o, m, t)*experiment_series.active_experiment.qubits[j, o, m, t]
        #                 for t in range(int(P[(j,o)]), int(experiment_series.data.tmax -S[(j, o, m)]-1) ) #experiment_series.data.T
        #             ])
        #             for m in experiment_series.data.M ])  
        #         for o in experiment_series.data.O_j[j]])
        #     for j in experiment_series.data.J])


def Hamiltonian(Ha, Hm, Ho, Hg, constraint_coefs ): 
    alpha, beta, gamma, delta = constraint_coefs
    print('Ha : ', alpha, '  Hm : ',beta,'  Ho : ', gamma,'  Hg : ', delta)
    return alpha*Ha+beta*Hm+gamma*Ho+delta*Hg


def create_qubits(jomt):
    qubit_dict = dict()
    for k in jomt:
        qubit_dict[k] = Binary('x_%i_%i_%i_%i' % k)
    return qubit_dict


def build_matrix(experiment_series, pruned=True, best_knwon_schedule = None): 
    
    if pruned: 
        MachinePruner.prune(experiment_series)
        TimePruner.prune(experiment_series)
    
    qubits = create_qubits(experiment_series.data.jomt)
    experiment_series.active_experiment.qubits = qubits
    
    if best_knwon_schedule is not None:
        dummiest_energy = goal_function_energy(best_knwon_schedule, experiment_series)
    else: 
        dummiest_energy = dummy_schedule_goal_penalty(experiment_series)
    
    print(dummiest_energy)
    constraint_coefs = [1., 1., 1., 1./dummiest_energy]
    
    Ha = assignment_constraint(experiment_series)
    Hm = machine_constraint(experiment_series)
    Ho = order_constraint(experiment_series)
    Hg = goal_function(experiment_series)
    
    H = Hamiltonian(Ha,Hm,Ho,Hg, constraint_coefs)
    
    model = H.compile()
    qubo, offset = model.to_qubo()
    


    used_qubits = dict()
    qubo_per_tuple = dict()
    for (k1, k2), val in qubo.items():
        
        k1n = k1.split('_')
        k2n = k2.split('_')
        k1n.pop(0)
        k2n.pop(0)
        k1 = tuple(int(s) for s in k1n)
        k2 = tuple(int(s) for s in k2n)

        # k1 = k1.replace('x', '')
        # k2 = k2.replace('x', '')
        # k1 = k1.replace('_', '')
        # k2 = k2.replace('_', '')
        # k1 = tuple(int(s) for s in k1)
        # k2 = tuple(int(s) for s in k2)
        
        qubo_per_tuple[(k1, k2)] = val
        k1_index = experiment_series.data.jomt.index(k1)
        k2_index = experiment_series.data.jomt.index(k2)
        used_qubits[k1_index] = k1
        used_qubits[k2_index] = k2

    used_qubits_indices_sorted = list(used_qubits.keys())
    used_qubits_indices_sorted.sort()

    used_keys_sorted = list()
    for i in used_qubits_indices_sorted:
        used_keys_sorted.append(experiment_series.data.jomt[i])
    
    Q = np.zeros((len(qubits), len(qubits)))
    for (k1, k2), val in qubo_per_tuple.items():
         Q[used_keys_sorted.index(k2),
                              used_keys_sorted.index(k1)] = val
        
#     qubo_per_qubit_index = dict()
#     for (k1, k2), val in qubo_per_tuple.items():
#         qubo_per_qubit_index[used_keys_sorted.index(k1),
#                              used_keys_sorted.index(k2)] = val

#     qprog = QuadraticProgram('FJS_qprog')
#     qprog.clear()  # just in case
#     qprog.binary_var_dict(
#         list(f'x{i}' for i in used_keys_sorted), name='')
    
#     qprog.minimize(quadratic=qubo_per_qubit_index)
#     Q = np.matrix(qprog.objective.quadratic.to_array(symmetric=False)) + np.diag(qprog.objective.linear.to_array())

    experiment_series.active_experiment.Q = Q
    experiment_series.active_experiment.offset = offset
    
    if pruned : 
        pruned_indices = np.array([index for index in experiment_series.active_experiment.pruning_data.orig_index.values()])
        Q_pruned = np.delete(experiment_series.active_experiment.Q, pruned_indices, 0)
        experiment_series.active_experiment.pruning_data.Qpruned = np.delete(Q_pruned, pruned_indices, 1)
    
#     return Q,offset

def profile_qubo(specs): 
    constraint_coefs, experiment_series = specs
    
    Q, off = build_matrix(constraint_coefs, experiment_series)
    
    
    
    
    
    