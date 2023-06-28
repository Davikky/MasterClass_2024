import numpy as np
import pandas as pd
import json
import jsonpickle
from json import JSONEncoder
import itertools
from collections import OrderedDict, namedtuple
from typing import Sequence, Union, Iterable, Callable
from abc import ABCMeta, abstractmethod, abstractproperty

from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit.opflow import OperatorBase
from qiskit import QuantumCircuit

#import igraph as ig

__all__ = ['SchedulingData', 'PruningData', 'FjsSchedulingData', 'Experiment', 'ExperimentSeries'] #'Result'

class SchedulingData(metaclass=ABCMeta):
    """abstract class for scheduling data. Used as base class
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def from_file(self):
        pass


class PruningData():
    def __init__(self) -> None:
        self._pruned_qubits = dict()
        self._orig_index = dict()
        self._reason_for_pruning = dict()
        self._nb_pruned_variables = 0
        self.MT_post_pruning = dict()
        self._Qpruned = None


    def prune_variable(self, j: int, o: int, m: int, t: int,
                       index: int, reason: str = 'unknown', value: int = 0) -> None:
        if (j, o, m, t) in self._pruned_qubits:
            raise ValueError("Different fixed values specified for the same qubit")
        self._pruned_qubits[j, o, m, t] = value
        self._orig_index[j, o, m, t] = index
        self._reason_for_pruning[j, o, m, t] = reason
        self._nb_pruned_variables += 1
        
    @property
    def Qpruned(self) -> np.matrix:
        """The Q property."""
        return self._Qpruned
    
    @Qpruned.setter
    def Qpruned(self, value: np.matrix):
        self._Qpruned = value

    @property
    def pruned_qubits(self):
        '''dictionary of pruned qubits in the format
        {(j,o,m,t): [pruned_value]}
        '''
        return self._pruned_qubits

    @pruned_qubits.setter
    def pruned_qubits(self, inp):
        raise NotImplementedError("Setter disallowed")

    @property
    def orig_index(self):
        return self._orig_index

    @orig_index.setter
    def orig_index(self, inp):
        raise NotImplementedError("Setter disallowed")

    @property
    def reason_for_pruning(self):
        return self._reason_for_pruning

    @reason_for_pruning.setter
    def reason_for_pruning(self, inp):
        raise NotImplementedError("Setter disallowed")

    @property
    def nb_pruned_variables(self):
        return self._nb_pruned_variables

    @nb_pruned_variables.setter
    def nb_pruned_variables(self, inp):
        raise NotImplementedError("Setter disallowed")
        


class FjsSchedulingData(SchedulingData):
    '''
        The format of the data files is as follows:
        First line: number of jobs, number of machines (+ average number of machines per operations, not needed)
        From the second line, for each job:
        Number of operations in that job
        For each operation:
        Number of machines compatible with this operation
        For each compatible machine: a pair of numbers (machine, processing time)

    methods:
        plot_solution(path[str]): (old) plot the solution and save it to "./images/[path]" both as .svg and .html
        from_file(path[str]): import data from textfile

    attributes:
        file: the text file from which the data was read
        filename[str]: name of the file from which the data was read
        solution[list[dict]]: (old) solution data
        nb_qubits[int]: the number of qubits needed calculated by |O|*|M|*|T|
        nb_jobs[int]: number of jobs
        nb_mashines[int]: number of mashines
        nb_operations[list[int]]: list of number of operations for each j
        nb_tasks[int]: total number of tasks
        nb_timesteps[int]: trivial maximal time for schedules
        tmax[int]: trvial maximal end time
        task_processing_time[np.array[2dim]]: 2 dimensional array for (task,mshine) assigning the processing time for each task on each mashine
        job_operation_task[np.array]: 2 dimensional array translating job j operation o to task n
        JO[set]: set of tuples of pairs of job and allowed operation indices depending on j
        J[set]: set of job indices
        M[set]: set of mashine indices
        T[set]: set of time indices
        O_j[set]: for each j a set of all operation indices
        O[set]: O_j
        djom[dict]: nested dictionary with keys for job, operation, mashine and values equalling the time for the corresponding task on given mashine
    '''

    def __init__(self, manual_tmax = None):
        self._filename = ''
        #self._nb_timesteps = None
        self._nb_machines = None
        self._nb_jobs = None
        self._nb_machines = None
        self._nb_operations = None
        self._task_processing_time = None
        self._job_operation_task = None
        self._jomt = None
        self._duration = None
        self._tmax = None
        # self._max_nb_timesteps = None
        self._minimum_anterior_times = None
        self._minimum_posterior_times = None
        self.manual_tmax = None
        if manual_tmax is not None: 
            self.manual_tmax = manual_tmax
        
        # if nb_timesteps is not None:
        #     self.nb_timesteps = nb_timesteps
    
    def __repr__(self):
        return "{} with {} jobs, {} operations (in total) and {} machines".format(
            self.__class__.__name__, self.nb_jobs, self.total_operations, self.nb_machines
            )
    
    def plot_solution(self, path: str = ''):
        """plots the solution and writes it to '.\\images\\[path]' if a path is given

        Args:
            path (str, optional): the path where the plot should be stored if empty string does not save the imgage. Defaults to ''.
        """
        raise NotImplementedError("Use the plotting functions in the visualization module")
        print('This is old. Use the plotting functions in the visualization module')
        fig = go.Figure()
        for j in range(len(self.solution)):
            df = self.solution[j]
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
                tickvals=[t+1 for t in self.T],
                ticktext=[x+1 if (x != max(self.T)) else f'{x+1}=tmax' for x in self.T]),
            xaxis_range=[0, max(self.T)+1],
            yaxis=dict(
                tickmode='array',
                tickvals=[m for m in self.M],
                ticktext=[f'machine {m}' for m in self.M])
        )
        fig.show()

        if not(path == ''):
            fig.write_image(f"images/{path}.svg")
            fig.write_html(f"images/{path}.svg")

    def from_file(self, filename: str, nb_timesteps: Union[int, None] = None) -> None:
        self._filename = filename
        with open(self._filename) as f:
            lines = f.readlines()
        string = '/'.join([el.strip() for el in lines])
        self.from_string(string, nb_timesteps = nb_timesteps)

    def from_string(self, string: str, nb_timesteps: Union[int, None] = None) -> None:
        """Imports the data given in a textfile in the format
            First line: number of jobs, number of machines (+ average number of machines per operations, not needed)
        From the second line, for each job:
        Number of operations in that job
        For each operation:
        Number of machines compatible with this operation
        For each compatible machine: a pair of numbers (machine, processing time)

        also generates various variables for later use

        Args:
            file (str): the plain text file from which the data should be read
        """

        #self.nb_timesteps = nb_timesteps
        self._string = string

        lines = string.split('/')
        first_line = lines[0].split()
        # Number of jobs
        self._nb_jobs = int(first_line[0])
        # Number of machines
        self._nb_machines = int(first_line[1])
        # Number of opertations for each job
        self._nb_operations = np.array([int(lines[j+1].split()[0]) for j in self.J])
        # Number of tasks
        self._nb_tasks = sum(self._nb_operations)
        # sets job_operation_task and task_processing_time
        self._set_task_data()

    def _set_task_data(self):
        # Processing time for each task, for each machine
        task_processing_time = np.full((self.nb_tasks, self.nb_machines), np.inf)
        # For each job, operation, define corresponding task id
        job_operation_task = np.array([[0 for o in self.O[j]] for j in self.J], dtype='object')
        id_ = 0
        lines = self._string.split('/')[1:]
        for j in self.J:
            line = lines[j].split()
            tmp = 0
            for o in self.O[j]:
                nb_machines_operation = int(line[tmp + o + 1])
                for i in range(nb_machines_operation):
                    # read data from the textfile lines
                    machine = int(line[tmp + o + 2*i + 2]) - 1
                    time = int(line[tmp + o + 2*i + 3])
                    task_processing_time[id_][machine] = time
                job_operation_task[j][o] = id_
                id_ += 1
                tmp += 2*nb_machines_operation
        self._task_processing_time = task_processing_time
        self._job_operation_task = job_operation_task

    # def _get_constraint_graph(self):
    #     '''Build the graph associated with the constraints of the FJSP. Usage mainly for the Alternating Operator Ansatz'''
    #     constraint_graph = ig.Graph(n=self.nb_qubits)
    #     constraint_graph.vs['i'] = [v.index for v in constraint_graph.vs]
    #     constraint_graph.vs['jomt'] = [self.jomt[v.index] for v in constraint_graph.vs]
    #     constraint_graph.vs['label'] = [str(jomt).replace(' ','') for jomt in constraint_graph.vs['jomt']]
    #     # go through all assignments and check if they are allowed at the same time
    #     for (jomt1, jomt2) in itertools.product(self.jomt, repeat=2):
    #         j1, o1, m1, t1 = jomt1
    #         j2, o2, m2, t2 = jomt2
    #         index1 = self.jomt_to_index[jomt1]
    #         index2 = self.jomt_to_index[jomt2]
    #         # assignment constraint
    #         if j1 == j2 and o1 == o2 and jomt1 != jomt2:
    #             constraint_graph.add_edge(index1, index2)
    #             continue
    #         # order constraint
    #         if j1 == j2 and o1 < o2 and t2 < t1+self.djom[j1][o1][m1]:
    #             constraint_graph.add_edge(index1, index2)
    #             continue
    #         # machine constraint
    #         if jomt1 != jomt2 and m1 == m2 and t1 <= t2 and t2 < t1+self.djom[j1][o1][m1]:
    #             constraint_graph.add_edge(index1, index2)
    #             continue
    #     return constraint_graph
    
    def _get_tmax(self):
        if self.manual_tmax is not None: 
            return int(self.manual_tmax)
        tmax=0
        for j in self.J: 
            for o in self.O_j[j]:
                aux_dict= {k:v for k,v in self.djom[j][o].items() if v!= np.inf}
                tmax += max([v for k,v in aux_dict.items()])
        
        return int(tmax)
        # machines_total_time = np.zeros(len(self.M))
        # for j in self.J: 
        #     for o in self.O_j[j]:
        #         aux_dict= {k:v for k,v in self.djom[j][o].items() if v!= np.inf}
        #         machine = max(aux_dict, key=aux_dict.get)
        #         machines_total_time[machine]+= self.djom[j][o][machine]
        # return int(np.max(machines_total_time))
        
        
        
#     def _get_max_nb_timesteps(self):
#         """maximal begining time
#         """

#         max_time = self.dummy_schedule_time()        
#         return max_time

    def _get_duration(self):
        dur = {}
        for j in self.J:
            for o in self.O[j]:
                for m in self.M:
                    dur[j, o, m] = self.task_processing_time[self._job_operation_task[j][o]][m]
        return dur

    def _get_jomt(self):
        jomt = []        
        for j in self.J:
            for o in self.O[j]:
                for m in self.M:
                    for t in self.T:
                        jomt.append((j, o, m, t))
        return jomt
    
    
    
    def _get_minimum_anterior_times(self):
        
        def minimal_predecessor_time(j, o):
            # this definition is recursive
            if o == 0:
                return 0.
            else:
                P_j_o = minimal_predecessor_time(j, o-1)+min([self.djom[j][o-1][m] for m in self.M])
                return(P_j_o)
        
        times = {}
        for j in self.J:
            for o in self.O_j[j]:
                times[(j, o)] = minimal_predecessor_time(j,o)
        return times

    def _get_minimum_posterior_times(self):
        times = {}
        for j in self.J:
            for o in self.O_j[j]:
                for m in self.M:
                    times[(j, o, m)] = self.djom[j][o][m]
                    for oo in self.O_j[j]:
                        if oo >o:
                            times[(j,o,m)] += min([ self.djom[j][oo][mm] for mm in self.M])

                   
        return times

    
    @property
    def duration(self) -> dict[tuple:int]:
        """Dictionary:
            keys: (j, o, m)
            values: duration of operation o of job j on machine m
        """
        if self._duration is None:
            self._duration = self._get_duration()
        return self._duration

    @property
    def jomt(self) -> list[tuple]:
        """Nested dictionaries of durations
        {j: {o: {m: d_jom, ...}, ...}, ...}
        where d_jom is the duration of operation o of job j on machine m
        """
        if self._jomt is None:
            self._jomt = self._get_jomt()
        return self._jomt

    @property
    def tmax(self):
        if self._tmax is None:
            self._tmax = self._get_tmax()
        return self._tmax

    @property
    def nb_qubits(self) -> int:
        return len(self.jomt_to_index)

    @property
    def string(self) -> str:
        return self._string

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, file: str):
        self._filename = file

    @property
    def nb_jobs(self) -> int:
        return self._nb_jobs

    @property
    def nb_machines(self) -> int:
        return self._nb_machines

    @property
    def nb_operations(self) -> int:
        return self._nb_operations

    @property
    def task_processing_time(self) -> np.ndarray:
        return self._task_processing_time

    @property
    def job_operation_task(self) -> np.ndarray:
        return self._job_operation_task

    @property
    def nb_tasks(self) -> int:
        return self._nb_tasks

#     @property
#     def nb_timesteps(self) -> int:
#         if self._nb_timesteps is None:
#             print("nb_timesteps not set. Fallback to max_nb_timesteps")
#             self._nb_timesteps = self._get_max_nb_timesteps()
#         return self._nb_timesteps

#     @nb_timesteps.setter
#     def nb_timesteps(self, input_: Union[int, None]) -> None:
#         if input_ is None:
#             return
#         if not isinstance(input_, int):
#             raise TypeError("nb_timesteps must be an int, not a {}".format(type(input_)))
#         if input_ > self.max_nb_timesteps:
#             raise ValueError("nb_timesteps must be lower or equal than max_nb_timesteps")
#         self._nb_timesteps = input_

#     @property
#     def max_nb_timesteps(self) -> int:
#         if self._max_nb_timesteps is None:
#             self._max_nb_timesteps = self._get_max_nb_timesteps()
#         return self._max_nb_timesteps

    @property
    def total_operations(self):
        return sum(self._nb_operations)

    @property
    def J(self) -> list:
        return list(range(self._nb_jobs))

    @property
    def M(self) -> list:
        return list(range(self._nb_machines))

    @property
    def T(self) -> list:
        return list(range(self.tmax))

    @property
    def O_j(self) -> dict[int, list[int]]:
        return {j: list(range(self._nb_operations[j])) for j in self.J}

    @property
    def O(self):
        return self.O_j

    @property
    def JO(self) -> list[tuple[int, int]]:
        return [(j, o) for j in self.J for o in self.O[j]]

    @property
    def index_to_jomt(self):
        return dict(enumerate(self.jomt))
    
    @property
    def jomt_to_index(self) -> dict[tuple, int]:
        return {v: k for k, v in self.index_to_jomt.items()}

    @property
    def djom(self) -> dict[int, dict[int, dict[int, int]]]:
        return  {j: {o: {m: self.task_processing_time[ self._job_operation_task[j][o]][m]
                     for m in self.M}
                 for o in self.O[j]}
             for j in self.J}

    @property
    def allowed_machines(self) -> dict[int, dict[int, list[int]]]:
        djom = self.djom
        return {j: {o: [m for m, t in djom[j][o].items() if t != np.inf] for o in djom[j].keys()} for j in djom.keys()}

    @property
    def max_durations(self) -> dict[int, dict[int, int]]:
        djom = self.djom
        return {j: {o: safe_max(djom[j][o].values()) for o in djom[j].keys()} for j in djom.keys()}

    @property
    def minimum_anterior_times(self) -> dict[int, dict[int, dict[int, int]]]:
        if self._minimum_anterior_times is None:
            self._minimum_anterior_times = self._get_minimum_anterior_times()
        return self._minimum_anterior_times

    @property
    def minimum_posterior_times(self) -> dict[int, dict[int, dict[int, int]]]:
        if self._minimum_posterior_times is None:
            self._minimum_posterior_times = self._get_minimum_posterior_times()
        return self._minimum_posterior_times

    @property
    def constraint_graph(self):
        """The constraint_graph property."""
        if self._constraint_graph is None:
            self._constraint_graph = self._get_constraint_graph()
        return self._constraint_graph

    @constraint_graph.setter
    def constraint_graph(self, value):
        return # forbidden





class Experiment():
    def __init__(self) -> None:
        self._qubits = None
        self._pruning_data = PruningData()
        self._circuit = None
        self._qubit_index_to_key = dict()
        self._qubit_key_to_index = dict()
        self._Q = None
        self._offset = None
        self._H = None
        self._Hm = None
        self._qubo = None
        # self._result = None
        # self._result_pruned = None
        self._solver = 'unknown'
        self._prob_correct = 0
        self._prob_feasible = 0
        self._avg_approximation_ratio = 0
        self._coeffs = dict(assignment_coeff=-1, order_coeff=-1,
            machine_coeff=-1, goal_coeff=-1)


    @property
    def coeffs(self) -> dict[str, float]:
        """The coeffs property."""
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value: Union[tuple[float, float, float, float], dict[str, float]]):
        if isinstance(value, tuple):
            coeff_dict = dict(assignment_coeff=value[0],
                              order_coeff=value[1],
                              machine_coeff=value[2],
                              goal_coeff=value[3])
        else:
            coeff_dict = value
        self._coeffs = coeff_dict

    # @property
    # def constraint_graph(self):
    #     """The constraint_graph property."""
    #     return self._constraint_graph

#     @constraint_graph.setter
#     def constraint_graph(self, value):
#         self._constraint_graph = value

    @property
    def qubits(self) -> dict:
        """The qubits property."""
        return self._qubits

    @qubits.setter
    def qubits(self, value: dict):
        self._qubits = value

    @property
    def pruning_data(self) -> PruningData:
        """The pruning_data property."""
        return self._pruning_data

    @pruning_data.setter
    def pruning_data(self, value: PruningData):
        self._pruning_data = value

    @property
    def circuit(self) -> QuantumCircuit:
        """The circuit property."""
        return self._circuit

    @circuit.setter
    def circuit(self, value: QuantumCircuit):
        self._circuit = value

    @property
    def qubit_index_to_key(self) -> dict[int, tuple[int, int, int, int]]:
        """The qubit_index_to_key property."""
        return self._qubit_index_to_key

    @qubit_index_to_key.setter
    def qubit_index_to_key(self, value: dict[int, tuple[int, int, int, int]]):
        self._qubit_index_to_key = value

    @property
    def qubit_key_to_index(self) -> dict[tuple[int, int, int, int], int]:
        """The qubit_key_to_index property."""
        return self._qubit_key_to_index

    @qubit_key_to_index.setter
    def qubit_key_to_index(self, value: dict[tuple[int, int, int, int], int]):
        self._qubit_key_to_index = value

    @property
    def Q(self) -> np.matrix:
        """The Q property."""
        return self._Q
    
    @Q.setter
    def Q(self, value: np.matrix):
        self._Q = value
        
    @property
    def offset(self) -> float:
        """The offset property."""
        return self._offset
    
    @offset.setter
    def offset(self, value: float):
        self._offset = value

    @property
    def H(self) -> OperatorBase:
        """The H property."""
        return self._H

    @H.setter
    def H(self, value: Union[Sequence, OperatorBase]):
        if isinstance(value, OperatorBase):
            self._H = value
        if isinstance(value, Sequence):
            self._H = value[0]

    @property
    def Hm(self) -> OperatorBase:
        """The Hm property."""
        return self._Hm

    @Hm.setter
    def Hm(self, value: OperatorBase):
        self._Hm = value

    @property
    def qubo(self) -> dict[tuple[str, str], float]:
        """The qubo property."""
        return self._qubo

    @qubo.setter
    def qubo(self, value: dict[tuple[str, str], float]):
        self._qubo = value

#     @property
#     def result(self) -> Result:
#         """The result property."""
#         return self._result

#     @result.setter
#     def result(self, value: Result):
#         self._result = value

#     @property
#     def result_pruned(self) -> Result:
#         """The result_pruned property."""
#         return self._result_pruned

#     @result_pruned.setter
#     def result_pruned(self, value: Result):
#         self._result_pruned = value
#         self.insert_pruned_bits()
#         self.result.source = value.source

#     @property
#     def solver(self) -> str:
#         """The solver property."""
#         return self._solver

#     @solver.setter
#     def solver(self, value: str):
#         self._solver = value

#     @property
#     def prob_correct(self) -> float:
#         """The prob_correct property."""
#         return self._prob_correct

#     @prob_correct.setter
#     def prob_correct(self, value: float):
#         self._prob_correct = value

#     @property
#     def prob_feasible(self) -> float:
#         """The prob_feasible property."""
#         return self._prob_feasible

#     @prob_feasible.setter
#     def prob_feasible(self, value: Union[int, float]):
#         self._prob_feasible = float(value)

#     @property
#     def avg_approximation_ratio(self) -> float:
#         """The avg_approximation_ratio property."""
#         return self._avg_approximation_ratio

#     @avg_approximation_ratio.setter
#     def avg_approximation_ratio(self, value: float):
#         self._avg_approximation_ratio = value


class ExperimentSeries():
    def __init__(self, data: FjsSchedulingData = None, name='ExperimentSeries00') -> None:
        if data is None:
            self._data = FjsSchedulingData()
        else:
            self._data = data
        # self._correct_solution_set = set()
        self._experiments = []
        self._active_experiment = Experiment()
        self._name = name

    
#     def compute_feasible_probs(self):
#         for e in self.experiments:
#             e.prob_feasible = prob_feasible(e, self.data)

#     def compute_correct_probs(self):
#         for e in self.experiments:
#             e.prob_correct = prob_correct(e, self.correct_solution_set)


    def deactivate_experiment(self):
        if self.active_experiment is not Experiment():
            self.experiments.append(self.active_experiment)
            self.active_experiment = Experiment()
        else:
            print('active experiment is empty, skipping deactivation')

    # def constraint_graph(self):
    #     return self.active_experiment.constraint_graph()

    @property
    def name(self):
        """The name property."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def data(self) -> FjsSchedulingData:
        """The data property."""
        return self._data

    @data.setter
    def data(self, value: FjsSchedulingData):
        self._data = value

#     @property
#     def correct_solution_set(self) -> set[str]:
#         """The correct_solution_set property."""
#         return self._correct_solution_set

#     @correct_solution_set.setter
#     def correct_solution_set(self, value: set[str]):
#         self._correct_solution_set = value
#         print(f'set correct solution set for {self.name}')
#         if self.experiments != [] and self.correct_solution_set != set():
#             self.compute_correct_probs()
#             print(f'... and calculated probabilities of correctness for alls experiments.')

    @property
    def experiments(self) -> list[Experiment]:
        """The experiments property."""
        return self._experiments

    @experiments.setter
    def experiments(self, value: list[Experiment]):
        self._experiments = value
        if self.experiments != [] and self.correct_solution_set != set():
            self.compute_correct_probs()

    @property
    def active_experiment(self) -> Experiment:
        """The active_experiment property."""
        return self._active_experiment

    @active_experiment.setter
    def active_experiment(self, value: Experiment):
        self._active_experiment = value

    ae = active_experiment




# def prob_feasible(sample: Union[Experiment, Result], data: FjsSchedulingData):
#     if isinstance(sample, Experiment):
#         probs = sample.result.probs
#     if isinstance(sample, Result):
#         probs = sample.probs

#     feas_prob = 0
#     for bs in probs:
#         if is_feasible(bs, data):
#             feas_prob += 1
#     return feas_prob


# def prob_correct(sample: Union[Experiment, Result], sol: set[str]):
#     """"For a given sample which is either an experiment or directly a result,
#      compute the probability of correct solltion given by sol
#      """
#     if isinstance(sample, Experiment):
#         res = sample.result
#     if isinstance(sample, Result):
#         res = sample
#     return sum([res.probs.get(s, 0) for s in sol])


# def avg_approx_ratio(corr_solutions: list[pd.DataFrame], counts: dict[str, int], goal_func: Callable[[str], float] = lambda x: float(int(x))):
#     N = sum(counts.values())
#     best_energy = min([goal_func(x) for x in corr_solutions])
#     avg_approx_ratio = 0
#     for x, c in counts.items():
#         approx_ratio = goal_func(x)/best_energy
#         avg_approx_ratio += approx_ratio*c
#     return avg_approx_ratio/N


def insert_pruned_bits(bs: str, p_data: PruningData) -> str:
    nb_all_qubits = len(bs) + p_data.nb_pruned_variables
    orig_index_to_key = {v: k for k, v in p_data.orig_index.items()}
    bs_pointer = 0
    total_bs = ''
    for i in range(nb_all_qubits):
        if i in orig_index_to_key.keys():
            new_bit = str(p_data.pruned_qubits[orig_index_to_key[i]])
        else:
            new_bit = bs[bs_pointer]
            bs_pointer += 1
        total_bs += new_bit
    return total_bs


def prune_string(x: str, prun: PruningData):
    pruned_x = [int(item) for index, item in enumerate(
        x) if index not in prun.orig_index.values()]
    bs = ''
    for b in pruned_x:
        bs += str(b)
    return bs

def safe_max(sequence):
    return max(filter(lambda x: x != np.inf, sequence))

