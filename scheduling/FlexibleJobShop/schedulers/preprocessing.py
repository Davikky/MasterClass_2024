from winreg import ExpandEnvironmentStrings
import pyqubo as pq
import numpy as np
from typing import *
from abc import abstractmethod

from FlexibleJobShop.core import Experiment, PruningData, FjsSchedulingData, ExperimentSeries
from .abstract import BasePreprocessing


class Pruner(BasePreprocessing):

    @classmethod
    def get_name(cls):
        return cls.__name__

    @classmethod
    def get_pruning_data(cls, experiment_series: ExperimentSeries, inplace: bool = True) -> PruningData:
        if not inplace:
            pruning_data = PruningData()
        else:
            pruning_data = experiment_series.active_experiment.pruning_data
        data = experiment_series.data
        n = 0
        nn = 0
        
        # for (j,o) in data.JO:
        #     experiment_series.active_experiment.pruning_data.MT_post_pruning.update({(j,o):0}) 
        
        for (j, o, m, t), index in data.jomt_to_index.items():
            if cls.pruning_criterion(j, o, m, t, data):
                n+=1
                pruning_data.prune_variable(j, o, m, t, index, reason=cls.get_name())
        
        experiment_series.active_experiment.pruning_data.MT_post_pruning = dict()
        for (j,o,m,t) in experiment_series.data.jomt:
            if  (j,o,m,t) not in experiment_series.active_experiment.pruning_data.pruned_qubits.keys():
                nn+=1
                if (j,o) in experiment_series.active_experiment.pruning_data.MT_post_pruning.keys(): 
                    experiment_series.active_experiment.pruning_data.MT_post_pruning[(j,o)] += 1
                else:
                    experiment_series.active_experiment.pruning_data.MT_post_pruning[(j,o)] =1 

    
        
        print(f'{cls.get_name()} left {nn} variables ')       
        print(f'{cls.get_name()} pruned {n} qubits')
        if not inplace:
            return pruning_data

    @classmethod
    def prune(cls, experiment_series: ExperimentSeries):
        # if not isinstance(experiment_series, ExperimentSeries):
        #     raise TypeError("Pruner must be called with an instance of {}, not {}".format(
        #         ExperimentSeries.__name__, type(experiment_series).__name__))
        cls.get_pruning_data(experiment_series, inplace=True)
    
    @classmethod
    @abstractmethod
    def pruning_criterion(cls, j, o, m, t, data: FjsSchedulingData) -> bool:
        pass


class TimePruner(Pruner):
    @classmethod
    def pruning_criterion(cls, j, o, m, t, data: FjsSchedulingData) -> bool:
        if data.duration[j, o, m] == np.inf:
            return False
        P = data.minimum_posterior_times
        A = data.minimum_anterior_times
        return t < A[j, o] or t > data.tmax - P[j, o, m]


class MachinePruner(Pruner):
    @classmethod
    def pruning_criterion(cls, j, o, m, t, data: FjsSchedulingData) -> bool:
        return data.duration[j, o, m] == np.inf
    
