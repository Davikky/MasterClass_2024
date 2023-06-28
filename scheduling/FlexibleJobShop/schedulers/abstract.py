from abc import ABCMeta, abstractmethod, abstractproperty

from FlexibleJobShop.core import FjsSchedulingData, ExperimentSeries #, Result


class BaseScheduler(metaclass=ABCMeta):
    @abstractmethod
    def solve(self, data: FjsSchedulingData, **kwargs):
        pass

    @abstractproperty
    def get_name(self):
        pass

class BasePreprocessing(metaclass=ABCMeta):
    @abstractmethod
    def preprocess(self, experiment_series: ExperimentSeries):
        pass

    @abstractproperty
    def get_name():
        return 'abstract_preprocessing'


# class BasePostprocessing(metaclass=ABCMeta):
#     @abstractmethod
#     def postprocess(self, experiment_series: ExperimentSeries, result: Result) -> str:
#         pass

#     @abstractproperty
#     def get_name():
#         return 'abstract_postprocessing'