import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class ExtrapMethod(Enum):
    
    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class Interpolator1D(ABC):

    def __init__(self,
                 axis1 : np.ndarray, 
                 values : np.ndarray, 
                 interpolation_method : InterpMethod,
                 extrpolation_method : ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrpolation_method
        self.length_ = len(self.axis1)

    @abstractmethod
    def interpolate(self, x : float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x : float, end_x : float):
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x : float):
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):
        pass
    
    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_
    
    @property
    def values(self) -> np.ndarray:
        return self.values_
    
    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()
    
    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()

class Interpolator1DPCP(Interpolator1D):

    def __init__(self, axis1: np.ndarray, values: np.ndarray, extrpolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.LINEAR, extrpolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        axis1 = np.asarray(self.axis1_)
        values = np.asarray(self.values_)

        # sort values based on axis1
        idx = np.argsort(axis1)
        sorted_axis1 = axis1[idx]
        sorted_values = values[idx]

        # flat extrapolation
        if x <= sorted_axis1[0]:
            return sorted_values[0]
        elif x >= sorted_axis1[-1]:
            return sorted_values[-1]
        
        # linear interpolation
        for i in range(len(sorted_axis1)-1):
            x0 = sorted_axis1[i]
            x1 = sorted_axis1[i+1]

            if x0 <= x < x1:
                y0 = sorted_values[i]
                y1 = sorted_values[i+1]
                if x1==x0:
                    raise ValueError("Duplicate axis values cause division by zero.")
                return y0 + (x - x0)/(x1 - x0) * (y1 - y0)
        raise ValueError("x is out of bounds after extrapolation check.")
        pass

    def gradient_wrt_ordinate(self, x : float):
        ### TODO
        


        pass

    def integrate(self, start_x : float, end_x : float):
        ### TODO
        pass

    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):
        ### TODO
        pass

class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1 : np.ndarray | List, 
                               values : np.ndarray | List, 
                               interpolation_method : InterpMethod,
                               extrpolation_method : ExtrapMethod):


        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)
    
        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrpolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')
