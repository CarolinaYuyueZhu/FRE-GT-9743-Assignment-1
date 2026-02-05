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
        if len(self.axis1_) != len(self.values_):
            raise ValueError("axis1 and values should have the same length.")
        # sort values based on axis1
        idx = np.argsort(self.axis1_)
        sorted_axis1 = self.axis1_[idx]
        sorted_values = self.values_[idx]

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
                    raise ValueError("duplicate axis values cause division by zero.")
                return y0 + (x - x0)/(x1 - x0) * (y1 - y0)
        raise ValueError("x is out of bounds after extrapolation check.")
        pass

    def gradient_wrt_ordinate(self, x : float):
        if len(self.axis1_) != len(self.values_):
            raise ValueError("axis1 and values should have the same length.")
        
        # sort values based on axis1 (like what we did in def_interpolate)
        idx = np.argsort(self.axis1_)
        sorted_axis1 = self.axis1_[idx]    
        sorted_values = self.values_[idx]

        # create a gradient array initialized to zero
        g = np.zeros(len(sorted_values))

        # flat extrapolation gradient
        # if the x is out of bounds, the gradient is 1 for the closest point
        if x <= sorted_axis1[0]:
            g[0] = 1.0
            return g
        elif x >= sorted_axis1[-1]:
            g[-1] = 1.0
            return g
        
        # linear interpolation gradient
        for i in range(len(sorted_axis1)-1):
            x0 = sorted_axis1[i]
            x1 = sorted_axis1[i+1]

            if x0 <= x < x1:
                if x1==x0:
                    raise ValueError("duplicate axis values, cause error because the division of zero.")
                # because we can write the linear interpolation as:
                # y = y0 + (x - x0)/(x1 - x0) * (y1 - y0)
                # we can write the weights as w = (x - x0)/(x1 - x0)
                w = (x - x0)/(x1 - x0)
                # so the function becomes y(x) = (1-w) * y0 + w * (y1)
                # calculate the gradient with respect to y0 - i.e.(yi) and y1 - i.e.(yi+1)
                g[i] = 1-w
                g[i+1] = w
                return g
        raise ValueError("Can not bracket x for gradient computation.")
        pass

    def integrate(self, start_x : float, end_x : float):
        if len(self.axis1_) != len(self.values_):
            raise ValueError("axis1 and values should have the same length.")
        if len(self.axis1_) < 2:
            raise ValueError("you need at least two grid points to integrate.")
        
        # sort values based on axis1
        idx = np.argsort(self.axis1_)
        sorted_axis1 = self.axis1_[idx]
        sorted_values = self.values_[idx]
        
        # use trapezoidal rule to integrate, for FLAT extrapolation
        # initialize the integral value to zero
        integral = 0.0

        if start_x >= end_x:
            return 0.0
        # if limits are reversed, the integral should change the sign
        if end_x < start_x:
            return -self.integrate(end_x, start_x)

        y_left = sorted_values[0]
        y_right = sorted_values[-1]

        # handle the extrapolation on the left side (left extrapolation)
        if start_x < sorted_axis1[0]:
            a = start_x
            b = min(end_x, sorted_axis1[0])
            if b > a:
                integral += y_left * (b - a)
        # handle the extrapolation on the right side (right extrapolation)
        if end_x > sorted_axis1[-1]:
            a = max(start_x, sorted_axis1[-1])
            b = end_x
            if b > a:
                integral += y_right * (b - a)
       
        # handle the integration within the grid
        L = max(start_x, sorted_axis1[0])
        U = min(end_x, sorted_axis1[-1])
        if U > L:
            # find the indices that bracket L and U
            left_idx = np.searchsorted(sorted_axis1, L, side='right') - 1
            right_idx = np.searchsorted(sorted_axis1, U, side='left')

            # integrate from L to U
            x_prev = L
            y_prev = self.interpolate(L)
            # walk through interior grid knots and add trapezoids
            for i in range(left_idx + 1, right_idx):
                x_curr = sorted_axis1[i]
                y_curr = sorted_values[i]
                integral += 0.5 * (y_prev + y_curr) * (x_curr - x_prev)
                # moves forward
                x_prev = x_curr
                y_prev = y_curr
            # handles the final piece [x_prev, U]
            x_curr = U
            y_curr = self.interpolate(U)
            integral += 0.5 * (y_prev + y_curr) * (x_curr - x_prev)
        return integral

        pass

    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):
        if len(self.axis1_) != len(self.values_):
            raise ValueError("axis1 and values should have the same length.")
        if len(self.axis1_) < 2:
            raise ValueError("you need at least two grid points to integrate.")
        
        # if limits are reversed, the integral should change the sign
        if end_x < start_x:
            return -self.gradient_of_integrated_value_wrt_ordinate(end_x, start_x)      
        
        # sort values based on axis1
        idx = np.argsort(self.axis1_)
        sorted_axis1 = self.axis1_[idx]
        sorted_values = self.values_[idx]

        # create a gradient array initialized to zero
        g = np.zeros(len(sorted_values))
        
        if start_x == end_x:
            return g
        
        # handle the extrapolation on the left side (left extrapolation)
        if start_x < sorted_axis1[0]:
            a = start_x
            b = min(end_x, sorted_axis1[0])
            if b > a:
                g[0] +=  (b - a)
        # handle the extrapolation on the right side (right extrapolation)
        if end_x > sorted_axis1[-1]:
            a = max(start_x, sorted_axis1[-1])
            b = end_x
            if b > a:
                g[-1] += (b - a)

        # handle the integration within the grid
        L = max(start_x, sorted_axis1[0])
        U = min(end_x, sorted_axis1[-1])
        if U > L:
            # find the indices that bracket L and U
            left_idx = np.searchsorted(sorted_axis1, L, side='right') - 1
            right_idx = np.searchsorted(sorted_axis1, U, side='left')

            # integrate from L to U
            x_prev = L
            g_prev = self.gradient_wrt_ordinate(L)
            # walk through interior grid knots and add trapezoids
            for i in range(left_idx + 1, right_idx):
                x_curr = sorted_axis1[i]
                g_curr = sorted_values[i]
                integral += 0.5 * (g_prev + g_curr) * (x_curr - x_prev)
                # moves forward
                x_prev = x_curr
                g_prev = g_curr
            # handles the final piece [x_prev, U]
            x_curr = U
            g_curr = self.gradient_wrt_ordinate(U)
            g += 0.5 * (g_prev + g_curr) * (x_curr - x_prev)
        return g


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
