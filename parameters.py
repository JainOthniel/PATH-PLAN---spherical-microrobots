"""
Paramter object defines the variable required for entry in the 
U (velocity ) equation.
enter position_array as the relatiuve position of the obstacle wrt Robot
"""

import numpy as np 
from dataclasses import dataclass

# 
@dataclass
class Parameters:
    viscosity: float
    radius_robot: float
    radius_obstacle: float
    Force: np.ndarray
    position_robot: np.ndarray
    position_obstacle: np.ndarray

    @property
    def position_array(self):
         return self.position_obstacle - self.position_robot
    
    
    @property
    def position_robot(self):
        return self._position_robot
        
        
    @position_robot.setter
    def position_robot(self, value: np.ndarray) -> None:
        self._position_robot = value
        #self.position_array =  update_position_array()
        
    @property
    def position_obstacle(self):
        return self._position_obstacle
        
        
    @position_obstacle.setter
    def position_obstacle(self, value: np.ndarray) -> None:
        self._position_obstacle = value
        #self.position_array = update_position_array()
    
    @property
    def Force(self):
        return self._Force
    
    @Force.setter    
    def Force(self, value: np.ndarray) -> None:
        self._Force = value    
    
    @property
    def distance_cal(self) -> np.ndarray:
        return np.linalg.norm(self.position_array, ord=2, axis=1)
        
    @property
    def outer_product(self) -> np.ndarray:
        """
        outer product is nothing but matrix product of the flatten array, where the
        product is taken between the column array(first operand) and row array(second operand) 
        """
        column_array = self.position_array.reshape((self.position_array.shape[0], self.position_array.shape[1], 1))
        row_array    = self.position_array.reshape((self.position_array.shape[0], 1, self.position_array.shape[1]))
                                              
        return column_array @ row_array
        
    @property
    def unit_tensor_mat(self) -> np.ndarray:
        """
    we want the unit tensor to be of the size of outer prosuct and also since 
    we are evaluvating all the obstacle-robot pair ingeraction together we want 
    a column array with each element being the unit tensor
    """
        return np.tile(np.eye(self.position_array.shape[1]), (self.position_array.shape[0], 1, 1))#(self.position_array.shape[0], self.position_array.shape[1], 1, 1))
        

    
