import numpy as np
from dataclasses import dataclass,field


@dataclass
class Mobility_Functions:

    """
   A12(s,lam) = A21(s,lam) = A12(s, 1 /lam) = A21(s, 1 / lam) 
   similar for B12 and B21 as well
   A11(s,lam) = A22(s, 1/lam) or A11(s,1 / lam) = A22(s, lam)
   B11(s,lam) = B22(s, 1/lam) or B11(s,1 / lam) = B22(s, lam)
   
   """
    dimensionless_distance: np.ndarray
    lamda: np.ndarray
    la: np.ndarray = field(init=False)
    s: np.ndarray  = field(init=False)
    
    def __post_init__(self):
         self.update_aliases()

    def update_aliases(self):
        self.la = self.lamda
        self.s = self.dimensionless_distance
    
    @property
    def dimensionless_distance(self):        
        return self._dimensionless_distance

    @property
    def lamda(self):        
        return self._lamda
    
    @dimensionless_distance.setter
    def dimensionless_distance(self,new_dim: np.ndarray) -> None:
        self.s, self._dimensionless_distance = new_dim, new_dim
        
    @lamda.setter
    def lamda(self, new_lamda: np.ndarray) -> None:
        self.s, self._lamda = new_lamda, new_lamda
      
        
        
    #@staticmethod
    def A_alp_alp(self, dimen_dist: np.ndarray, sz_rat: np.ndarray) -> np.ndarray:
        term1_A11 = 1
        term2_A11 = (60 * sz_rat **3) / np.power((1 + sz_rat) * dimen_dist, 4)
        term3_A11 = (32 * sz_rat **3) * (15 - 4 * sz_rat **2) / np.power((1 + sz_rat) * dimen_dist, 6)
        term4_A11 = (192 * sz_rat **3) * (5 - 22 * sz_rat **2 + 3 * sz_rat **4) / np.power((1 + sz_rat) * dimen_dist, 8)
    
        return term1_A11 - term2_A11 + term3_A11 - term4_A11
        
    #@staticmethod   
    def B_alp_alp(self, dimen_dist: np.ndarray, sz_rat: np.ndarray) -> np.ndarray:
        term1_B11 = 1
        term2_B11 = (68 * sz_rat **5) / np.power((1 + sz_rat) * dimen_dist, 6)
        term3_B11 = (32 * sz_rat **3) * (10 - 9 * (sz_rat **2) + 9 * sz_rat **4) / np.power((1 + sz_rat) * dimen_dist, 8)
    
        return term1_B11 - term2_B11 - term3_B11
        
    #@staticmethod            
    def A_alp_bet(self, dimen_dist: np.ndarray, sz_rat: np.ndarray) -> np.ndarray:
        term1_A12 = 3 / (2 * dimen_dist)
        term2_A12 = (2 *(1 + sz_rat **2)) / ((1 + sz_rat) **2 * dimen_dist **3)
        term3_A12 = (1200 * sz_rat **3) / ((1 + sz_rat) **6 * dimen_dist **7)
        
        return term1_A12 - term2_A12 + term3_A12

    #@staticmethod
    def B_alp_bet(self, dimen_dist: np.ndarray, sz_rat: np.ndarray) -> np.ndarray:
        term1_B12 = 3 / (4 * dimen_dist)
        term2_B12 = (1 + sz_rat **2) / ((1 + sz_rat) **2 * dimen_dist **3)
        
        return term1_B12 + term2_B12

    # A - mob functions
    @property
    def A11(self) -> np.ndarray:
        return self.A_alp_alp(self.s, self.la)
    
    @property
    def A22(self) -> np.ndarray:
        return self.A_alp_alp(self.s, 1 / self.la)
    
    @property
    def A12(self) -> np.ndarray:
        return self.A_alp_bet(self.s, self.la)
    
    @property
    def A21(self) -> np.ndarray:
        return self.A_alp_bet(self.s, 1 / self.la) 

    # B -mob functions
    @property
    def B11(self) -> np.ndarray:
        return self.B_alp_alp(self.s, self.la)
    
    @property
    def B22(self) -> np.ndarray:
        return self.B_alp_alp(self.s, 1 / self.la)
    
    @property
    def B12(self) -> np.ndarray:
        return self.B_alp_bet(self.s, self.la)
    
    @property
    def B21(self) -> np.ndarray:
        return self.B_alp_bet(self.s, 1 / self.la)
