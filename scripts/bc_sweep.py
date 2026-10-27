from pathlib import Path
import csv 
import sys

import matplotlib.pyplot as plt 
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))

from plasticity.config import load_config
from plasticity.model import initialize_web

B_RANGE = (0.05, 1.0)
GRID_SIZE = 5
REFINE_ROUNDS = 2
POINTS_PER_ROUND = 10
SEARCH_SEED = 123


