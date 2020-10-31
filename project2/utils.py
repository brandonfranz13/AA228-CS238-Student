import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from collections import namedtuple, defaultdict

def writePolicy(filename, actions):
    np.savetxt(filename, actions, fmt='%i', delimiter=",")
    
def importCSV(filename):
    contents = pd.read_csv(filename)
    return contents.columns, contents.to_numpy()