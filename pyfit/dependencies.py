import pandas as pd
import numpy as np
from lmfit import Minimizer, Parameters
import subprocess as sp
import os
from datetime import datetime
import pickle

def pickle_results(result, fname):
    with open(fname, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)