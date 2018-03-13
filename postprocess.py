import time as timelib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from thema_mapper import *
from log2traces import *
from markov import *
from information_theory import *
from log_functions import *

from plots import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns

##################################
##################################
##################################
####                          ####
#### Main                     ####
####                          ####
##################################
##################################

begin_time = timelib.time()

###########
# CONSTANTS
filename = "Outputs/Mini_sessions.csv"
requests_threshold = 20

####################
# READING DATA FILES 
start_time = timelib.time()
sessions = pd.read_csv(filename, sep=',', dtype='object', na_filter=False)

############################################################################################################
# TYPE CASTING (read csv from previously generated file session import data in str type, annoying isn't it?)
sessions.requests = sessions.requests.apply(lambda x: int(x))

###############
# NORMALIZATION
sessions["normalized_requests"] = normalize(sessions.requests)

print(sessions[sessions.requests.values>requests_threshold][["requests", "normalized_requests"]])

###############
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))
