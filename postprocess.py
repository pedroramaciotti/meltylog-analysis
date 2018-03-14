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
requests_threshold = 6
parameters = ["requests","timespan","requested_category_richness","requested_my_thema_richness","star_chain_like","bifurcation","entropy","variance","popularity_mean","inter_req_mean_seconds","TV_proportion","Celebrities_proportion","Series_proportion","Movies_proportion","Music_proportion","Unclassifiable_proportion","Comic_proportion","VideoGames_proportion","Other_proportion","Sport_proportion","News_proportion","read_pages"]
log_scale_parameters = ["requests", "timespan", "inter_req_mean_seconds"]

####################
# READING DATA FILES 
start_time = timelib.time()
print("\n   * Loading "+filename+" ...", end="\r")
sessions = pd.read_csv(filename, sep=',')
print("   * "+filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)

###############
# NORMALIZATION
start_time = timelib.time()
normalized_parameters = []
print("\n   * Normalizing parameters ...", end="\r")
for p in parameters:
    normalized_parameters.append("normalized_"+p)
    sessions["normalized_"+p] = normalize(sessions[p])
print("   * Parameters normalized in %.1f seconds." %(timelib.time()-start_time))

sessions = sessions[sessions.requests>requests_threshold]


#######################
# GENERATING HISTOGRAMS
start_time = timelib.time()
print("\n   * Generating histograms ...")
for p in parameters:
    # regular histograms
    print("        Generating Matplot/"+p+".pdf ...", end="\r")
    plt.hist(sessions[p].values, align="left")
    plt.xlabel(p)
    plt.ylabel("Frequency")
    if p in log_scale_parameters:
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
    plt.savefig("Matplot/"+p+".pdf",format='pdf')
    print("        Matplot/"+p+".pdf successfully generated")
    plt.clf()
    # normalized histograms
    print("        Generating Matplot/Normalized/normalized_"+p+".pdf ...", end="\r")
    plt.hist(sessions["normalized_"+p].values, align="left")
    plt.xlabel("Normalized "+p)
    plt.ylabel("Frequency")
    if p in log_scale_parameters:
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
    plt.savefig("Matplot/Normalized/normalized_"+p+".pdf",format='pdf')
    print("        Matplot/Normalized/normalized_"+p+".pdf successfully generated")
    plt.clf()
print("   * Histograms generated in %.1f seconds." %(timelib.time()-start_time))

##########################
# GENERATING SCATTER PLOTS
start_time = timelib.time()
print("\n   * Generating scatter plots ...")
# TODO
print("   * Scatter plots generated in %.1f seconds." %(timelib.time()-start_time))

###############
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))
