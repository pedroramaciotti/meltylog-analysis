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

filename = "Outputs/Sessions.csv"

parameters = ["requests","timespan","requested_category_richness","requested_my_thema_richness","star_chain_like","bifurcation","entropy","variance","popularity_mean","inter_req_mean_seconds","TV_proportion","Celebrities_proportion","Series_proportion","Movies_proportion","Music_proportion","Unclassifiable_proportion","Comic_proportion","VideoGames_proportion","Other_proportion","Sport_proportion","News_proportion","read_pages"]

log_scale_parameters = ["requests", "timespan", "inter_req_mean_seconds"]

# GENERATORS
histogen = True
scattergen = True
scatter3d = False

####################
# READING DATA FILES 
start_time = timelib.time()
print("\n   * Loading "+filename+" ...", end="\r")
sessions = pd.read_csv(filename, sep=',')
print("   * "+filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)

# FILTER
sessions = sessions[sessions.requests>6]

###############
# NORMALIZATION
start_time = timelib.time()
normalized_parameters = []
print("\n   * Normalizing parameters ...", end="\r")
for p in parameters:
    normalized_parameters.append("normalized_"+p)
    sessions["normalized_"+p] = normalize(sessions[p])
print("   * Parameters normalized in %.1f seconds." %(timelib.time()-start_time))

#######################
# GENERATING HISTOGRAMS
if histogen:
    start_time = timelib.time()
    print("\n   * Generating histograms ...", end="\r")
    for p in parameters:
        # regular histograms
        plt.hist(sessions[p].values, align="left")
        plt.grid(alpha=0.5)
        plt.xlabel(p)
        plt.ylabel("Frequency")
        if p in log_scale_parameters:
            plt.gca().set_xscale('log')
            plt.gca().set_yscale('log')
        plt.savefig("Matplot/"+p+".png",format='png')
        plt.clf()
        # normalized histograms
        bincuts=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        plt.hist(sessions["normalized_"+p].values,bins=bincuts)
        plt.grid(alpha=0.5)
        plt.xlabel("Normalized "+p)
        plt.ylabel("Frequency")
        plt.savefig("Matplot/Normalized/normalized_"+p+".png",format='png')
        plt.clf()
    print("   * Histograms generated in %.1f seconds." %(timelib.time()-start_time))

##########################
# GENERATING SCATTER PLOTS
if scattergen:
    start_time = timelib.time()
    print("\n   * Generating scatter plots ...")
    for i in range (0, len(parameters)):
        for j in range (i+1, len(parameters)):
            plt.scatter(sessions[parameters[i]].values, sessions[parameters[j]].values)
            plt.grid(alpha=0.5)
            plt.xlabel(parameters[i])
            plt.ylabel(parameters[j])
            plt.savefig("Matplot/Scatter/"+parameters[i]+"-VS-"+parameters[j]+".png",format='png')
            plt.clf()
            plt.scatter(sessions["normalized_"+parameters[i]].values, sessions["normalized_"+parameters[j]].values)
            plt.grid(alpha=0.5)
            plt.xlabel("Normalized "+parameters[i])
            plt.ylabel("Normalized "+parameters[j])
            plt.savefig("Matplot/Scatter/Normalized/normalized_"+parameters[i]+"-VS-normalized_"+parameters[j]+".png",format='png')
            plt.clf()
    print("   * Scatter plots generated in %.1f seconds." %(timelib.time()-start_time))

###############
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))
