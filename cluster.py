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
from graph import *

from plots import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
import mpl_scatter_density
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize


##################################
##################################
##################################
####                          ####
#### Main                     ####
####                          ####
##################################
##################################

begin_time = timelib.time()

log_filename = "Files/MyLog.csv"
url_data_filename = "Files/MyURLs.csv"
filename = "Outputs/Sessions.csv"

####################
# READING DATA FILES
print("\n   * Loading files ...")
start_time = timelib.time()
print("        Loading "+log_filename+" ...", end="\r")
log = pd.read_csv(log_filename, sep=',', na_filter=False, low_memory=False)
print("        "+log_filename+" loaded ({} rows) in {:.1f} seconds.".format(log.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+filename+" ...", end="\r")
sessions = pd.read_csv(filename, sep=',')
print("        "+filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)
start_time = timelib.time()
print("        Loading "+url_data_filename+" ...", end="\r")
urldata = pd.read_csv(url_data_filename, sep=',', na_filter=False)
print("        "+url_data_filename+" loaded ({} rows) in {:.1f} seconds.".format(urldata.shape[0], timelib.time()-start_time))


########
# FILTER
sessions = sessions[sessions.requests > 6]
sessions = sessions[sessions.variance > 0]
sessions = sessions[sessions.inter_req_mean_seconds > 0]
print("\n   * Sessions filtered: {} rows".format(sessions.shape[0]))

############
# CLUSTERING
start_time = timelib.time()
print("\n   * Clustering ...")
kmeans = KMeans(n_clusters=10, random_state=0).fit(sessions[["requests", "timespan", "inter_req_mean_seconds", "variance"]].values)
cluster_labels=kmeans.labels_
sessions["cluster_id"] = cluster_labels
for cluster_id in sessions.cluster_id.unique():
    cluster_session = sessions[sessions.cluster_id == cluster_id].global_session_id.unique()
    i = 0
    for gsid in cluster_session:
        if i >= 10:
            break
        session_draw("Graphs/"+str(cluster_id)+"_gsid"+str(gsid)+".png", gsid, log)
        i += 1
    print("        Producing display of sessions for cluster %d"%cluster_id) 
    cluster_sessions = sessions[sessions.cluster_id == cluster_id].global_session_id.unique()
    cluster_log = log[log.global_session_id.isin(cluster_sessions)]
    plot_sessions(cluster_log,'Clusters/cluster%d.png'%cluster_id,
                  labels=list(log.requested_my_thema.unique()),
                  N_max_sessions=5,field="requested_my_thema",
                  max_time=None,time_resolution=None,mark_requests=False)
plot_palette(labels=list(log.requested_my_thema.unique()), filename="Clusters/palette.png")
print("     Clustered in {:.1f} seconds.".format((timelib.time()-start_time)))

###############
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))