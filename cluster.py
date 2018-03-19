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

log_filename = "Files/MeltyLog_2Sep2017.csv"
url_data_filename = "Files/MeltyURLs_2Sep2017.csv"
filename = "Outputs/Sessions.csv"

####################
# READING DATA FILES
print("\n   * Loading files ...")
start_time = timelib.time()
print("        Loading "+log_filename+" ...", end="\r")
log = pd.read_csv(log_filename, sep=',', dtype='object', na_filter=False)
print("        "+log_filename+" loaded ({} rows) in {:.1f} seconds.".format(log.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+filename+" ...", end="\r")
sessions = pd.read_csv(filename, sep=',')
print("        "+filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)
start_time = timelib.time()
print("        Loading "+url_data_filename+" ...", end="\r")
urldata = pd.read_csv(url_data_filename, sep=',',dtype='object', na_filter=False)
urldata.drop_duplicates(subset=['url'], inplace=True)
print("        "+url_data_filename+" loaded ({} rows) in {:.1f} seconds.".format(urldata.shape[0], timelib.time()-start_time))

# Computing global session ids
start_time = timelib.time()
print("\n   * Computing session IDs ...",end='\r')
log=log_sessions(log,30)
print("     Session IDs computed in in %.1f seconds."%(timelib.time()-start_time)) 

#######################################################
## MANAGING URL CLASSIFICATION: CATEGORY, THEMA, OTHER?
start_time = timelib.time()
print("\n   * Computing my_thema ...")
urldata['my_thema'] = urldata['melty_thema_name'].apply(thema_mapper)
urldata['my_thema'] = urldata.apply(lambda row: row.category+'(ext)' if not(row.url.startswith('www.melty.fr')) else row.my_thema, axis=1)
print("     my_thema field computed in %.1f seconds." %(timelib.time()-start_time))

#######################################
## ASSIGNING URL TO URLs IN LOG ENTRIES
start_time = timelib.time()
print("\n   * Asigning URL data to log entries ...")
log = log_classification(log, urldata, fields=['category', 'melty_thema_name', 'my_thema'])
## ASSIGNING VALUES TO NA VALUES
log.requested_category.fillna('other', inplace=True)
log.referrer_category.fillna('other', inplace=True)
log.requested_melty_thema_name.fillna('', inplace=True)
log.referrer_melty_thema_name.fillna('', inplace=True)
log.requested_my_thema.fillna('Unclassifiable', inplace=True)
log.referrer_my_thema.fillna('Unclassifiable', inplace=True)
print("     URL data assigned to log entries in %.1f seconds." %(timelib.time()-start_time))

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
    print("        Producing display of sessions for cluster %d"%cluster_id)
    cluster_sessions = sessions[sessions.cluster_id == cluster_id].global_session_id.unique()
    cluster_log = log[log.global_session_id.isin(cluster_sessions)]
    plot_sessions(cluster_log,'Clusters/cluster%d.pdf'%cluster_id,
                  labels=list(log.requested_my_thema.unique()),
                  N_max_sessions=30,field="requested_my_thema",
                  max_time=None,time_resolution=None,mark_requests=False)
print("     Clustered in {:.1f} seconds.".format((timelib.time()-start_time)))

###############
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))