import time as timelib
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from log2traces import *
from markov import *
from information_theory import *
from log_functions import *
from graph import *

from plots import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *
import pathlib

##################################
##################################
##################################
####                          ####
#### Main                     ####
####                          ####
##################################
##################################

begin_time = timelib.time()

log_filename = "Outputs/MyLog.csv"
session_filename = "Outputs/Sessions.csv"

pathlib.Path("Report").mkdir(parents=True, exist_ok=True)
pathlib.Path("Report/graphs").mkdir(parents=True, exist_ok=True)
pathlib.Path("Report/clusters").mkdir(parents=True, exist_ok=True)
pathlib.Path("Report/pca").mkdir(parents=True, exist_ok=True)
pathlib.Path("Report/silhouette").mkdir(parents=True, exist_ok=True)
latex_output = open("Report/latex_clusters.tex", "w")

###########
# VARIABLES
dimensions = ["requests", "timespan", "standard_deviation",  "inter_req_mean_seconds", "read_pages"]
# dimensions = ["star_chain_like", "bifurcation"]
# dimensions = ["popularity_mean","entropy", "requested_category_richness", "requested_topic_richness", 'TV_proportion', 'Series_proportion',
#               'News_proportion', 'Celebrities_proportion', 'VideoGames_proportion', 'Music_proportion', 'Movies_proportion', 
#               'Sport_proportion', 'Comic_proportion', 'Look_proportion', 'Other_proportion', 'Humor_proportion', 'Student_proportion', 
#               'Events_proportion', 'Wellbeing_proportion', 'None_proportion', 'Food_proportion', 'Tech_proportion']
#NB_CLUSTERS = [2,3,4, 5, 6, 7, 8, 9, 10,11,12,13,14,15]
range_n_clusters = [6, 7, 8, 9, 10]
max_components=len(dimensions)
threshold_explained_variance=0.90

graph = True

####################
# READING DATA FILES
print("\n   * Loading files ...")
start_time = timelib.time()
print("        Loading "+log_filename+" ...", end="\r")
#log = pd.read_csv(log_filename, sep=',', na_filter=False, low_memory=False)
#print("        "+log_filename+" loaded ({} rows) in {:.1f} seconds.".format(log.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+session_filename+" ...", end="\r")
sessions = pd.read_csv(session_filename, sep=',')
print("        "+session_filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)

#################
# FILTER SESSIONS
sessions = sessions[sessions.requests > 6]
print("\n   * Sessions filtered: {} rows".format(sessions.shape[0]))

normalized_dimensions = list(map(lambda x: "normalized_"+x, dimensions)) # normalized dimensions labels list

print("   > Session graph generation: {}".format(graph))
print("   > range_n_clusters: {}".format(range_n_clusters))

latex_output.write("\\begin{frame}{Clustering}\n    Clustering on "+str(len(dimensions))+" dimensions:\n    \\begin{enumerate}\n")
for d in dimensions:
    latex_output.write("        \\item "+d.replace("_", "\_")+"\n")
latex_output.write("    \\end{enumerate}\n\\end{frame}\n\n")


#################
# FILTER NEW BOTS
#log=log[~log.agent.str.contains('BTWebClient')] 
# This should be moved to extraction of session data


###############################################################################
# PCA

pca = PCA(n_components=max_components)

# Data in PCA coordinates: n_samples x n_components
normalized_pca_data=pca.fit_transform(sessions[normalized_dimensions].values)

# selecting components that explain variance
n_components_threshold=len(pca.explained_variance_ratio_[pca.explained_variance_ratio_.cumsum()<threshold_explained_variance])+1

plt.figure()
plt.plot(range(1,max_components+1),100.0*pca.explained_variance_ratio_, 'r+')
plt.axis([-1, max_components+1, 0, 100])
plt.gca().axvline(x=n_components_threshold,c='b',alpha=0.25)
plt.text(n_components_threshold+0.5,75,
         '%0.2f%% explained variancce.'%(100*pca.explained_variance_ratio_.cumsum()[n_components_threshold-1]))
plt.xlabel('Component')
plt.ylabel('% Explained Variance')
plt.grid()
plt.savefig('Report/pca/explained_variance_ratio.pdf')
plt.clf()

clustering_data=normalized_pca_data[:,:n_components_threshold]


############
# CLUSTERING
silhouette_index=[]
for n_clusters in range_n_clusters:
    start_time = timelib.time()
    print("\n  n_clusters * Clustering ("+str(n_clusters)+" clusters) ...")
    pathlib.Path("Report/graphs/"+str(n_clusters)).mkdir(parents=True, exist_ok=True)
    pathlib.Path("Report/clusters/"+str(n_clusters)).mkdir(parents=True, exist_ok=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(clustering_data)
    cluster_labels=kmeans.labels_
    sessions["cluster_id"] = cluster_labels
    num_cluster = sessions.cluster_id.unique()
    num_cluster.sort()

    latex_output.write("\\begin{frame}{Clustering: "+str(n_clusters)+" clusters}\n    \\begin{center}\n        \\resizebox{\\textwidth}{!}{\n            \\begin{tabular}{ccccc}\n                \\includegraphics[width=\\textwidth, keepaspectratio]{clusters/"+str(n_clusters)+"/cluster0}")
    for i in range(1, 10):
        if i >= n_clusters: # no clusters left
            break
        if i == 5: # second row
            latex_output.write(" \\\\\n                \\includegraphics[width=\\textwidth, keepaspectratio]{clusters/"+str(n_clusters)+"/cluster5}")
            continue
        latex_output.write(" & \\includegraphics[width=\\textwidth, keepaspectratio]{clusters/"+str(n_clusters)+"/cluster"+str(i)+"}")
    latex_output.write("\n            \\end{tabular}\n        }\n\n        \\begin{columns}\n            \\begin{column}{.65\\textwidth}\n                \\begin{center}\n                \\scalebox{.25}{\n")

    # recap center
    latex_output.write("                    \\begin{tabular}{|c|c|")
    for dim in dimensions:
        latex_output.write("c|")
    latex_output.write("}\n                        \\hline\n                        cluster & size")
    for dim in dimensions:
        latex_output.write(" & "+str(dim).replace("_", "\_"))
    latex_output.write(" \\\\\n                        \\hline\n")
    for cluster_id in num_cluster:
        latex_output.write("                        "+str(cluster_id)+" & "+str(sessions[sessions.cluster_id==cluster_id].shape[0]))
        for j in range(0, kmeans.cluster_centers_.shape[1]):
            latex_output.write(" & {:.3f}".format(kmeans.cluster_centers_[cluster_id][j]))
        latex_output.write(" \\\\\n                        \\hline\n")
    latex_output.write("                    \\end{tabular}\n                }\n                \\end{center}\n            \\end{column}\n            \\begin{column}{.35\\textwidth}\n                \\includegraphics[width=\\textwidth, keepaspectratio]{clusters/palette_topic}\n            \\end{column}\n        \\end{columns}\n    \\end{center}\n\\end{frame}\n\n")

#    # display
#    for cluster_id in num_cluster:
#        cluster_session = sessions[sessions.cluster_id == cluster_id].global_session_id.unique()
#        print("          Producing display of sessions for cluster %d"%cluster_id,end="\r") 
#        cluster_sessions = sessions[sessions.cluster_id == cluster_id].global_session_id.unique()
#        cluster_log = log[log.global_session_id.isin(cluster_sessions)]
#        sessions_id = plot_sessions(cluster_log,'Report/clusters/'+str(n_clusters)+'/cluster%d.png'%cluster_id, cluster_id,
#                    labels=list(log.requested_topic.unique()),
#                    N_max_sessions=10,field="requested_topic",
#                    max_time=None,time_resolution=None,mark_requests=False)
#        if graph:
#            session_draw(cluster_id, n_clusters, sessions_id, log)
#            latex_output.write("% cluster "+str(cluster_id)+"\n\\begin{frame}{Cluster "+str(cluster_id)+"}\n    \\begin{columns}\n        \\begin{column}{.6\\textwidth}\n            \\includegraphics[width=\\textwidth, keepaspectratio]{clusters/"+str(n_clusters)+"/cluster"+str(cluster_id)+"}\n        \\end{column}\n        \\begin{column}{.4\\textwidth}\n            \\begin{center}\n              \\scalebox{.5}{\\begin{tabular}{|c|c|}\n                  \\hline\n                  \\multicolumn{2}{|c|}{mean} \\\\\n                  \\hline\n                  size & "+str(sessions[sessions.cluster_id==cluster_id].shape[0])+" \\\\\n                  \\hline\n")
#            for i in range(0, kmeans.cluster_centers_.shape[1]):
#                latex_output.write("                  "+normalized_dimensions[i].replace("_", "\_")+" & {:.3f} \\\\\n                  \\hline\n".format(kmeans.cluster_centers_[cluster_id][i]))
#            latex_output.write("              \\end{tabular}}\n\n              \\includegraphics[width=\\textwidth, keepaspectratio]{clusters/palette_topic}\n            \\end{center}\n        \\end{column}\n    \\end{columns}\n\\end{frame}\n\n")
#
#            latex_output.write("\\begin{frame}{Cluster "+str(cluster_id)+" -- Graphs}\n    \\resizebox{\\textwidth}{!}{\n    \\begin{tabular}{c|c|c|c|c}\n        \\huge{"+str(sessions_id[0])+"} & \\huge{"+str(sessions_id[1])+"} & \\huge{"+str(sessions_id[2])+"} & \\huge{"+str(sessions_id[3])+"} & \\huge{"+str(sessions_id[4])+"} \\\\\n        \\includegraphics[width=\\textwidth, keepaspectratio]{Report/graphs/"+str(n_clusters)+"/"+str(cluster_id)+"_session"+str(sessions_id[0])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Report/graphs/"+str(n_clusters)+"/"+str(cluster_id)+"_session"+str(sessions_id[1])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Report/graphs/"+str(n_clusters)+"/"+str(cluster_id)+"_session"+str(sessions_id[2])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Report/graphs/"+str(n_clusters)+"/"+str(cluster_id)+"_session"+str(sessions_id[3])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Report/graphs/"+str(n_clusters)+"/"+str(cluster_id)+"_session"+str(sessions_id[4])+"} \\\\\n        \\hline\n        \\huge{"+str(sessions_id[5])+"} & \\huge{"+str(sessions_id[6])+"} & \\huge{"+str(sessions_id[7])+"} & \\huge{"+str(sessions_id[8])+"} & \\huge{"+str(sessions_id[9])+"} \\\\\n        \\includegraphics[width=\\textwidth, keepaspectratio]{Report/graphs/"+str(n_clusters)+"/"+str(cluster_id)+"_session"+str(sessions_id[5])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Report/graphs/"+str(n_clusters)+"/"+str(cluster_id)+"_session"+str(sessions_id[6])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Report/graphs/"+str(n_clusters)+"/"+str(cluster_id)+"_session"+str(sessions_id[7])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Report/graphs/"+str(n_clusters)+"/"+str(cluster_id)+"_session"+str(sessions_id[8])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Report/graphs/"+str(n_clusters)+"/"+str(cluster_id)+"_session"+str(sessions_id[9])+"}\n    \\end{tabular}}\n\\end{frame}\n\n")
#            print("          Display of sessions succesfully produced for cluster %d"%cluster_id) 
#    print("   * Clustered in {:.1f} seconds.".format((timelib.time()-start_time)))
    
    silhouette_avg = silhouette_score(clustering_data, cluster_labels)
    silhouette_index.append(silhouette_avg)
    
    # Compute the silhouette scores for each sample
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(clustering_data) + (n_clusters + 1) * 10])
    sample_silhouette_values = silhouette_samples(clustering_data, cluster_labels)  
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(clustering_data[:, 0], clustering_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = kmeans.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("PC-1")
    ax2.set_ylabel("PC-2")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.savefig('Report/silhouette/silhouettes_%d.pdf'%n_clusters)
    plt.clf()
    
    # Scatter plot
    fig=plt.figure(1)
    plt.subplot(121)
    plt.scatter(normalized_pca_data[:,0],normalized_pca_data[:,1], c=data['class'].values, alpha=0.5)
    plt.subplot(122)
    plt.scatter(normalized_pca_data[:,0],normalized_pca_data[:,1], c=kmeans.labels_, alpha=0.5)
    plt.savefig('Figures/cluterisation_%d.pdf'%n_clusters)
    plt.clf()
    
plot_palette(labels=list(log.requested_topic.unique()), filename="Report/clusters/palette_topic.png")
plot_palette(labels=list(log.requested_category.unique()), filename="Report/clusters/palette_category.png")

###############
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time)) 