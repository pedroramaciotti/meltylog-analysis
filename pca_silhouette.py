#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 08:35:22 2018

@author: pedro
"""

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

session_filename = "Outputs/Sessions.csv"

pathlib.Path("Report").mkdir(parents=True, exist_ok=True)
pathlib.Path("Report/pca").mkdir(parents=True, exist_ok=True)
pathlib.Path("Report/pca/pairwise").mkdir(parents=True, exist_ok=True)
pathlib.Path("Report/silhouette").mkdir(parents=True, exist_ok=True)

###########
# VARIABLES
dimensions = ["requests", "timespan", "standard_deviation",  "inter_req_mean_seconds", "read_pages"]
#dimensions = ["star_chain_like", "bifurcation"]
#dimensions = ["popularity_mean","entropy", "requested_category_richness", "requested_topic_richness", 'TV_proportion', 'Series_proportion',
#               'News_proportion', 'Celebrities_proportion', 'VideoGames_proportion', 'Music_proportion', 'Movies_proportion', 
#               'Sport_proportion', 'Comic_proportion', 'Look_proportion', 'Other_proportion', 'Humor_proportion', 'Student_proportion', 
#               'Events_proportion', 'Wellbeing_proportion', 'None_proportion', 'Food_proportion', 'Tech_proportion']
#NB_CLUSTERS = [2,3,4, 5, 6, 7, 8, 9, 10,11,12,13,14,15]
range_n_clusters = [2,3,4,5,6]
max_components=len(dimensions)
threshold_explained_variance=0.90

####################
# READING DATA FILES
print("\n   * Loading files ...")
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

print("   > range_n_clusters: {}".format(range_n_clusters))

###############################################################################
# BEFORE PCA, CORRELATION ANALYSIS

corr=sessions[normalized_dimensions].corr()

fig, ax = plt.subplots()
fig.set_size_inches([ 14, 14])
matrix = corr.values
ax.matshow(matrix, cmap=plt.cm.coolwarm)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[0]):
        c = matrix[j,i]
        ax.text(i, j, '%0.2f'%c, va='center', ha='center')
ax.set_xticks(range(len(dimensions)))
ax.set_yticks(range(len(dimensions)))
ax.set_xticklabels(dimensions)
ax.set_yticklabels(dimensions)
plt.savefig('Report/pca/corr_before_pca.pdf', format='pdf')
plt.clf()

###############################################################################
# PCA

pca = PCA(n_components=max_components)

# Data in PCA coordinates: n_samples x n_components
normalized_pca_data=pca.fit_transform(sessions[normalized_dimensions].values)

# selecting components that explain variance
n_components_threshold=len(pca.explained_variance_ratio_[pca.explained_variance_ratio_.cumsum()<threshold_explained_variance])+1

plt.figure()
plt.plot(range(1,max_components+1),100.0*pca.explained_variance_ratio_, 'r+')
plt.axis([0, max_components+1, 0, 100])
plt.gca().axvline(x=n_components_threshold,c='b',alpha=0.25)
plt.text(n_components_threshold+0.5,75,
         '%0.2f%% explained variancce.'%(100*pca.explained_variance_ratio_.cumsum()[n_components_threshold-1]))
plt.xlabel('Component')
plt.ylabel('% Explained Variance')
plt.grid()
plt.savefig('Report/pca/explained_variance_ratio.pdf')
plt.clf()

pca = PCA(n_components=n_components_threshold)
clustering_data=pca.fit_transform(sessions[normalized_dimensions].values)

#clustering_data=normalized_pca_data[:,:n_components_threshold]
#del sessions, normalized_pca_data

###############################################################################
# EXPLAINING PCA COMPONENTS

fig, ax = plt.subplots()
fig.set_size_inches([ 14, 14])
matrix = pca.components_[:n_components_threshold,:].T
ax.matshow(matrix, cmap=plt.cm.coolwarm)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        c = matrix[i,j]
        ax.text(j,i, '%0.2f'%c, va='center', ha='center')
ax.set_xticks(range(n_components_threshold))
ax.set_yticks(range(len(dimensions)))
ax.set_xticklabels(['PC-%d'%n for n in range(1,n_components_threshold+1)])
ax.set_yticklabels(dimensions)
plt.savefig('Report/pca/components.pdf', format='pdf')
plt.clf()
del matrix



############
# CLUSTERING
silhouette_index=[]
for n_clusters in range_n_clusters:
    kmeans=KMeans(n_clusters=n_clusters)
    cluster_labels=kmeans.fit_predict(clustering_data)
    
    
    
    ###############################################################################
    # Scatterplot
    #
#    X=scaler.inverse_transform(normalized_data)
    fig=plt.figure(1)
    plt.scatter(clustering_data[:,0],clustering_data[:,1], c=kmeans.labels_, alpha=0.1)
    plt.axis('equal')
    plt.xlabel('PC-1')
    plt.ylabel('PC-2')
    plt.grid(True)
    # Labeling the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(kmeans.cluster_centers_):
        plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    plt.savefig('Report/pca/pca_scatterplot_cluster_%d.png'%n_clusters)
    plt.clf()
    
    ########################################################
    # Scatterplot in pairwise original feature space pairs #
    ########################################################
    centroids_inverse_pca=pca.inverse_transform(kmeans.cluster_centers_)
    for ftr1 in range(len(dimensions)):
        for ftr2 in range(ftr1+1,len(dimensions)):
            fig=plt.figure(1)
            plt.scatter(sessions[normalized_dimensions].values[:,ftr1],sessions[normalized_dimensions].values[:,ftr2], c=kmeans.labels_, alpha=0.1)
            plt.axis('equal')
            plt.xlabel(dimensions[ftr1])
            plt.ylabel(dimensions[ftr2])
            plt.grid(True)
            # Labeling the clusters
            plt.scatter(centroids_inverse_pca[:,ftr1], centroids_inverse_pca[:, ftr2], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')
            for i, c in enumerate(centroids_inverse_pca):
                plt.scatter(c[ftr1], c[ftr2], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')
            plt.savefig('Report/pca/pairwise/pca_scatterplot_%d_clusters_ftr1_%d_ftr2_%d.png'%(n_clusters,ftr1,ftr2))
            plt.clf()
    
    
    ########################################
    # Silhouette Analysis (Memory Error!!) #
    ########################################
    
    
#    silhouette_avg = silhouette_score(clustering_data, cluster_labels)#,sample_size=50000)
#    silhouette_index.append(silhouette_avg)
    
#    # Compute the silhouette scores for each sample
#    fig, (ax1, ax2) = plt.subplots(1, 2)
#    fig.set_size_inches(18, 7)
#
#    # The 1st subplot is the silhouette plot
#    # The silhouette coefficient can range from -1, 1 but in this example all
#    # lie within [-0.1, 1]
#    ax1.set_xlim([-0.1, 1])
#    # The (n_clusters+1)*10 is for inserting blank space between silhouette
#    # plots of individual clusters, to demarcate them clearly.
#    ax1.set_ylim([0, len(clustering_data) + (n_clusters + 1) * 10])
#    sample_silhouette_values = silhouette_samples(clustering_data, cluster_labels)  
#    y_lower = 10
#    for i in range(n_clusters):
#        # Aggregate the silhouette scores for samples belonging to
#        # cluster i, and sort them
#        ith_cluster_silhouette_values = \
#            sample_silhouette_values[cluster_labels == i]
#
#        ith_cluster_silhouette_values.sort()
#
#        size_cluster_i = ith_cluster_silhouette_values.shape[0]
#        y_upper = y_lower + size_cluster_i
#
#        color = cm.spectral(float(i) / n_clusters)
#        ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                          0, ith_cluster_silhouette_values,
#                          facecolor=color, edgecolor=color, alpha=0.7)
#
#        # Label the silhouette plots with their cluster numbers at the middle
#        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#        # Compute the new y_lower for next plot
#        y_lower = y_upper + 10  # 10 for the 0 samples
#
#    ax1.set_title("The silhouette plot for the various clusters.")
#    ax1.set_xlabel("The silhouette coefficient values")
#    ax1.set_ylabel("Cluster label")
#
##    # The vertical line for average silhouette score of all the values
##    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
##
##    ax1.set_yticks([])  # Clear the yaxis labels / ticks
##    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#    # 2nd Plot showing the actual clusters formed
#    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
#    ax2.scatter(clustering_data[:, 0], clustering_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                c=colors, edgecolor='k')
#
#    # Labeling the clusters
#    centers = kmeans.cluster_centers_
#    # Draw white circles at cluster centers
#    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                c="white", alpha=1, s=200, edgecolor='k')
#
#    for i, c in enumerate(centers):
#        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                    s=50, edgecolor='k')
#
#    ax2.set_title("The visualization of the clustered data.")
#    ax2.set_xlabel("PC-1")
#    ax2.set_ylabel("PC-2")
#
#    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                  "with n_clusters = %d" % n_clusters),
#                 fontsize=14, fontweight='bold')
#    plt.savefig('Report/silhouette/silhouettes_%d.pdf'%n_clusters)
#    plt.clf()
