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

filename = "Outputs/Sessions.csv"

parameters = ["requests","timespan","requested_category_richness","requested_my_thema_richness","star_chain_like","bifurcation","entropy","variance","popularity_mean","inter_req_mean_seconds","TV_proportion","Celebrities_proportion","Series_proportion","Movies_proportion","Music_proportion","Unclassifiable_proportion","Comic_proportion","VideoGames_proportion","Other_proportion","Sport_proportion","News_proportion","read_pages"]

log_scale_parameters = ["requests", "timespan", "inter_req_mean_seconds", "variance", "read_pages", "popularity_mean"]

log_scale_parameters_y = ["entropy", "bifurcation", "TV_proportion","Celebrities_proportion","Series_proportion","Movies_proportion","Music_proportion","Unclassifiable_proportion","Comic_proportion","VideoGames_proportion","Other_proportion","Sport_proportion","News_proportion", "requested_category_richness", "requested_my_thema_richness"]

# for scatter plots
my_paramaters = ["requests", "timespan", "variance", "star_chain_like", "bifurcation", "inter_req_mean_seconds"]

# for log normalization
lognorm = ["requests", "timespan", "inter_req_mean_seconds", "variance"]

# latex code generation output file
latex_ouput = open("Outputs/latex", "w")

# GENERATORS
histogen = True
scattergen = True
scatter_density = True

####################
# READING DATA FILES 
start_time = timelib.time()
print("\n   * Loading "+filename+" ...", end="\r")
sessions = pd.read_csv(filename, sep=',')
print("   * "+filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)

########
# FILTER
sessions = sessions[sessions.requests > 6]
sessions = sessions[sessions.variance > 0]
sessions = sessions[sessions.inter_req_mean_seconds > 0]
print("\n   * Sessions filtered: {} rows".format(sessions.shape[0]))

###############
# NORMALIZATION
start_time = timelib.time()
normalized_parameters = []
print("\n   * Normalizing parameters ...", end="\r")
for p in parameters:
    normalized_parameters.append("normalized_"+p)
    if p in lognorm:
        sessions["normalized_"+p] = log_normalize(sessions[p])
    else:
        sessions["normalized_"+p] = normalize(sessions[p])
print("   * Parameters normalized in %.1f seconds." %(timelib.time()-start_time))

#######################
# GENERATING HISTOGRAMS
print("\n   > Histograms: {}".format(histogen))
print("   > Scatter plots: {}".format(scattergen))
print("   > Density scatter plots: {}".format(scatter_density))

if histogen:
    start_time = timelib.time()
    print("\n   * Generating histograms ...", end="\r")
    for p in parameters:
        # regular histograms
        plt.hist(sessions[p].values, align="left")
        plt.grid(alpha=0.5)
        plt.xlabel(p)
        plt.ylabel("Frequency")
        latex_ouput.write("\\begin{frame}\n\\frametitle{"+p.replace("_", "\_")+"}\n    \\begin{center}\n        \\begin{tabular}{>{\\centering\\arraybackslash}m{0.4\\linewidth}>{\\centering\\arraybackslash}m{0.15\\linewidth}>{\\centering\\arraybackslash}m{0.4\\linewidth}}\n            \\includegraphics[scale=0.27]{plots/"+p+"} &")
        if p in lognorm:
            latex_ouput.write(" \\tiny{$\\xrightarrow[\\frac{\\log(x)-\\log(min)}{\\log(max)-\\log(min)}]{}$} &")
        else:
            latex_ouput.write(" $\\xrightarrow[\\frac{x-min}{max-min}]{}$ &")
        latex_ouput.write(" \\includegraphics[scale=0.27]{plots/Normalized/normalized_"+p+"} \\\\\n")
        if p in log_scale_parameters:
            plt.gca().set_xscale('log')
            plt.gca().set_yscale('log')
            latex_ouput.write("            log scale on x and y & & log scale on y\n")
        elif p in log_scale_parameters_y:
            plt.gca().set_yscale('log')
            latex_ouput.write("            log scale on y & & log scale on y\n")
        latex_ouput.write("        \\end{tabular}\n    \\end{center}\n\\end{frame}\n\n")
        plt.savefig("Matplot/"+p+".png",format='png')
        plt.clf()
        # normalized histograms
        bincuts=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        plt.hist(sessions["normalized_"+p].values, bins=bincuts, align="left")
        plt.grid(alpha=0.5)
        plt.xlabel("Normalized "+p)
        plt.ylabel("Frequency")
        if p in log_scale_parameters or p in log_scale_parameters_y:
            plt.gca().set_yscale('log')
        plt.savefig("Matplot/Normalized/normalized_"+p+".png",format='png')
        plt.clf()
        plt.close()
    print("   * Histograms generated in %.1f seconds." %(timelib.time()-start_time))

##########################
# GENERATING SCATTER PLOTS
if scattergen:
    start_time = timelib.time()
    print("\n   * Generating scatter plots ...", end="\r")
    for i in range (0, len(my_paramaters)):
        for j in range (i+1, len(my_paramaters)):
            plt.scatter(sessions[my_paramaters[i]].values, sessions[my_paramaters[j]].values)
            plt.grid(alpha=0.5)
            plt.xlabel(my_paramaters[i])
            plt.ylabel(my_paramaters[j])
            if my_paramaters[i] in log_scale_parameters:
                plt.gca().set_xscale('log')
            if my_paramaters[j] in log_scale_parameters:
                plt.gca().set_yscale('log')
            plt.savefig("Matplot/Scatter/"+my_paramaters[i]+"-VS-"+my_paramaters[j]+".png",format='png')
            plt.clf()
            plt.scatter(sessions["normalized_"+my_paramaters[i]].values, sessions["normalized_"+my_paramaters[j]].values)
            plt.grid(alpha=0.5)
            plt.xlabel("Normalized "+my_paramaters[i])
            plt.ylabel("Normalized "+my_paramaters[j])
            plt.savefig("Matplot/Scatter/Normalized/normalized_"+my_paramaters[i]+"-VS-normalized_"+my_paramaters[j]+".png",format='png')
            plt.clf()
            plt.close()
    print("   * Scatter plots generated in %.1f seconds." %(timelib.time()-start_time))

##################################
# GENERATING DENSITY SCATTER PLOTS
if scatter_density:
    start_time = timelib.time()
    print("\n   * Generating scatter density ...", end="\r")
    latex_ouput.write("\n%%%%%%%%\n% SCATTER PLOTS\n%%%%%%%%\n\n")
    for i in range (0, len(my_paramaters)):
        for j in range (i+1, len(my_paramaters)):
            norm = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())

            plt.hist2d(sessions["normalized_"+my_paramaters[i]].values, sessions["normalized_"+my_paramaters[j]].values, norm=norm, bins=(20,20))
            plt.colorbar(label="Number of points per pixel")
            plt.xlabel("normalized_"+my_paramaters[i])
            plt.ylabel("normalized_"+my_paramaters[j])
            plt.savefig("Matplot/Density/"+my_paramaters[i]+"-VS-"+my_paramaters[j]+".png",format='png')
            plt.clf()
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
            density = ax.scatter_density(sessions["normalized_"+my_paramaters[i]].values, sessions["normalized_"+my_paramaters[j]].values, norm=norm)
            plt.xlabel("normalized_"+my_paramaters[i])
            plt.ylabel("normalized_"+my_paramaters[j])
            fig.colorbar(density, label="Number of points per pixel")
            plt.savefig("Matplot/Density/v2/"+my_paramaters[i]+"-VS-"+my_paramaters[j]+".png",format='png')
            plt.clf()

            latex_ouput.write("\\begin{frame}\n\\frametitle{"+my_paramaters[i].replace("_", "\_")+" VS "+my_paramaters[j].replace("_", "\_")+"}\n    \\begin{center}\n        \\includegraphics[scale=0.3]{plots/Scatter/"+my_paramaters[i]+"-VS-"+my_paramaters[j]+"}\n    \\end{center}\n    \\begin{tabular}{cc}\n        \\includegraphics[scale=0.3]{plots/Density/"+my_paramaters[i]+"-VS-"+my_paramaters[j]+"} & \\includegraphics[scale=0.3]{plots/Density/v2/"+my_paramaters[i]+"-VS-"+my_paramaters[j]+"}\n    \\end{tabular}\n\\end{frame}\n\n")
    print("   * Scatter density generated in %.1f seconds." %(timelib.time()-start_time))

###############
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))
