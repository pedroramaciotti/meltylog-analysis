import time as timelib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as News_proportion
from sklearn.cluster import KMeans

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
import pathlib
import sys, os

#############################################################
#############################################################
####                 _ _               _              _  ####
####  _ __ ___   ___| | |_ _   _      | |_ ___   ___ | | ####
#### | '_ ` _ \ / _ \ | __| | | |_____| __/ _ \ / _ \| | ####
#### | | | | | |  __/ | |_| |_| |_____| || (_) | (_) | | ####
#### |_| |_| |_|\___|_|\__|\__, |      \__\___/ \___/|_| ####
####                       |___/                         ####
####                                                     ####
#############################################################
#############################################################

logo = "\n                _ _               _              _ \n _ __ ___   ___| | |_ _   _      | |_ ___   ___ | |\n| '_ ` _ \ / _ \ | __| | | |_____| __/ _ \ / _ \| |\n| | | | | |  __/ | |_| |_| |_____| || (_) | (_) | |\n|_| |_| |_|\___|_|\__|\__, |      \__\___/ \___/|_|\n                      |___/                        \n\n"

# ====================== #
#     MENU FUNCTIONS     #
# ====================== #

menu_actions = {}

def main_menu():
    os.system("clear")
    print(logo)
    print("# ==================== #\n#     MAIN PROGRAM     #\n# ==================== #\n")
    print("1. Clustering")
    print("2. Session log")
    print("\n0. Quit")
    choice = input(" >>  ")
    exec_menu(choice)
    return

def exec_menu(choice):
    os.system("clear")
    print(logo)
    ch = choice.lower()
    if ch == "":
        menu_actions["main_menu"]()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Invalid selection, please try again.\n")
            menu_actions["main_menu"]()
    return

def clustering_menu():
    print("# ================== #\n#     CLUSTERING     #\n# ================== #\n")
    print("9. Back")
    print("0. Quit")
    choice = input(" >>  ")
    exec_menu(choice)
    return

def session_log_menu():
    print("# ======================== #\n#     SESSION LOG MENU     #\n# ======================== #\n")
    print("9. Back")
    print("0. Quit")
    choice = input(" >>  ")
    exec_menu(choice)
    return

def back():
    menu_actions["main_menu"]()

def exit():
    os.system("clear")
    sys.exit()

menu_actions = {
    "main_menu": main_menu,
    "1": clustering_menu,
    "2": session_log_menu,
    "0": exit,
}

# ==================== #
#     MAIN PROGRAM     #
# ==================== #

print(logo)

log_filename = "Files/MyLog.csv"
session_filename = "Files/Sessions.csv"

dimensions = ["requests","timespan","requested_category_richness","requested_my_thema_richness","star_chain_like","bifurcation","entropy","standard_deviation","popularity_mean","inter_req_mean_seconds","TV_proportion","Celebrities_proportion","Series_proportion","Movies_proportion","Music_proportion","Unclassifiable_proportion","Comic_proportion","VideoGames_proportion","Other_proportion","Sport_proportion","News_proportion","read_pages"]

# for log normalization
lognorm = ["requests", "timespan", "inter_req_mean_seconds", "standard_deviation", "popularity_mean"]

# loading files
# print("\n   * Loading files ...")
# start_time = timelib.time()
# print("\n        Loading "+log_filename+" ...", end="\r")
# log = pd.read_csv(log_filename, sep=',', na_filter=False, low_memory=False)
# print("        "+log_filename+" loaded ({} rows) in {:.1f} seconds.".format(log.shape[0], timelib.time()-start_time))
# start_time = timelib.time()
# print("        Loading "+session_filename+" ...", end="\r")
# sessions = pd.read_csv(session_filename, sep=',')
# print("        "+session_filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
# sessions.fillna(0, inplace=True)
pathlib.Path("Latex").mkdir(parents=True, exist_ok=True)
pathlib.Path("Graphs").mkdir(parents=True, exist_ok=True)
pathlib.Path("Clusters").mkdir(parents=True, exist_ok=True)

# normalization
# normalized_dimensions = []
# print("\n   * Normalizing dimensions ...", end="\r")
# for p in dimensions:
#     normalized_dimensions.append("normalized_"+p)
#     if p in lognorm:
#         sessions["normalized_"+p] = log_normalize(sessions[p])
#     else:
#         sessions["normalized_"+p] = normalize(sessions[p])
# print("   * Dimensions normalized in %.1f seconds." %(timelib.time()-start_time))

main_menu()