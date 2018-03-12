#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:03:47 2018

@author: pedro
"""
import pandas as pd
import numpy as np
from scipy.linalg import eig 
from information_theory import *

##################################
##################################
##################################
####                          ####
#### MARKOV                   ####
####                          ####
##################################
##################################

def mean_distribution(matrix):
    return matrix.sum(axis=0)/matrix.shape[0];

def js_radius(matrix):
    radius=0.0
    for i in range(0,matrix.shape[0]-1):
        for j in range(i+1,matrix.shape[0]):
            P=matrix[i,:]
            Q=matrix[j,:]
            if JensenShannon(P,Q)>radius:
                radius=JensenShannon(P,Q)
    return radius;
    
def tv_radius(matrix):
    radius=0.0
    for i in range(0,matrix.shape[0]-1):
        for j in range(i+1,matrix.shape[0]):
            P=matrix[i,:]
            Q=matrix[j,:]
            if total_variation(P,Q)>radius:
                radius=total_variation(P,Q)
    return radius;

def mean_tv_radius(matrix):
    radius=0.0
    counter=0
    for i in range(0,matrix.shape[0]-1):
        for j in range(i+1,matrix.shape[0]):
            counter+=1
            P=matrix[i,:]
            Q=matrix[j,:]
            radius+=total_variation(P,Q)
    return radius/counter;


def fit_distribution(reference_labels,labels,distribution):
    
    new_distribution=np.zeros((len(reference_labels)))
    
    for label in reference_labels:
        if label in labels:
            new_index=reference_labels.index(label)
            current_index=labels.index(label)
            new_distribution[new_index]=distribution[current_index]
    distribution_sum=new_distribution.sum()
    if distribution_sum<1e-8:
#        print("(fit_distribution)ERROR: distribution is zero, hence it was not normalized.")
        pass
    else:
        new_distribution = new_distribution / distribution_sum
    return new_distribution;
        
        

def stationary_distribution(matrix):
    S, U = eig(matrix.T)
    stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
#    if stationary.sum()<1e-8:
#        print("(stationary_distribution)ERROR: division by zero attempted.")
    stationary = stationary / np.sum(stationary);
#    if np.abs(np.imag(stationary).sum())>1e-8:
#        print("(stationary_distribution)ERROR: stationary distribution has imaginary part.")
#    if np.abs(stationary.sum()-1)>1e-8:
#        print("(stationary_distribution)ERROR: stationary distribution does not sum up to one.")
    return np.real(stationary);

#def table_to_matrix(table):
#    
#    # table order convention for each row: referrer_label, requested_label
#    
#    # Retrieving the list of labels (requested and referrer)
#    list_of_labels=list(set(table.iloc[:,0].unique())|set(table.iloc[:,1].unique()))
#    number_of_labels=len(list_of_labels)
#    
#    # The Markov matrix     
#    matrix=np.zeros((number_of_labels,number_of_labels))
#    
#    # Adding transitions to the matrix
#    for table_row in range(0,table.shape[0]):
#        matrix_i=list_of_labels.index(table.iloc[table_row,0])
#        matrix_j=list_of_labels.index(table.iloc[table_row,1])
#        
#        matrix[matrix_i,matrix_j] += 1
#    
#    # Row-normalizing the matrix
#    non_stochastic_matrix=False
#    for i in range(0,number_of_labels):
#        row_sum=matrix[i,:].sum()
#        if row_sum<1e-6:
#            non_stochastic_matrix=True
#            print("WARNING: no transitions from %s"%list_of_labels[i])
#        else:
#            matrix[i,:]=matrix[i,:]/row_sum
#    if non_stochastic_matrix:
#        print("(table_to_matrix) WARNING: Matrix is non-stochastic. Some rows are zero and were not normalized.")
#    return number_of_labels,list_of_labels,matrix;
#
#def table_to_matrix_with_endstate(table):
#    
#    # table order convention for each row: referrer_label, requested_label
#    
#    # Retrieving the list of labels (requested and referrer)
#    list_of_labels=list(set(table.iloc[:,0].unique())|set(table.iloc[:,1].unique())|set(['end']))
#    number_of_labels=len(list_of_labels)
#    # The Markov matrix     
#    matrix=np.zeros((number_of_labels,number_of_labels))
#    # Adding transitions to the matrix
#    for table_row in range(0,table.shape[0]):
#        matrix_i=list_of_labels.index(table.iloc[table_row,0])
#        matrix_j=list_of_labels.index(table.iloc[table_row,1])    
#        matrix[matrix_i,matrix_j] += 1
#    # Row-normalizing the matrix, sending leaves to 'end' state, and looping 'end' to itself
#    end_index=list_of_labels.index('end')
#    matrix[end_index,end_index]=1.0
#    for i in range(0,number_of_labels):
#        row_sum=matrix[i,:].sum()
#        if row_sum<1e-6:
#            matrix[i,end_index]=1.0
#        else:
#            matrix[i,:]=matrix[i,:]/row_sum
#    return number_of_labels,list_of_labels,matrix;

def table_to_matrix_with_extstate(table):
    
    # table order convention for each row: referrer_label, requested_label
    
    # Retrieving the list of labels (requested and referrer)
    list_of_labels=list(set(table.iloc[:,0].unique())|set(table.iloc[:,1].unique())|set(['ext']))
    list_of_zero_entry_degree=list(set(table.iloc[:,0].unique())-set(table.iloc[:,1].unique()))
    number_of_labels=len(list_of_labels)
    ext_index=list_of_labels.index('ext')
    # The Markov matrix     
    matrix=np.zeros((number_of_labels,number_of_labels))
    # Adding transitions to the matrix
    for table_row in range(0,table.shape[0]):
        matrix_i=list_of_labels.index(table.iloc[table_row,0])
        matrix_j=list_of_labels.index(table.iloc[table_row,1])    
        matrix[matrix_i,matrix_j] += 1
    for state in list_of_zero_entry_degree:
        state_index=list_of_labels.index(state)
        matrix[ext_index,state_index] += 1
    # Row-normalizing the matrix, sending leaves to 'end' state, and looping 'end' to itself    
    matrix[ext_index,ext_index]=1.0
    for i in range(0,number_of_labels):
        row_sum=matrix[i,:].sum()
        if row_sum<1e-6:
            matrix[i,ext_index]=1.0
        else:
            matrix[i,:]=matrix[i,:]/row_sum
    return number_of_labels,list_of_labels,matrix;