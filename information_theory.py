#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:48:14 2018

@author: pedro
"""
import numpy as np

def uniform(m):
    return np.ones(m)/m;

def Richness(P):
    return P[P>1e-10].shape[0];

def ShannonEntropy(P,normalize=False):
    if normalize:
        P=P/P.sum()
    P=P[P>1e-20]
    return -np.sum(P*np.log2(P));

def KullbackLeibler(P,Q):
    return 0.5*np.sum(P*np.log2(P/Q))+0.5*np.sum(Q*np.log2(Q/P));

def JensenShannon(P,Q):
    M = 0.5*(P+Q)
    return 0.5*KullbackLeibler(P,M)+0.5*KullbackLeibler(Q,M);

def total_variation(P,Q):
    return 0.5*np.abs(P-Q).sum();