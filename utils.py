# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:03:34 2020

@author: user
"""

import os
import neuroml.loaders as loaders
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset
from matplotlib import cm
import matplotlib.patches as mpatches
import seaborn
import pandas as pd
import scipy.optimize
from collections import Counter
import networkx as nx
import copy
import time


def sortPhysLoc(morph_dist):
    physLoc = np.empty(len(morph_dist))
    
    for i in range(len(morph_dist)):
        if (np.array(morph_dist[i])[:,1] < -200).all():
            physLoc[i] = 0
        elif (np.array(morph_dist[i])[:,1] > 200).all():
            physLoc[i] = 2
        else:
            physLoc[i] = 1

    return physLoc