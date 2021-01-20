# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:32:04 2019

@author: user
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import cm
import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost
import matplotlib.patches as mpatches
from scipy.spatial.transform import Rotation
import seaborn
import pandas as pd
import scipy.optimize
from sklearn import neighbors
from collections import Counter
import multiprocessing as mp
import copy
import time

os.chdir(os.path.dirname(__file__))

import utils

class Parameter:

    PATH = r'./TEMCA2/Skels connectome_mod'
    
    RUN = True
    SAVE = False
    PLOT = True
    numScaleSample = 1000
    numBranchSample = 10
    RN = '1'
    
    sSize = 100
    nSize = [10, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
    dSize = 100
    
    SEED = 1234
    
    outputdir = './output_TEMCA2/RN_' + str(RN)

fp = [f for f in os.listdir(Parameter.PATH) if os.path.isfile(os.path.join(Parameter.PATH, f))]
fp = [os.path.join(Parameter.PATH, f) for f in fp]

fp.pop(17)
#fp = fp[:7]

class MorphData():
    
    def __init__(self):
        self.morph_id = []
        self.morph_parent = []
        self.morph_dist = []
        self.neuron_id = []
        self.endP = []
        self.somaP = []
        self.indMorph_dist = []
        self.calyxdist = []
        self.calyxdist_trk = []
        self.calyxdist_per_n = []
        self.LHdist = []
        self.LHdist_trk = []
        self.LHdist_per_n = []
        self.ALdist = []
        self.ALdist_trk = []
        self.ALdist_per_n = []
    
    def plotNeuronFromPoints(self, listOfPoints, scale=False, showPoint=False):
        """
        plot 3-D neuron morphology plot using a list of coordinates.
        
        :param listOfPoints: List of 3-D coordinates
        :param showPoint: Flag to visualize points
        """
        
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        if scale:
            ax.set_xlim(400, 600)
            ax.set_ylim(400, 150)
            ax.set_zlim(50, 200)
        cmap = cm.get_cmap('viridis', len(listOfPoints))
        for f in range(len(listOfPoints)-1):
    #        tararr = np.array(morph_dist[f])
    #        somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(f))
            if showPoint:
                ax.scatter3D(listOfPoints[f][0], listOfPoints[f][1], listOfPoints[f][2], color=cmap(f), marker='x')
    #        ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(f))
        
        maxval = np.max(np.array(listOfPoints)[:,1])
        minval = np.min(np.array(listOfPoints)[:,1])
        ax.set_ylim(maxval, minval)
        plt.show()
        
    
    def plotAllNeuron(self, showPoint=False):
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        ax.set_xlim(400, 600)
        ax.set_ylim(400, 150)
        ax.set_zlim(50, 200)
        cmap = cm.get_cmap('viridis', len(self.morph_id))
        for f in range(len(self.morph_id)):
            tararr = np.array(self.morph_dist[f])
            somaIdx = np.where(np.array(self.morph_parent[f]) < 0)[0]
            for p in range(len(self.morph_parent[f])):
                if self.morph_parent[f][p] < 0:
                    pass
                else:
                    morph_line = np.vstack((self.morph_dist[f][self.morph_id[f].index(self.morph_parent[f][p])], self.morph_dist[f][p]))
                    ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(f))
                    if showPoint:
                        ax.scatter3D(self.morph_dist[f][p][0], self.morph_dist[f][p][1], self.morph_dist[f][p][2], color=cmap(f), marker='x')
            ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(f))
        plt.show()
        

    def plotNeuronFromListPoints(self, multListOfPoints, scale=False, showPoint=False, axisLabel=True, save=None):
        """
        plot 3-D neuron morphology plot using a list of coordinates.
        
        :param listOfPoints: List of 3-D coordinates
        :param showPoint: Flag to visualize points
        """
        
        maxval = 0
        minval = 0
        
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        if scale:
            ax.set_xlim(400, 600)
            ax.set_ylim(400, 150)
            ax.set_zlim(50, 200)
        cmap = cm.get_cmap('viridis', len(multListOfPoints))
        for i in range(len(multListOfPoints)):
            listOfPoints = multListOfPoints[i]
            for f in range(len(listOfPoints)-1):
        #        tararr = np.array(morph_dist[f])
        #        somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
                morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i))
                if showPoint:
                    ax.scatter3D(listOfPoints[f][0], listOfPoints[f][1], listOfPoints[f][2], color=cmap(i), marker='x')
        #        ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(f))
            maxval_i = np.max(np.array(listOfPoints)[:,1])
            minval_i = np.min(np.array(listOfPoints)[:,1])
            if maxval_i > maxval:
                maxval = maxval_i
            if minval_i < minval:
                minval = minval_i
        if not axisLabel:
            ax.grid(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        
        ax.set_ylim(maxval, minval)
        
        if save != None:
            plt.savefig(os.path.join(Parameter.outputdir, save), dpi=300, bbox_inches='tight')
        plt.show()
        
        
    def plotNeuron(self, idx, scale=False, cmass=False, showPoint=False, lw=1, label=True, show=True, save=False):
        """
        plot 3-D neuron morphology of a single neuron.
        
        :param idx: an index of a list of indices of neurons in MorphData object
        :param showPoint: Flag to visualize points
        """
        
        maxval = 0
        minval = 0
        
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        if scale:
            ax.set_xlim(400, 600)
            ax.set_ylim(400, 150)
            ax.set_zlim(50, 200)
        cmap = cm.get_cmap('viridis', len(self.morph_id))
        
        if cmass:
            cMass = np.sum(self.morph_dist[idx], axis=0)/len(self.morph_dist[idx])
            ax.scatter3D(cMass[0], cMass[1], cMass[2])
        
        if type(idx) == list or type(idx) == np.ndarray:
            for i in idx:
                tararr = np.array(self.morph_dist[i])
                somaIdx = np.where(np.array(self.morph_parent[i]) < 0)[0]
                for p in range(len(self.morph_parent[i])):
                    if self.morph_parent[i][p] < 0:
                        pass
                    else:
                        morph_line = np.vstack((self.morph_dist[i][self.morph_id[i].index(self.morph_parent[i][p])], self.morph_dist[i][p]))
                        ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=lw)
                        if showPoint:
                            ax.scatter3D(self.morph_dist[i][p][0], self.morph_dist[i][p][1], self.morph_dist[i][p][2], color=cmap(i), marker='x')
                ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(i))
                maxval_i = np.max(np.array(self.morph_dist[i])[:,1])
                minval_i = np.min(np.array(self.morph_dist[i])[:,1])
                if maxval_i > maxval:
                    maxval = maxval_i
                if minval_i < minval:
                    minval = minval_i
        else:
            tararr = np.array(self.morph_dist[idx])
            somaIdx = np.where(np.array(self.morph_parent[idx]) < 0)[0]
            for p in range(len(self.morph_parent[idx])):
                if self.morph_parent[idx][p] < 0:
                    pass
                else:
                    morph_line = np.vstack((self.morph_dist[idx][self.morph_id[idx].index(self.morph_parent[idx][p])], self.morph_dist[idx][p]))
                    ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(idx), lw=lw)
                    if showPoint:
                        ax.scatter3D(self.morph_dist[idx][p][0], self.morph_dist[idx][p][1], self.morph_dist[idx][p][2], color=cmap(idx), marker='x')
            ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(idx))
            maxval = np.max(np.array(self.morph_dist[idx])[:,1])
            minval = np.min(np.array(self.morph_dist[idx])[:,1])
        if label:
            plt.title(np.array(self.neuron_id)[idx], fontsize=15)
        ax.set_ylim(maxval, minval)
        if save:
            plt.savefig(Parameter.outputdir + '/neuron_' + str(idx) + '.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
            
    def plotProjection(self, idx, project='z', scale=False, customBound=None, lw=1, label=True, show=True, save=False):
        if project != 'x' and project != 'y' and project != 'z':
            raise(Exception("Unrecognized plane to project"))
        fig = plt.figure(figsize=(24, 16))
        if scale:
            if project == 'z':
                plt.xlim(350, 600)
                plt.ylim(150, 400)
            elif project == 'y':
                plt.xlim(350, 600)
                plt.ylim(0, 200)
            else:
                plt.xlim(150, 400)
                plt.ylim(0, 200)
        
        if customBound != None:
            plt.xlim(customBound[0][0], customBound[0][1])
            plt.ylim(customBound[1][0], customBound[1][1])
        
        cmap = cm.get_cmap('viridis', len(self.morph_id))
        if type(idx) == list or type(idx) == np.ndarray:
            for i in idx:
                tararr = np.array(self.morph_dist[i])
                somaIdx = np.where(np.array(self.morph_parent[i]) < 0)[0]
                for p in range(len(self.morph_parent[i])):
                    if self.morph_parent[i][p] < 0:
                        pass
                    else:
                        morph_line = np.vstack((self.morph_dist[i][self.morph_id[i].index(self.morph_parent[i][p])], self.morph_dist[i][p]))
                        if project == 'z':
                            plt.plot(morph_line[:,0], morph_line[:,1], color=cmap(i), lw=lw)
                        elif project == 'y':
                            plt.plot(morph_line[:,0], morph_line[:,2], color=cmap(i), lw=lw)
                        elif project == 'x':
                            plt.plot(morph_line[:,1], morph_line[:,2], color=cmap(i), lw=lw)
                if project == 'z':
                    plt.scatter(tararr[somaIdx,0], tararr[somaIdx,1], color=cmap(i))
                elif project == 'y':
                    plt.scatter(tararr[somaIdx,0], tararr[somaIdx,2], color=cmap(i))
                elif project == 'x':
                    plt.scatter(tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(i))
        else:
            tararr = np.array(self.morph_dist[idx])
            somaIdx = np.where(np.array(self.morph_parent[idx]) < 0)[0]
            for p in range(len(self.morph_parent[idx])):
                if self.morph_parent[idx][p] < 0:
                    pass
                else:
                    morph_line = np.vstack((self.morph_dist[idx][self.morph_id[idx].index(self.morph_parent[idx][p])], self.morph_dist[idx][p]))
                    if project == 'z':
                        plt.plot(morph_line[:,0], morph_line[:,1], color=cmap(idx), lw=lw)
                    elif project == 'y':
                        plt.plot(morph_line[:,0], morph_line[:,2], color=cmap(idx), lw=lw)
                    elif project == 'x':
                        plt.plot(morph_line[:,1], morph_line[:,2], color=cmap(idx), lw=lw)
            if project == 'z':
                plt.scatter(tararr[somaIdx,0], tararr[somaIdx,1], color=cmap(idx))
            elif project == 'y':
                plt.scatter(tararr[somaIdx,0], tararr[somaIdx,2], color=cmap(idx))
            elif project == 'x':
                plt.scatter(tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(idx))
        if label:
            plt.title(np.array(self.neuron_id)[idx], fontsize=15)
        plt.gca().invert_yaxis()
        if save:
            plt.savefig(Parameter.outputdir + '/neuron_proj_' + str(idx) + '.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    
    def _spat_pos(self, nodeList):
        pos = {}
        
        for i in range(len(nodeList)):
            pos[nodeList[i]] = np.array([self.morph_dist[i][self.somaP[i]][1],
                                         self.morph_dist[i][self.somaP[i]][0]])
    
        return pos
        
def objFuncP(xdata, a, b):
    y = a*np.power(xdata, b)
    
    return y
    
def objFuncPL(xdata, a):
    y = np.power(xdata, a)
    
    return y

def objFuncPpow(xdata, a, b):
    y = np.power(10, b)*np.power(xdata, a)
    
    return y

def objFuncL(xdata, a):
    y = a*xdata
    
    return y

def objFuncGL(xdata, a, b):
    y = a*xdata + b
    
    return y


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return arr.mean(axis=-1)

def tolerant_std(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return arr.std(axis=-1)

def tolerant_std_error(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return np.divide(arr.std(axis=-1), np.sqrt(len(arrs)))

#%%
    
class LengthData:
    length_total = np.empty(len(fp))
    length_branch = []
    length_direct = []
    indMDistLen = []
    indMDistN = []
    length_calyx = []
    length_LH = []
    length_AL = []
    length_calyx_total = []
    length_LH_total = []
    length_AL_total = []
    
class BranchData:
    branchTrk = []
    branch_dist = []
    indBranchTrk = []
    branchP = []
    calyx_branchTrk = []
    calyx_branchP = []
    LH_branchTrk = []
    LH_branchP = []
    AL_branchTrk = []
    AL_branchP = []
    branchNum = np.empty(len(fp))

class OutputData:
    rGySeg = None
    cMLSeg = None
    segOrdN = None
    randTrk = None
    

np.random.seed(Parameter.SEED)

MorphData = MorphData()

t0 = time.time()

indMorph_dist_p_us = []
indMorph_dist_id = []

r_d_x = -10
r_rad_x = np.radians(r_d_x)
r_x = np.array([0, 1, 0])
r_vec_x = r_rad_x * r_x
rotx = Rotation.from_rotvec(r_vec_x)

r_d_y = -25
r_rad_y = np.radians(r_d_y)
r_y = np.array([0, 1, 0])
r_vec_y = r_rad_y * r_y
roty = Rotation.from_rotvec(r_vec_y)

r_d_z = -40
r_rad_z = np.radians(r_d_z)
r_z = np.array([0, 1, 0])
r_vec_z = r_rad_z * r_z
rotz = Rotation.from_rotvec(r_vec_z)

for f in range(len(fp)):
    print(f, fp[f])
    morph_neu_id = []
    morph_neu_parent = []
    morph_neu_prox = []
    morph_neu_dist = []
    
    df = pd.read_csv(fp[f], delimiter=' ', header=None)
    
    MorphData.neuron_id.append(os.path.basename(fp[f]).split('.')[0])
    
    scall = int(df.iloc[np.where(df[6] == -1)[0]].values[0][0])
    
    MorphData.morph_id.append(df[0].tolist())
    MorphData.morph_parent.append(df[6].tolist())
    MorphData.morph_dist.append(np.divide(np.array(df[[2,3,4]]), 1000).tolist()) # Scale
    ctr = Counter(df[6].tolist())
    ctrVal = list(ctr.values())
    ctrKey = list(ctr.keys())
    BranchData.branchNum[f] = sum(i > 1 for i in ctrVal)
    branchInd = np.array(ctrKey)[np.where(np.array(ctrVal) > 1)[0]]
    
    neu_branchTrk = []
    startid = []
    endid = []
    neu_indBranchTrk = []
    branch_dist_temp1 = []
    length_branch_temp = []
    indMorph_dist_temp1 = []
    indMDistLen_temp = []
    
    list_end = np.setdiff1d(MorphData.morph_id[f], MorphData.morph_parent[f])
    
    BranchData.branchP.append(branchInd.tolist())
    MorphData.endP.append(list_end)
    bPoint = np.append(branchInd, list_end)
    
    calyxdist_per_n_temp = []
    LHdist_per_n_temp = []
    ALdist_per_n_temp = []
    length_calyx_per_n = []
    length_LH_per_n = []
    length_AL_per_n = []
    calyx_branchTrk_temp = []
    calyx_branchP_temp = []
    LH_branchTrk_temp = []
    LH_branchP_temp = []
    AL_branchTrk_temp = []
    AL_branchP_temp = []
    
    for bp in range(len(bPoint)):
        if bPoint[bp] != scall:
            neu_branchTrk_temp = []
            branch_dist_temp2 = []
            dist = 0
            
            neu_branchTrk_temp.append(bPoint[bp])
            branch_dist_temp2.append(MorphData.morph_dist[f][MorphData.morph_id[f].index(bPoint[bp])])
            parentTrck = bPoint[bp]
            parentTrck = MorphData.morph_parent[f][MorphData.morph_id[f].index(parentTrck)]
            if parentTrck != -1:
                neu_branchTrk_temp.append(parentTrck)
                rhs = branch_dist_temp2[-1]
                lhs = MorphData.morph_dist[f][MorphData.morph_id[f].index(parentTrck)]
                branch_dist_temp2.append(lhs)
                dist += np.linalg.norm(np.subtract(rhs, lhs))
            while (parentTrck not in branchInd) and (parentTrck != -1):
                parentTrck = MorphData.morph_parent[f][MorphData.morph_id[f].index(parentTrck)]
                if parentTrck != -1:
                    neu_branchTrk_temp.append(parentTrck)
                    rhs = branch_dist_temp2[-1]
                    lhs = MorphData.morph_dist[f][MorphData.morph_id[f].index(parentTrck)]
                    branch_dist_temp2.append(lhs)
                    dist += np.linalg.norm(np.subtract(rhs, lhs))
                    
            if len(neu_branchTrk_temp) > 1:
                neu_branchTrk.append(neu_branchTrk_temp)
                startid.append(neu_branchTrk_temp[0])
                endid.append(neu_branchTrk_temp[-1])
                branch_dist_temp1.append(branch_dist_temp2)
                length_branch_temp.append(dist)
                
                # rotate -25 degrees on y-axis
                branch_dist_temp2_rot = roty.apply(branch_dist_temp2)
                
                # rotate -35 degrees on x-axis
                branch_dist_temp2_rot2 = rotx.apply(branch_dist_temp2)
                
                # rotate 50 degrees on z-axis
                branch_dist_temp2_rot3 = rotz.apply(branch_dist_temp2)
                
                # if ((np.array(branch_dist_temp2_rot)[:,0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[:,0] < 426.14).all() and
                #     (np.array(branch_dist_temp2_rot)[:,1] > 190.71).all() and (np.array(branch_dist_temp2_rot)[:,1] < 272.91).all() and
                #     (np.array(branch_dist_temp2_rot)[:,2] > 354.95).all() and (np.array(branch_dist_temp2_rot)[:,2] < 399.06).all()):
                if ((np.array(branch_dist_temp2_rot)[:,0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[:,0] < 426.14).all() and
                    (np.array(branch_dist_temp2_rot)[:,1] > 176.68).all() and (np.array(branch_dist_temp2_rot)[:,1] < 272.91).all() and
                    (np.array(branch_dist_temp2_rot3)[:,2] > 434.08).all() and (np.array(branch_dist_temp2_rot3)[:,2] < 496.22).all()):
                    MorphData.calyxdist.append(branch_dist_temp2)
                    MorphData.calyxdist_trk.append(f)
                    calyxdist_per_n_temp.append(branch_dist_temp2)
                    length_calyx_per_n.append(dist)
                    calyx_branchTrk_temp.append(neu_branchTrk_temp)
                    calyx_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                # elif ((np.array(branch_dist_temp2_rot)[:,0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[:,1] > 190.71).all() and
                #       (np.array(branch_dist_temp2_rot)[:,1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[:,2] > 278.76).all() and
                #       (np.array(branch_dist_temp2_rot)[:,2] < 345.93).all()):
                elif ((np.array(branch_dist_temp2_rot)[:,0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[:,1] > 176.68).all() and
                      (np.array(branch_dist_temp2_rot)[:,1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[:,2] > 286.78).all() and
                      (np.array(branch_dist_temp2_rot)[:,2] < 343.93).all()):
                    MorphData.LHdist.append(branch_dist_temp2)
                    MorphData.LHdist_trk.append(f)
                    LHdist_per_n_temp.append(branch_dist_temp2)
                    length_LH_per_n.append(dist)
                    LH_branchTrk_temp.append(neu_branchTrk_temp)
                    LH_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                # elif ((np.array(branch_dist_temp2_rot)[:,0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[:,0] < 533.42).all() and 
                #       (np.array(branch_dist_temp2_rot)[:,1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[:,1] < 361.12).all() and
                #       (np.array(branch_dist_temp2_rot2)[:,2] < -77.84).all()):
                elif ((np.array(branch_dist_temp2_rot)[:,0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[:,0] < 533.42).all() and 
                      (np.array(branch_dist_temp2_rot)[:,1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[:,1] < 363.12).all() and
                      (np.array(branch_dist_temp2_rot2)[:,2] < 180.77).all()):
                    MorphData.ALdist.append(branch_dist_temp2)
                    MorphData.ALdist_trk.append(f)
                    ALdist_per_n_temp.append(branch_dist_temp2)
                    length_AL_per_n.append(dist)
                    AL_branchTrk_temp.append(neu_branchTrk_temp)
                    AL_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                
    BranchData.branchTrk.append(neu_branchTrk)
    BranchData.branch_dist.append(branch_dist_temp1)
    LengthData.length_branch.append(length_branch_temp)
    
    MorphData.calyxdist_per_n.append(calyxdist_per_n_temp)
    MorphData.LHdist_per_n.append(LHdist_per_n_temp)
    MorphData.ALdist_per_n.append(ALdist_per_n_temp)
    LengthData.length_calyx.append(length_calyx_per_n)
    LengthData.length_LH.append(length_LH_per_n)
    LengthData.length_AL.append(length_AL_per_n)
    BranchData.calyx_branchTrk.append(calyx_branchTrk_temp)
    BranchData.calyx_branchP.append(np.unique([item for sublist in calyx_branchP_temp for item in sublist]).tolist())
    BranchData.LH_branchTrk.append(LH_branchTrk_temp)
    BranchData.LH_branchP.append(np.unique([item for sublist in LH_branchP_temp for item in sublist]).tolist())
    BranchData.AL_branchTrk.append(AL_branchTrk_temp)
    BranchData.AL_branchP.append(np.unique([item for sublist in AL_branchP_temp for item in sublist]).tolist())

    for ep in range(len(list_end)):
        neu_indBranchTrk_temp = []
        indMorph_dist_temp2 = []
        
        startidind = startid.index(list_end[ep])
        neu_indBranchTrk_temp.append(BranchData.branchTrk[f][startidind])
        indMorph_dist_temp2.append(BranchData.branch_dist[f][startidind])
        endidval = endid[startidind]
        while endidval != scall:
            startidind = startid.index(endidval)
            neu_indBranchTrk_temp.append(BranchData.branchTrk[f][startidind][1:])
            indMorph_dist_temp2.append(BranchData.branch_dist[f][startidind][1:])
            endidval = endid[startidind]
            
        if len(neu_indBranchTrk_temp) > 1:
            neu_indBranchTrk.append([item for sublist in neu_indBranchTrk_temp for item in sublist])
            indMorph_dist_p_us.append(1/len(neu_indBranchTrk))
            val = [item for sublist in indMorph_dist_temp2 for item in sublist]
            indMorph_dist_temp1.append(val)
            indMorph_dist_id.append(f)
            
            x = np.array(val)[:,0]
            y = np.array(val)[:,1]
            z = np.array(val)[:,2]
            xd = [j-i for i, j in zip(x[:-1], x[1:])]
            yd = [j-i for i, j in zip(y[:-1], y[1:])]
            zd = [j-i for i, j in zip(z[:-1], z[1:])]
            indMDistLen_temp.append(np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd))))
        
    BranchData.indBranchTrk.append(neu_indBranchTrk)
    MorphData.indMorph_dist.append(indMorph_dist_temp1)
    LengthData.indMDistLen.append(indMDistLen_temp)

t1 = time.time()

print('checkpoint 1: ' + str(t1-t0))


#%%

if Parameter.RUN:
    BranchData.indMorph_dist_p_us = np.array(indMorph_dist_p_us)
    MorphData.endP_len = [len(arr) for arr in MorphData.endP]
else:
    (BranchData.branchTrk, BranchData.branch_dist, BranchData.indBranchTrk, 
     MorphData.indMorph_dist, MorphData.indMorph_dist_p_us, LengthData.length_branch) = utils.importMorph(Parameter)


MorphData.indMorph_dist_flat = [item for sublist in MorphData.indMorph_dist for item in sublist]
LengthData.indMDistN = np.array([len(arr) for arr in MorphData.indMorph_dist_flat])

LengthData.length_branch_flat = [item for sublist in LengthData.length_branch for item in sublist]
LengthData.length_average = np.empty(len(fp))

for lb in range(len(LengthData.length_branch)):
    LengthData.length_total[lb] = np.sum(LengthData.length_branch[lb])
    LengthData.length_average[lb] = np.average(LengthData.length_branch[lb])

MorphData.morph_dist_len = np.array([len(arr) for arr in MorphData.morph_dist])
MorphData.morph_dist_flat = np.array([item for sublist in MorphData.morph_dist for item in sublist])
#MorphData.morph_dist_len_EP = np.empty((len(MorphData.morph_dist_len)))


t2 = time.time()

print('checkpoint 2: ' + str(t2-t1))

# (MorphData.regMDist, MorphData.regMDistLen) = utils.segmentMorph(Parameter, BranchData)
#(MorphData.indRegMDist, MorphData.indMDistN) = utils.indSegmentMorph(Parameter, BranchData)


(rGy, cML) = utils.radiusOfGyration(MorphData.morph_dist)

# (rGyEP, cMLEP) = utils.endPointRadiusOfGyration(MorphData, BranchData)

t3 = time.time()

print('checkpoint 3: ' + str(t3-t2))

# (rGyReg, cMLReg) = utils.regularRadiusOfGyration(MorphData.regMDist, MorphData.regMDistLen)

t4 = time.time()

print('checkpoint 4: ' + str(t4-t3))

if Parameter.RUN:
    # (OutputData.rGySeg, 
    #  OutputData.cMLSeg, 
    #  OutputData.segOrdN, 
    #  OutputData.randTrk) = utils.regularSegmentRadiusOfGyration(Parameter,
    #                                                              BranchData,
    #                                                              np.array(MorphData.indMorph_dist_flat), 
    #                                                              LengthData.indMDistN, 
    #                                                              numScaleSample=Parameter.numScaleSample,
    #                                                              stochastic=True,
    #                                                              p=indMorph_dist_id)
    if Parameter.SAVE:
        utils.exportMorph(Parameter, t4-t0, MorphData, BranchData, LengthData)

calyx_dist_flat = np.array([item for sublist in MorphData.calyxdist for item in sublist])
LH_dist_flat = np.array([item for sublist in MorphData.LHdist for item in sublist])
AL_dist_flat = np.array([item for sublist in MorphData.ALdist for item in sublist])

calyxCM = (np.sum(calyx_dist_flat, axis=0)/len(calyx_dist_flat))
LHCM = (np.sum(LH_dist_flat, axis=0)/len(LH_dist_flat))
ALCM = (np.sum(AL_dist_flat, axis=0)/len(AL_dist_flat))

fullCM = cML#np.average(OutputData.cMLSeg, axis=0)

xmax_calyx = np.max(calyx_dist_flat[:,0])
xmin_calyx = np.min(calyx_dist_flat[:,0])
ymax_calyx = np.max(calyx_dist_flat[:,1])
ymin_calyx = np.min(calyx_dist_flat[:,1])
zmax_calyx = np.max(calyx_dist_flat[:,2])
zmin_calyx = np.min(calyx_dist_flat[:,2])

xmax_LH = np.max(LH_dist_flat[:,0])
xmin_LH = np.min(LH_dist_flat[:,0])
ymax_LH = np.max(LH_dist_flat[:,1])
ymin_LH = np.min(LH_dist_flat[:,1])
zmax_LH = np.max(LH_dist_flat[:,2])
zmin_LH = np.min(LH_dist_flat[:,2])

xmax_AL = np.max(AL_dist_flat[:,0])
xmin_AL = np.min(AL_dist_flat[:,0])
ymax_AL = np.max(AL_dist_flat[:,1])
ymin_AL = np.min(AL_dist_flat[:,1])
zmax_AL = np.max(AL_dist_flat[:,2])
zmin_AL = np.min(AL_dist_flat[:,2])

# Find the BP closest to the CM
calyxcent_bp = np.array([0,0,0])

for nidx in range(len(BranchData.calyx_branchP)):
    for bidx in range(len(BranchData.calyx_branchP[nidx])):
        calyxcent_temp = MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(BranchData.calyx_branchP[nidx][bidx])]
        
        if np.linalg.norm(np.subtract(calyxCM, calyxcent_temp)) < np.linalg.norm(np.subtract(calyxCM, calyxcent_bp)):
            calyxcent_bp = calyxcent_temp
        
LHcent_bp = np.array([0,0,0])

for nidx in range(len(BranchData.LH_branchP)):
    for bidx in range(len(BranchData.LH_branchP[nidx])):
        LHcent_temp = MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(BranchData.LH_branchP[nidx][bidx])]
        
        if np.linalg.norm(np.subtract(LHCM, LHcent_temp)) < np.linalg.norm(np.subtract(LHCM, LHcent_bp)):
            LHcent_bp = LHcent_temp
        
ALcent_bp = np.array([0,0,0])

for nidx in range(len(BranchData.AL_branchP)):
    for bidx in range(len(BranchData.AL_branchP[nidx])):
        ALcent_temp = MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(BranchData.AL_branchP[nidx][bidx])]
        
        if np.linalg.norm(np.subtract(ALCM, ALcent_temp)) < np.linalg.norm(np.subtract(ALCM, ALcent_bp)):
            ALcent_bp = ALcent_temp

# Find the point closest to the CM
calyxcent_npl = np.array([0,0,0])

for nidx in range(len(MorphData.calyxdist)):
    for bidx in range(len(MorphData.calyxdist[nidx])):
        if np.linalg.norm(np.subtract(calyxCM, MorphData.calyxdist[nidx][bidx])) < np.linalg.norm(np.subtract(calyxCM, calyxcent_npl)):
            calyxcent_npl = MorphData.calyxdist[nidx][bidx]
        
LHcent_npl = np.array([0,0,0])

for nidx in range(len(MorphData.LHdist)):
    for bidx in range(len(MorphData.LHdist[nidx])):
        if np.linalg.norm(np.subtract(LHCM, MorphData.LHdist[nidx][bidx])) < np.linalg.norm(np.subtract(LHCM, LHcent_npl)):
            LHcent_npl = MorphData.LHdist[nidx][bidx]
        
ALcent_npl = np.array([0,0,0])

for nidx in range(len(MorphData.ALdist)):
    for bidx in range(len(MorphData.ALdist[nidx])):
        if np.linalg.norm(np.subtract(ALCM, MorphData.ALdist[nidx][bidx])) < np.linalg.norm(np.subtract(ALCM, ALcent_npl)):
            ALcent_npl = MorphData.ALdist[nidx][bidx]

t5 = time.time()

print('checkpoint 5: ' + str(t5-t4))


#%% Neuropil Segmentation

from scipy.signal import argrelextrema

r_d_x = -30
r_rad_x = np.radians(r_d_x)
r_x = np.array([1, 0, 0])

r_d_y = 0#-25
r_rad_y = np.radians(r_d_y)
r_y = np.array([0, 1, 0])

r_d_z = 0
r_rad_z = np.radians(r_d_z)
r_z = np.array([0, 0, 1])

r_vec_x = r_rad_x * r_x
rotx = Rotation.from_rotvec(r_vec_x)
morph_dist_flat_rot = rotx.apply(MorphData.morph_dist_flat)
calyxdist_flat_rot = rotx.apply(calyx_dist_flat)
LHdist_flat_rot = rotx.apply(LH_dist_flat)
ALdist_flat_rot = rotx.apply(AL_dist_flat)

r_vec_y = r_rad_y * r_y
roty = Rotation.from_rotvec(r_vec_y)
morph_dist_flat_rot = roty.apply(morph_dist_flat_rot)
# calyxdist_flat_rot = roty.apply(calyxdist_flat)
# LHdist_flat_rot = roty.apply(LHdist_flat)
# ALdist_flat_rot = roty.apply(ALdist_flat)

r_vec_z = r_rad_z * r_z
rotz = Rotation.from_rotvec(r_vec_z)
morph_dist_flat_rot = rotz.apply(morph_dist_flat_rot)

x = np.histogram(morph_dist_flat_rot[:,0], bins=int((np.max(morph_dist_flat_rot[:,0]) - np.min(morph_dist_flat_rot[:,0]))/1))
y = np.histogram(morph_dist_flat_rot[:,1], bins=int((np.max(morph_dist_flat_rot[:,1]) - np.min(morph_dist_flat_rot[:,1]))/1))
z = np.histogram(morph_dist_flat_rot[:,2], bins=int((np.max(morph_dist_flat_rot[:,2]) - np.min(morph_dist_flat_rot[:,2]))/1))

xex = argrelextrema(x[0], np.less)[0]
yex = argrelextrema(y[0], np.less)[0]
zex = argrelextrema(z[0], np.less)[0]

fig = plt.figure(figsize=(8,6))
plt.hist(morph_dist_flat_rot[:,0], 
         bins=int((np.max(morph_dist_flat_rot[:,0]) - np.min(morph_dist_flat_rot[:,0]))/1), 
         color='tab:purple', alpha=0.5)
plt.hist(np.array(ALdist_flat_rot)[:,0], 
         bins=int((np.max(np.array(ALdist_flat_rot)[:,0]) - np.min(np.array(ALdist_flat_rot)[:,0]))/1), 
         color='tab:blue', alpha=0.5)
plt.hist(np.array(calyxdist_flat_rot)[:,0], 
         bins=int((np.max(np.array(calyxdist_flat_rot)[:,0]) - np.min(np.array(calyxdist_flat_rot)[:,0]))/1), 
         color='tab:orange', alpha=0.5)
plt.hist(np.array(LHdist_flat_rot)[:,0], 
         bins=int((np.max(np.array(LHdist_flat_rot)[:,0]) - np.min(np.array(LHdist_flat_rot)[:,0]))/1), 
         color='tab:green', alpha=0.5)
plt.xlabel('x Coordinates', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.legend(['All', 'AL', 'MB calyx', 'LH'], fontsize=13)
plt.scatter(x[1][xex[[13,27,40]]], x[0][xex[[13,27,40]]], color='tab:red')
# plt.savefig(Parameter.outputdir + '/x_segment_hist_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.hist(morph_dist_flat_rot[:,1],
         bins=int((np.max(morph_dist_flat_rot[:,1]) - np.min(morph_dist_flat_rot[:,1]))/1),
         color='tab:purple', alpha=0.5)
plt.hist(np.array(ALdist_flat_rot)[:,1], 
         bins=int((np.max(np.array(ALdist_flat_rot)[:,1]) - np.min(np.array(ALdist_flat_rot)[:,1]))/1), 
         color='tab:blue', alpha=0.5)
plt.hist(np.array(calyxdist_flat_rot)[:,1], 
         bins=int((np.max(np.array(calyxdist_flat_rot)[:,1]) - np.min(np.array(calyxdist_flat_rot)[:,1]))/1), 
         color='tab:orange', alpha=0.5)
plt.hist(np.array(LHdist_flat_rot)[:,1], 
         bins=int((np.max(np.array(LHdist_flat_rot)[:,1]) - np.min(np.array(LHdist_flat_rot)[:,1]))/1),
         color='tab:green', alpha=0.5)
plt.xlabel('y Coordinates', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.legend(['All', 'AL', 'MB calyx', 'LH'], fontsize=13)
plt.scatter(y[1][yex[[9,26,46]]], y[0][yex[[9,26,46]]], color='tab:red')
# plt.savefig(Parameter.outputdir + '/y_segment_hist_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.hist(morph_dist_flat_rot[:,2], 
         bins=int((np.max(morph_dist_flat_rot[:,2]) - np.min(morph_dist_flat_rot[:,2]))/1), 
         color='tab:purple', alpha=0.5)
plt.hist(np.array(ALdist_flat_rot)[:,2], 
         bins=int((np.max(np.array(ALdist_flat_rot)[:,2]) - np.min(np.array(ALdist_flat_rot)[:,2]))/1),
         color='tab:blue', alpha=0.5)
plt.hist(np.array(calyxdist_flat_rot)[:,2],
         bins=int((np.max(np.array(calyxdist_flat_rot)[:,2]) - np.min(np.array(calyxdist_flat_rot)[:,2]))/1), 
         color='tab:orange', alpha=0.5)
plt.hist(np.array(LHdist_flat_rot)[:,2], 
         bins=int((np.max(np.array(LHdist_flat_rot)[:,2]) - np.min(np.array(LHdist_flat_rot)[:,2]))/1), 
         color='tab:green', alpha=0.5)
plt.xlabel('z Coordinates', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.legend(['All', 'AL', 'MB calyx', 'LH'], fontsize=13)
# plt.scatter(z[1][zex[[7,12,22,24,28]]], z[0][zex[[7,12,22,24,28]]], color='tab:red')
plt.scatter(z[1][zex[[14]]], z[0][zex[[14]]], color='tab:red')
# plt.savefig(Parameter.outputdir + '/z_segment_hist_2_xrot.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%% Neuropil Segmentation Rotation

refcalyx = np.array([512, 221, 171])
refLH = np.array([431, 223, 153])
refAL = np.array([539, 314,  44])

xex_list = []
for i in np.linspace(-45, 0, 10):
    r_d_y = i
    r_rad_y = np.radians(r_d_y)
    r_y = np.array([0, 1, 0])
    r_vec_y = r_rad_y * r_y
    roty = Rotation.from_rotvec(r_vec_y)
    morph_dist_flat_rot = roty.apply(MorphData.morph_dist_flat)
    refcalyx_rot = roty.apply(refcalyx)
    refLH_rot = roty.apply(refLH)
    refAL_rot = roty.apply(refAL)
    
    x = np.histogram(morph_dist_flat_rot[:,0], bins=int((np.max(morph_dist_flat_rot[:,0]) - np.min(morph_dist_flat_rot[:,0]))/1))
       
    xex = argrelextrema(x[0], np.less, order=15)[0]
    xidx = np.intersect1d(np.where(x[0] > 0)[0], xex)#np.where(x[0] > 500)[0], xex)
    LHcallogand = np.where(np.logical_and(x[1][xidx]>refLH_rot[0], x[1][xidx]<refcalyx_rot[0]))[0]
    calALlogand = np.where(np.logical_and(x[1][xidx]>refcalyx_rot[0], x[1][xidx]<refAL_rot[0]))[0]
    if (len(LHcallogand) > 0) and (len(calALlogand) > 0):
        xex_list.append([i, x[1][xidx], np.average(x[0][xidx]), np.std(x[0][xidx[[0,1]]])])
    
    fig = plt.figure(figsize=(8,6))
    plt.hist(morph_dist_flat_rot[:,0], bins=int((np.max(morph_dist_flat_rot[:,0]) - np.min(morph_dist_flat_rot[:,0]))/1), color='tab:purple', alpha=0.5)
    plt.vlines([refcalyx_rot[0], refLH_rot[0], refAL_rot[0]], 0, np.max(x[0]))
    plt.xlabel('x Coordinates', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.legend(['All', 'AL', 'MB calyx', 'LH'], fontsize=13)
    plt.title(str(i))
    plt.scatter(x[1][xidx], x[0][xidx], color='tab:red')
    plt.show()

yex_list = []
for i in np.linspace(0, 45, 10):
    r_d_z = i
    r_rad_z = np.radians(r_d_z)
    r_z = np.array([0, 0, 1])
    r_vec_z = r_rad_z * r_z
    rotz = Rotation.from_rotvec(r_vec_z)
    morph_dist_flat_rot = rotz.apply(MorphData.morph_dist_flat)
    refcalyx_rot = rotz.apply(refcalyx)
    refLH_rot = rotz.apply(refLH)
    refAL_rot = rotz.apply(refAL)
        
    y = np.histogram(morph_dist_flat_rot[:,1], bins=int((np.max(morph_dist_flat_rot[:,1]) - np.min(morph_dist_flat_rot[:,1]))/1))
       
    yex = argrelextrema(y[0], np.less, order=11)[0]
    yidx = np.intersect1d(np.where(y[0] > 0)[0], yex)
    LHcallogand = np.where(np.logical_and(y[1][yidx]>refLH_rot[1], y[1][yidx]<refcalyx_rot[1]))[0]
    calALlogand = np.where(np.logical_and(y[1][yidx]>refcalyx_rot[1], y[1][yidx]<refAL_rot[1]))[0]
    if (len(LHcallogand) > 0) and (len(calALlogand) > 0):
        yex_list.append([i, y[1][yidx], np.average(y[0][yidx]), np.std(y[0][yidx[[0,1]]])])

    fig = plt.figure(figsize=(8,6))
    plt.hist(morph_dist_flat_rot[:,1], bins=int((np.max(morph_dist_flat_rot[:,1]) - np.min(morph_dist_flat_rot[:,1]))/1), color='tab:purple', alpha=0.5)
    plt.vlines([refcalyx_rot[1], refLH_rot[1], refAL_rot[1]], 0, np.max(y[0]))
    plt.xlabel('y Coordinates', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.legend(['All', 'AL', 'MB calyx', 'LH'], fontsize=13)
    plt.title(str(i))
    plt.scatter(y[1][yidx], y[0][yidx], color='tab:red')
    plt.show()

zex_list = []
for i in np.linspace(-45, 0, 10):
    r_d_x = i
    r_rad_x = np.radians(r_d_x)
    r_x = np.array([1, 0, 0])
    r_vec_x = r_rad_x * r_x
    rotx = Rotation.from_rotvec(r_vec_x)
    morph_dist_flat_rot = rotx.apply(MorphData.morph_dist_flat)
    refcalyx_rot = rotx.apply(refcalyx)
    refLH_rot = rotx.apply(refLH)
    refAL_rot = rotx.apply(refAL)
        
    z = np.histogram(morph_dist_flat_rot[:,2], bins=int((np.max(morph_dist_flat_rot[:,2]) - np.min(morph_dist_flat_rot[:,2]))/1))
       
    zex = argrelextrema(z[0], np.less, order=15)[0]
    zidx = np.intersect1d(np.where(z[0] > 0)[0], zex)
    LHcallogand = np.where(np.logical_and(z[1][zidx]<refLH_rot[2], z[1][zidx]>refcalyx_rot[2]))[0]
    calALlogand = np.where(np.logical_and(z[1][zidx]<refcalyx_rot[2], z[1][zidx]>refAL_rot[2]))[0]
    if (len(LHcallogand) > 0) and (len(calALlogand) > 0):
        zex_list.append([i, z[1][zidx], np.average(z[0][zidx]), np.std(z[0][zidx])])
    elif (len(LHcallogand) > 0) and (len(calALlogand) < 1):
        zex_list.append([i, z[1][zidx], np.average(z[0][zidx]), np.std(z[0][zidx])])
    elif (len(LHcallogand) < 1) and (len(calALlogand) > 0):
        zex_list.append([i, z[1][zidx], np.average(z[0][zidx]), np.std(z[0][zidx])])
    
    fig = plt.figure(figsize=(8,6))
    plt.hist(morph_dist_flat_rot[:,2], bins=int((np.max(morph_dist_flat_rot[:,2]) - np.min(morph_dist_flat_rot[:,2]))/1), color='tab:purple', alpha=0.5)
    plt.vlines([refcalyx_rot[2], refLH_rot[2], refAL_rot[2]], 0, np.max(z[0]))
    plt.xlabel('z Coordinates', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.legend(['All', 'AL', 'MB calyx', 'LH'], fontsize=13)
    plt.title(str(i))
    plt.scatter(z[1][zidx], z[0][zidx], color='tab:red')
    plt.show()

zex_list_y_1 = []
zex_list_y_2 = []
for i in np.linspace(-90, 90, 19):
    r_d_y = i
    r_rad_y = np.radians(r_d_y)
    r_y = np.array([0, 1, 0])
    r_vec_y = r_rad_y * r_y
    roty = Rotation.from_rotvec(r_vec_y)
    morph_dist_flat_rot = roty.apply(MorphData.morph_dist_flat)
    refcalyx_rot = roty.apply(refcalyx)
    refLH_rot = roty.apply(refLH)
    refAL_rot = roty.apply(refAL)
    
    z = np.histogram(morph_dist_flat_rot[:,2], bins=int((np.max(morph_dist_flat_rot[:,2]) - np.min(morph_dist_flat_rot[:,2]))/1))
       
    zex = argrelextrema(z[0], np.less, order=15)[0]
    zidx = np.intersect1d(np.where(z[0] > 0)[0], zex)
    LHcallogand = np.where(np.logical_and(z[1][zidx]>refLH_rot[2], z[1][zidx]<refcalyx_rot[2]))[0]
    calALlogand = np.where(np.logical_and(z[1][zidx]<refLH_rot[2], z[1][zidx]>refAL_rot[2]))[0]
    if (len(LHcallogand) > 0):
        zex_list_y_1.append([i, z[1][zidx], np.average(z[0][zidx]), np.std(z[0][zidx])])
    if(len(calALlogand) > 0):
        zex_list_y_2.append([i, z[1][zidx], np.average(z[0][zidx]), np.std(z[0][zidx])])
        
    fig = plt.figure(figsize=(8,6))
    plt.hist(morph_dist_flat_rot[:,2],
             bins=int((np.max(morph_dist_flat_rot[:,2]) - np.min(morph_dist_flat_rot[:,2]))/1),
             color='tab:purple', 
             alpha=0.5)
    plt.vlines([refcalyx_rot[2], refLH_rot[2], refAL_rot[2]], 0, np.max(z[0]))
    plt.xlabel('z Coordinates', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.legend(['All', 'AL', 'MB calyx', 'LH'], fontsize=13)
    plt.title(str(i))
    plt.scatter(z[1][zidx], z[0][zidx], color='tab:red')
    plt.show()


#%%

if Parameter.PLOT:

    fig, ax = plt.subplots(1, 2, figsize=(20,6))
    hist0 = ax[0].hist(LengthData.length_total, 
              bins=int((np.max(LengthData.length_total) - np.min(LengthData.length_total))/100),
              density=True)
    ax[0].set_title("Distribution of Total Length", fontsize=20)
    ax[0].set_ylabel("Normalized Density", fontsize=15)
    ax[0].set_xlabel("Total Length", fontsize=15)
#    ax[0].set_xlim(0, 1000)
    
    hist1 = ax[1].hist(LengthData.length_branch_flat, 
              bins=int((np.max(LengthData.length_branch_flat) - np.min(LengthData.length_branch_flat))/10),
              density=True)
    ax[1].set_title("Distribution of Segment Length", fontsize=20)
    ax[1].set_ylabel("Normalized Density", fontsize=15)
    ax[1].set_xlabel("Segment Length", fontsize=15)
    
    plt.tight_layout()
    plt.show()
    
    
    hist0centers = 0.5*(hist0[1][1:] + hist0[1][:-1])
    hist1centers = 0.5*(hist1[1][1:] + hist1[1][:-1])
    
    
    #==============================================================================
    
    popt0, pcov0 = scipy.optimize.curve_fit(objFuncGL, np.log10(hist0centers[np.nonzero(hist0[0])]), 
                                            np.log10(hist0[0][np.nonzero(hist0[0])]), p0=[0.1, -0.1], maxfev=10000)
    
    popt1, pcov1 = scipy.optimize.curve_fit(objFuncGL, np.log10(hist1centers[np.nonzero(hist1[0])]), 
                                            np.log10(hist1[0][np.nonzero(hist1[0])]), p0=[0.1, -0.1], maxfev=10000)
    
#    popt1, pcov1 = scipy.optimize.curve_fit(objFuncP, hist1centers, hist1[0], p0=[0.1, -0.1], maxfev=10000)
    
    
    fitX = np.linspace(1, 10000, 1000)
    fitY1 = objFuncPpow(fitX, popt1[0], popt1[1])
    
    # Segment Length in Log-Log
    
    fig, ax = plt.subplots(1, 2, figsize=(20,6))
    ax[0].scatter(hist0centers, hist0[0])
    ax[0].set_title("Distribution of Total Length", fontsize=20)
    ax[0].set_ylabel("Normalized Density", fontsize=15)
    ax[0].set_xlabel("Total Length", fontsize=15)
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
#    ax[0].set_xlim(1, 10000)
#    ax[0].set_ylim(0.00001, 0.1)
    ax[0].plot(fitX, fitY1, 'r')
    
    ax[1].scatter(hist1centers, hist1[0])
    ax[1].set_title("Distribution of Segment Length", fontsize=20)
    ax[1].set_ylabel("Normalized Density", fontsize=15)
    ax[1].set_xlabel("Segment Length", fontsize=15)
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
#    ax[1].set_xlim(1, 10000)
#    ax[1].set_ylim(0.00001, 0.1)
    ax[1].plot(fitX, fitY1, 'r')
    plt.tight_layout()
    plt.show()
    
    
    #==============================================================================
    # Average Segment Length Histogram
    
    fig, ax = plt.subplots(1, 2, figsize=(20,6))
    hist9 = ax[0].hist(LengthData.length_average,
              bins=int((np.max(LengthData.length_average) - np.min(LengthData.length_average))),
              density=True)
    ax[0].set_title("Distribution of Average Segment Length", fontsize=20)
    ax[0].set_ylabel("Normalized Density", fontsize=15)
    ax[0].set_xlabel("Segment Length", fontsize=15)

    hist9centers = 0.5*(hist9[1][1:] + hist9[1][:-1])
    
    ax[1].scatter(hist9centers, hist9[0])
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
#    ax[1].set_xlim(1, 10000)
#    ax[1].set_ylim(0.0001, 0.1)
    ax[1].set_title("Distribution of Average Segment Length", fontsize=20)
    ax[1].set_ylabel("Normalized Density", fontsize=15)
    ax[1].set_xlabel("Segment Length", fontsize=15)
    
    plt.tight_layout()
    plt.show()
    
    
    
    #==============================================================================
    # BranchNum vs Total Segment Length vs Average Segment Length by Type
    
    poptL = []
    
    fig, ax = plt.subplots(4, figsize=(9,24))
    ax[0].scatter(LengthData.length_total, BranchData.branchNum)
    ax[0].set_xlabel("Total Length", fontsize=15)
    ax[0].set_ylabel("Number of Branches", fontsize=15)
#    ax[0][0].set_xlim(-50, 1000)
#    ax[0][0].set_ylim(-1, 8)

    ax[1].scatter(LengthData.length_average, BranchData.branchNum)
    ax[1].set_xlabel("Average Segment Length", fontsize=15)
    ax[1].set_ylabel("Number of Branches", fontsize=15)
#    ax[1][0].set_xlim(-50, 1000)
#    ax[1][0].set_ylim(-1, 8)
    
    for i in range(len(np.unique(BranchData.branchNum))):
        scttrInd = np.where(BranchData.branchNum ==
                            np.unique(BranchData.branchNum)[i])[0]
        ax[2].scatter(LengthData.length_average[scttrInd], 
                         LengthData.length_total[scttrInd])
        fitX = np.linspace(0, 10, 10)
    ax[2].set_xlabel("Average Segment Length", fontsize=15)
    ax[2].set_ylabel("Total Length", fontsize=15)
#    ax[2].legend(np.unique(BranchData.branchNum)[:-1], fontsize=15)
    for i in range(len(np.unique(BranchData.branchNum))):
        scttrInd = np.where(BranchData.branchNum == 
                            np.unique(BranchData.branchNum)[i])[0]
        if np.unique(BranchData.branchNum)[i] == 0:
            fitY = objFuncL(fitX, 1)
            ax[2].plot(fitX, fitY)
        elif (np.unique(BranchData.branchNum)[i] == 1 or 
              np.unique(BranchData.branchNum)[i] == 2):
            popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                    LengthData.length_average[scttrInd], 
                                                    LengthData.length_total[scttrInd],
                                                    p0=[1.],
                                                    maxfev=10000)
            fitY = objFuncL(fitX, popt[0])
            ax[2].plot(fitX, fitY)
            poptL.append(popt[0])
#    ax[2][0].set_xlim(-50, 1000)
#    ax[2][0].set_ylim(0, 1000)
    
    length_branch_len = np.array([len(arr) for arr in LengthData.length_branch])
    repeated_length_total = np.repeat(LengthData.length_total, length_branch_len)
    
    for i in range(len(np.unique(BranchData.branchNum))):
        scttrInd = np.where(BranchData.branchNum == 
                            np.unique(BranchData.branchNum)[i])[0]
        length_branch_len_sensory = [len(arr) for arr in np.array(LengthData.length_branch)[scttrInd]]
        repeated_length_total_sensory = np.repeat(LengthData.length_total[scttrInd], 
                                                  length_branch_len[scttrInd])
        ax[3].scatter([item for sublist in np.array(LengthData.length_branch)[scttrInd].tolist() for item in sublist], 
                         repeated_length_total_sensory)
    ax[3].set_xlabel("Segment Length", fontsize=15)
    ax[3].set_ylabel("Total Length", fontsize=15)
#    ax[3].legend(np.unique(BranchData.branchNum)[:-1], fontsize=15)
    for i in range(len(np.unique(BranchData.branchNum))):
        scttrInd = np.where(BranchData.branchNum == 
                            np.unique(BranchData.branchNum)[i])[0]
        length_branch_len_sensory = [len(arr) for arr in np.array(LengthData.length_branch)[scttrInd]]
        repeated_length_total_sensory = np.repeat(LengthData.length_total[scttrInd], 
                                                  length_branch_len[scttrInd])
        if np.unique(BranchData.branchNum)[i] == 0:
            fitY = objFuncL(fitX, 1)
            ax[3].plot(fitX, fitY)
        elif (np.unique(BranchData.branchNum)[i] == 1 or 
            np.unique(BranchData.branchNum)[i] == 2):
            popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                  [item for sublist in 
                                                   np.array(LengthData.length_branch)[scttrInd].tolist() for item in sublist], 
                                                  repeated_length_total_sensory,
                                                  p0=[1.],
                                                  maxfev=10000)
            fitY = objFuncL(fitX, popt[0])
            ax[3].plot(fitX, fitY)
            poptL.append(popt[0])
#    ax[3][0].set_xlim(-50, 1000)
#    ax[3][0].set_ylim(0, 1000)
    
    plt.tight_layout()
    plt.show()
    
    
    #==============================================================================
    
    
    # branchEndPDict = {'branch': BranchData.branchNum, 'endP': MorphData.endP_len}
    # branchEndPDF = pd.DataFrame(data=branchEndPDict)
    fig = plt.figure(figsize=(8,6))
    # seaborn.swarmplot(x='branch', y='endP', data=branchEndPDF)
    plt.scatter(BranchData.branchNum, MorphData.endP_len)
    plt.title("Distribution of Number of Tips\n for Given Number of Branches", fontsize=20)
    plt.xlabel("Number of Branches", fontsize=15)
    plt.ylabel("Number of Tips", fontsize=15)
    #plt.xlim(-1, 10)
    #plt.ylim(-1, 10)
    plt.tight_layout()
    plt.show()
    
    
    #==============================================================================
    
    
    fig = plt.figure(figsize=(8,6))
    seaborn.kdeplot(BranchData.branchNum, 
                    bw=25)
#    plt.xlim(-2, 8)
    plt.title("Estimated Distribution of Number of Branches by Type", fontsize=20)
    plt.xlabel("Number of Branches", fontsize=15)
    plt.ylabel("Estimated Probability Density", fontsize=15)
    plt.tight_layout()
    plt.show()
    
    
    #==============================================================================
    
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(MorphData.morph_dist_len, rGy)
    plt.yscale('log')
    plt.xscale('log')
    #plt.xlim(1, 10000)
    #plt.ylim(0.005, 1000)
    plt.title("Scaling Behavior of $R_{g}$ to $N$", fontsize=20)
    plt.xlabel("Number of Points", fontsize=15)
    plt.ylabel("Radius of Gyration", fontsize=15)
    plt.tight_layout()
    plt.show()
    
    
    #==============================================================================
    
    #reg_len_scale = np.average(np.divide(regMDistLen, morph_dist_len))
    poptR, pcovR = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(LengthData.length_total), 
                                            np.log10(rGy), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    perrR = np.sqrt(np.diag(pcovR))
    fitYR = objFuncPpow(LengthData.length_total, poptR[0], poptR[1])
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(LengthData.length_total, rGy)
    plt.plot(LengthData.length_total, fitYR, color='tab:red')
    plt.yscale('log')
    plt.xscale('log')
#    plt.xlim(10, 10000)
#    plt.ylim(7, 4000)
    plt.legend([str(round(poptR[0], 3)) + '$\pm$' + str(round(perrR[0], 3))], fontsize=15)
    plt.title(r"Scaling Behavior of $R_{g}$ to Length", fontsize=20)
    plt.xlabel(r"Length ($\lambda N$)", fontsize=15)
    plt.ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
    plt.tight_layout()
    plt.show()
    
    
    #==============================================================================
    
    OutputData.segOrdLen = np.empty(len(Parameter.nSize)*Parameter.numScaleSample)
    for r in range(len(OutputData.randTrk)):
        val = np.array(MorphData.indMorph_dist_flat[OutputData.randTrk[r][0]])[OutputData.randTrk[r][1]:OutputData.randTrk[r][2]]
        x = val[:,0]
        y = val[:,1]
        z = val[:,2]
        
        xd = [j-i for i, j in zip(x[:-1], x[1:])]
        yd = [j-i for i, j in zip(y[:-1], y[1:])]
        zd = [j-i for i, j in zip(z[:-1], z[1:])]
        dist = np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
        OutputData.segOrdLen[r] = dist
    
    nSize_l = [0.1, 1, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 
               300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550]
    
    nSize_lfcm = []
    rGySeg_avg = np.empty(len(nSize_l)-1)
    for i in range(len(nSize_l)-1):
        RS_s = np.where((OutputData.segOrdLen <= nSize_l[i+1]) &
                        (OutputData.segOrdLen >= nSize_l[i]))[0]
        nSize_lfcm.append(np.average(OutputData.segOrdLen[RS_s]))
        rGySeg_avg[i] = np.average(OutputData.rGySeg[RS_s])
    
    poptRS1, pcovRS1 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(OutputData.segOrdLen), 
                                              np.log10(np.sqrt(np.square(OutputData.rGySeg))), 
                                              p0=[1., 0.], 
                                              maxfev=100000)
    perrRS1 = np.sqrt(np.diag(pcovRS1))
    fitYRS1 = objFuncPpow(np.unique(OutputData.segOrdLen), poptRS1[0], poptRS1[1])
    
    fig, ax1 = plt.subplots(figsize=(12,8))
    ax1.xaxis.label.set_fontsize(15)
    ax1.xaxis.set_tick_params(which='major', length=7)
    ax1.xaxis.set_tick_params(which='minor', length=5)
    ax1.yaxis.label.set_fontsize(15)
    ax1.yaxis.set_tick_params(which='major', length=7)
    ax1.yaxis.set_tick_params(which='minor', length=5)
    ax1.scatter(LengthData.length_total, 
                np.sqrt(np.square(rGy)), color='tab:blue')
    ax1.plot(LengthData.length_total, fitYR, color='tab:red', lw=2)
    ax1.scatter(OutputData.segOrdLen, 
                np.sqrt(np.square(OutputData.rGySeg)), 
                color='tab:blue',
                facecolors='none')
    ax1.scatter(nSize_lfcm, 
                np.sqrt(np.square(rGySeg_avg)), 
                color='tab:orange')
    ax1.plot(np.unique(OutputData.segOrdLen), fitYRS1, color='tab:red', lw=2, linestyle='--')
#    ax1.plot(np.unique(OutputData.segOrdN[RS2])*Parameter.sSize, fitYRS2, color='tab:red', lw=2, linestyle='--')
#    ax1.plot(np.unique(OutputData.segOrdN[RS3])*Parameter.sSize, fitYRS3, color='tab:red', lw=2, linestyle='--')
#    ax1.vlines(0.8, 0.01, 11000, linestyles='dashed')
#    ax1.vlines(0.4, 0.01, 11000, linestyles='dashed')
    ax1.legend([str(round(poptR[0], 3)) + '$\pm$' + str(round(perrR[0], 3)),
            str(round(poptRS1[0], 3)) + '$\pm$' + str(round(perrRS1[0], 3))], fontsize=15)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim(0.1, 50000)
    ax1.set_ylim(0.05, 1000)
    ax1.set_xlabel(r"Length ($\lambda N$)", fontsize=15)
    ax1.set_ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
    #plt.tight_layout()
    if Parameter.SAVE:
        plt.savefig(Parameter.outputdir + '/segRG_morphScale_' + str(Parameter.RN) + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    #==============================================================================
    
    
    shift_N = 6
    poptRS_sl = []
    RS_x = []
    for i in range(len(nSize_l) - shift_N):
        RS_s = np.where((OutputData.segOrdLen <= nSize_l[i+shift_N]) &
                        (OutputData.segOrdLen >= nSize_l[i]))[0]
        
        if len(RS_s) > 0:
#            RS_x.append(np.average(nSize_l[i:i+shift_N]))
            RS_x.append(np.average(OutputData.segOrdLen[RS_s]))
            
            poptRS_s, pcovRS_s = scipy.optimize.curve_fit(objFuncGL, 
                                                          np.log10(OutputData.segOrdLen[RS_s]), 
                                                          np.log10(np.sqrt(np.square(OutputData.rGySeg[RS_s]))), 
                                                          p0=[1., 0.], 
                                                          maxfev=100000)
            poptRS_sl.append(poptRS_s[0])
    
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(RS_x, poptRS_sl)
    #plt.plot(regMDistLen*Parameter.sSize, fitYR, color='tab:red')
    #plt.yscale('log')
#    plt.hlines(poptR[0], 0.1, 1000, linestyles='--', color='tab:red')
#    plt.hlines(poptRS1[0], 0.1, 1000, linestyles='--', color='tab:green')
#    plt.hlines(poptRS3[0], 0.1, 1000, linestyles='--', color='tab:orange')
    #plt.yscale('log')
    plt.xscale('log')
#    plt.xlim(25, 2500)
    #plt.ylim(0.005, 1000)
    #plt.title(r"Scaling Behavior of Regularized $R_{g}$ to Regularized $N$", fontsize=20)
    plt.xlabel(r"Average Length ($\lambda N_{avg}$)", fontsize=15)
    plt.ylabel(r"Slope ($\nu$)", fontsize=15)
    #plt.tight_layout()
    if Parameter.SAVE:
        plt.savefig(Parameter.outputdir + '/segRG_slope_' + str(Parameter.RN) + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    #==============================================================================
#    
#    poptRS_sl = []
#    RS_x = []
#    for i in range(len(Parameter.nSize) - shift_N):
#        RS_s = np.where((OutputData.segOrdN <= Parameter.nSize[i+shift_N]) &
#                        (OutputData.segOrdN >= Parameter.nSize[i]))[0]
#        
#        RS_x.append(np.average(Parameter.nSize[i:i+shift_N]))
#        
#        poptRS_s, pcovRS_s = scipy.optimize.curve_fit(objFuncGL, 
#                                                      np.log10(OutputData.segOrdN[RS_s]), 
#                                                      np.log10(np.sqrt(np.square(OutputData.rGySeg[RS_s]))), 
#                                                      p0=[1., 0.], 
#                                                      maxfev=100000)
#        poptRS_sl.append(poptRS_s[0])
#    
#    
#    fig = plt.figure(figsize=(8,6))
#    plt.scatter(RS_x, poptRS_sl)
#    #plt.plot(regMDistLen*Parameter.sSize, fitYR, color='tab:red')
#    #plt.yscale('log')
##    plt.hlines(poptR[0], 0.1, 1000, linestyles='--', color='tab:red')
##    plt.hlines(poptRS1[0], 0.1, 1000, linestyles='--', color='tab:green')
##    plt.hlines(poptRS3[0], 0.1, 1000, linestyles='--', color='tab:orange')
#    #plt.yscale('log')
#    plt.xscale('log')
#    plt.xlim(25, 2500)
#    #plt.ylim(0.005, 1000)
#    #plt.title(r"Scaling Behavior of Regularized $R_{g}$ to Regularized $N$", fontsize=20)
#    plt.xlabel(r"Average Length ($\lambda N_{avg}$)", fontsize=15)
#    plt.ylabel(r"Slope ($\nu$)", fontsize=15)
#    #plt.tight_layout()
#    if Parameter.SAVE:
#        plt.savefig(Parameter.outputdir + '/regSegRG_slope_' + str(Parameter.RN) + '.png', dpi=300, bbox_inches='tight')
#    plt.show()
#  
    

#%%
t6 = time.time()

print('checkpoint 6: ' + str(t6-t5))
    
bstrk = []
bstrkval = []
ibind = []
bstrk_len = []
mdistl_nz = np.where(BranchData.branchNum > 10)[0]
#mdistl_nz = [0, 1, 2]
bcML = np.empty((len(mdistl_nz)*Parameter.numBranchSample, 3))
brGy = np.empty(len(mdistl_nz)*Parameter.numBranchSample)
cnt = 0

for m in mdistl_nz:
    bsrand = np.random.choice(BranchData.branchP[m], size=Parameter.numBranchSample, replace=False)
    bstrk_temp = []
    bstrk_len_temp = []
    
    for b in range(Parameter.numBranchSample):
        bstrkval_temp = []
        ibind_temp2 = []
        dist = 0
        
        for k in range(len(BranchData.indBranchTrk[m])):
            bsw = np.where(np.array(BranchData.indBranchTrk[m][k]) == bsrand[b])[0]
            if len(bsw) > 0:
                bstrkval_temp.append(BranchData.indBranchTrk[m][k][:bsw[0]+1])
    
        bstrkval_u = np.unique([item for sublist in bstrkval_temp for item in sublist])
        bstrk_temp.append(np.where(np.isin(MorphData.morph_id[m], bstrkval_u))[0])
        
        for lb in range(len(BranchData.branchTrk[m])):
            if all(elem in bstrkval_u for elem in BranchData.branchTrk[m][lb]):
                dist += LengthData.length_branch[m][lb]
        
        bstrk_len_temp.append(dist)
    
    for i in range(Parameter.numBranchSample):
        bcML[cnt] = (np.sum(np.array(MorphData.morph_dist[m])[bstrk_temp[i]], 
                                                axis=0)/len(np.array(MorphData.morph_dist[m])[bstrk_temp[i]]))
        rList = scipy.spatial.distance.cdist(np.array(MorphData.morph_dist[m])[bstrk_temp[i]], 
                                             np.array([bcML[cnt]])).flatten()
        brGy[cnt] = np.sqrt(np.sum(np.square(rList))/len(rList))
        cnt += 1
    
    bstrk.append(bstrk_temp)
    bstrk_len.append(bstrk_len_temp)


#%%
    
bstrk_len_flat = [item for sublist in bstrk_len for item in sublist]
    
poptRS2, pcovRS2 = scipy.optimize.curve_fit(objFuncGL, 
                                          np.log10(bstrk_len_flat), 
                                          np.log10(np.sqrt(np.square(brGy))), 
                                          p0=[1., 0.], 
                                          maxfev=100000)
perrRS2 = np.sqrt(np.diag(pcovRS2))
fitYRS2 = objFuncPpow(np.unique(bstrk_len_flat), poptRS2[0], poptRS2[1])

fig, ax1 = plt.subplots(figsize=(12,8))
ax1.xaxis.label.set_fontsize(15)
ax1.xaxis.set_tick_params(which='major', length=7)
ax1.xaxis.set_tick_params(which='minor', length=5)
ax1.yaxis.label.set_fontsize(15)
ax1.yaxis.set_tick_params(which='major', length=7)
ax1.yaxis.set_tick_params(which='minor', length=5)
ax1.scatter(LengthData.length_total, 
            np.sqrt(np.square(rGy)), color='tab:blue')
ax1.plot(LengthData.length_total, fitYR, color='tab:red', lw=2)
ax1.scatter(bstrk_len_flat, 
            np.sqrt(np.square(brGy)), 
            color='tab:blue',
            facecolors='none')
ax1.plot(np.unique(bstrk_len_flat), fitYRS2, color='tab:red', lw=2, linestyle='--')
ax1.legend([str(round(poptR[0], 3)) + '$\pm$' + str(round(perrR[0], 3)),
            str(round(poptRS2[0], 3)) + '$\pm$' + str(round(perrRS2[0], 3))], fontsize=15)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim(0.1, 50000)
ax1.set_ylim(0.05, 1000)


ax1.set_xlabel(r"Length ($\lambda N$)", fontsize=15)
ax1.set_ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
#plt.tight_layout()
plt.show()


#==============================================================================

bstrk_len_flat = np.array(bstrk_len_flat)

nSize_brl = [0.1, 1, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 
               300, 350, 400, 450, 500, 750, 1000]

shift_N = 6
poptRS_brl = []
RS_xbr = []
for i in range(len(nSize_brl) - shift_N):
    RS_s = np.where((bstrk_len_flat <= nSize_brl[i+shift_N]) &
                    (bstrk_len_flat >= nSize_brl[i]))[0]
    
    if len(RS_s) > 0:
        RS_xbr.append(np.average(bstrk_len_flat[RS_s]))
        
        poptRS_br, pcovRS_br = scipy.optimize.curve_fit(objFuncGL, 
                                                      np.log10(bstrk_len_flat[RS_s]), 
                                                      np.log10(np.sqrt(np.square(brGy[RS_s]))), 
                                                      p0=[1., 0.], 
                                                      maxfev=100000)
        poptRS_brl.append(poptRS_br[0])


fig = plt.figure(figsize=(8,6))
plt.scatter(RS_xbr, poptRS_brl)
#plt.plot(regMDistLen*Parameter.sSize, fitYR, color='tab:red')
#plt.yscale('log')
#    plt.hlines(poptR[0], 0.1, 1000, linestyles='--', color='tab:red')
#    plt.hlines(poptRS1[0], 0.1, 1000, linestyles='--', color='tab:green')
#    plt.hlines(poptRS3[0], 0.1, 1000, linestyles='--', color='tab:orange')
#plt.yscale('log')
plt.xscale('log')
#    plt.xlim(25, 2500)
#plt.ylim(0.005, 1000)
#plt.title(r"Scaling Behavior of Regularized $R_{g}$ to Regularized $N$", fontsize=20)
plt.xlabel(r"Average Length ($\lambda N_{avg}$)", fontsize=15)
plt.ylabel(r"Slope ($\nu$)", fontsize=15)
#plt.tight_layout()
if Parameter.SAVE:
    plt.savefig(Parameter.outputdir + '/segRG_slope_' + str(Parameter.RN) + '.png', dpi=300, bbox_inches='tight')
plt.show()



t7 = time.time()

print('checkpoint 7: ' + str(t7-t6))


#%% Cluster scaling exponent calculation ver 1
        
def cons_check(val):
    val = sorted(set(val))
    gaps = [[s, e] for s, e in zip(val, val[1:]) if s+1 < e]
    edges = iter(val[:1] + sum(gaps, []) + val[-1:])
    return list(zip(edges, edges))

radiussize = np.logspace(0, 2, 100)[34:95]

spheredist_calyx_sum = np.empty(len(radiussize))
spheredist_LH_sum = np.empty(len(radiussize))
spheredist_AL_sum = np.empty(len(radiussize))
spheredist_calyx_count1 = np.empty(len(radiussize))
spheredist_LH_count1 = np.empty(len(radiussize))
spheredist_AL_count1 = np.empty(len(radiussize))
spheredist_calyx_count2 = np.empty(len(radiussize))
spheredist_LH_count2 = np.empty(len(radiussize))
spheredist_AL_count2 = np.empty(len(radiussize))

for b in range(len(radiussize)):
    spheredist_calyx_temp = []
    spheredist_LH_temp = []
    spheredist_AL_temp = []
    spheredist_calyx_c_temp1 = []
    spheredist_LH_c_temp1 = []
    spheredist_AL_c_temp1 = []
    spheredist_calyx_c_temp2 = []
    spheredist_LH_c_temp2 = []
    spheredist_AL_c_temp2 = []
    
    for ib in range(len(MorphData.calyxdist)):
        inbound_calyx = np.where(np.sqrt(np.square(np.array(MorphData.calyxdist[ib])[:,0] - calyxCM[0]) +
                                         np.square(np.array(MorphData.calyxdist[ib])[:,1] - calyxCM[1]) +
                                         np.square(np.array(MorphData.calyxdist[ib])[:,2] - calyxCM[2])) <= radiussize[b])[0]
        dist_calyx = 0
        lenc = 0
        if len(inbound_calyx) > 1:
            valist = cons_check(inbound_calyx)
            for ibx in range(len(valist)):
                val = np.array(MorphData.calyxdist[ib])[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                x = val[:,0]
                y = val[:,1]
                z = val[:,2]
            
                xd = [j-i for i, j in zip(x[:-1], x[1:])]
                yd = [j-i for i, j in zip(y[:-1], y[1:])]
                zd = [j-i for i, j in zip(z[:-1], z[1:])]
                dist_calyx += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                if len(val) > 1:
                    lenc += len(val)
        spheredist_calyx_temp.append(dist_calyx)
        spheredist_calyx_c_temp1.append(lenc)
        spheredist_calyx_c_temp2.append(len(inbound_calyx))
        
    for ib in range(len(MorphData.LHdist)):
        inbound_LH = np.where(np.sqrt(np.square(np.array(MorphData.LHdist[ib])[:,0] - LHCM[0]) +
                                      np.square(np.array(MorphData.LHdist[ib])[:,1] - LHCM[1]) +
                                      np.square(np.array(MorphData.LHdist[ib])[:,2] - LHCM[2])) <= radiussize[b])[0]
        dist_LH = 0
        lenc = 0
        if len(inbound_LH) > 1:
            valist = cons_check(inbound_LH)
            for ibx in range(len(valist)):
                val = np.array(MorphData.LHdist[ib])[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                x = val[:,0]
                y = val[:,1]
                z = val[:,2]
            
                xd = [j-i for i, j in zip(x[:-1], x[1:])]
                yd = [j-i for i, j in zip(y[:-1], y[1:])]
                zd = [j-i for i, j in zip(z[:-1], z[1:])]
                dist_LH += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                if len(val) > 1:
                    lenc += len(val)
        spheredist_LH_temp.append(dist_LH)
        spheredist_LH_c_temp1.append(lenc)
        spheredist_LH_c_temp2.append(len(inbound_LH))
    
    for ib in range(len(MorphData.ALdist)):
        inbound_AL = np.where(np.sqrt(np.square(np.array(MorphData.ALdist[ib])[:,0] - ALCM[0]) +
                                      np.square(np.array(MorphData.ALdist[ib])[:,1] - ALCM[1]) +
                                      np.square(np.array(MorphData.ALdist[ib])[:,2] - ALCM[2])) <= radiussize[b])[0]
        dist_AL = 0
        lenc = 0
        if len(inbound_AL) > 1:
            valist = cons_check(inbound_AL)
            for ibx in range(len(valist)):
                val = np.array(MorphData.ALdist[ib])[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                x = val[:,0]
                y = val[:,1]
                z = val[:,2]
            
                xd = [j-i for i, j in zip(x[:-1], x[1:])]
                yd = [j-i for i, j in zip(y[:-1], y[1:])]
                zd = [j-i for i, j in zip(z[:-1], z[1:])]
                dist_AL += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                if len(val) > 1:
                    lenc += len(val)
        spheredist_AL_temp.append(dist_AL)
        spheredist_AL_c_temp1.append(lenc)
        spheredist_AL_c_temp2.append(len(inbound_AL))
        
    spheredist_calyx_sum[b] = np.sum(spheredist_calyx_temp)
    spheredist_LH_sum[b] = np.sum(spheredist_LH_temp)
    spheredist_AL_sum[b] = np.sum(spheredist_AL_temp)
    spheredist_calyx_count1[b] = np.sum(spheredist_calyx_c_temp1)
    spheredist_LH_count1[b] = np.sum(spheredist_LH_c_temp1)
    spheredist_AL_count1[b] = np.sum(spheredist_AL_c_temp1)
    spheredist_calyx_count2[b] = np.sum(spheredist_calyx_c_temp2)
    spheredist_LH_count2[b] = np.sum(spheredist_LH_c_temp2)
    spheredist_AL_count2[b] = np.sum(spheredist_AL_c_temp2)


#%% 
   
poptD_calyx_all = []
poptD_LH_all = []
poptD_AL_all = []

poptD_calyx, pcovD_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(radiussize[0:30]), 
                                                    np.log10(spheredist_calyx_sum[0:30]), 
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_calyx = np.sqrt(np.diag(pcovD_calyx))

poptD_LH, pcovD_LH = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[0:35]), 
                                              np.log10(spheredist_LH_sum[0:35]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_LH = np.sqrt(np.diag(pcovD_LH))

poptD_AL1, pcovD_AL1 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[19:40]), 
                                              np.log10(spheredist_AL_sum[19:40]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL1 = np.sqrt(np.diag(pcovD_AL1))

poptD_AL2, pcovD_AL2 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[:17]), 
                                              np.log10(spheredist_AL_sum[:17]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL2 = np.sqrt(np.diag(pcovD_AL2))


fitYD_calyx = objFuncPpow(radiussize, poptD_calyx[0], poptD_calyx[1])
fitYD_LH = objFuncPpow(radiussize, poptD_LH[0], poptD_LH[1])
fitYD_AL1 = objFuncPpow(radiussize, poptD_AL1[0], poptD_AL1[1])
fitYD_AL2 = objFuncPpow(radiussize, poptD_AL2[0], poptD_AL2[1])

fig = plt.figure(figsize=(8,6))

# plt.scatter(radiussize[:16], 
                    # spheredist_AL_sum[:16], color='tab:blue', facecolors='none', marker='s', alpha=0.5)
plt.scatter(radiussize, 
                    spheredist_AL_sum, color='tab:blue', facecolors='none')
plt.scatter(radiussize[:49], 
                    spheredist_calyx_sum[:49], color='tab:orange', facecolors='none')
plt.scatter(radiussize[:53], 
                    spheredist_LH_sum[:53], color='tab:green', facecolors='none')


plt.plot(radiussize[5:], fitYD_AL1[5:], lw=2, color='tab:blue', alpha=0.5)
plt.plot(radiussize[:30], fitYD_AL2[:30], lw=2, linestyle='--', color='tab:blue', alpha=0.5)
plt.plot(radiussize, fitYD_calyx, lw=2, color='tab:orange', alpha=0.5)
plt.plot(radiussize, fitYD_LH, lw=2, color='tab:green', alpha=0.5)

line1 = 2*np.power(radiussize, 16/7)
line2 = 1/60*np.power(radiussize, 4/1)

plt.plot(radiussize[30:45], line1[30:45], lw=2, color='k')
plt.plot(radiussize[10:25], line2[10:25], lw=2, color='k')

plt.yscale('log')
plt.xscale('log')
plt.legend(['AL1: ' + str(round(poptD_AL1[0], 3)) + '$\pm$' + str(round(perrD_AL1[0], 3)),
            'AL2: ' + str(round(poptD_AL2[0], 3)) + '$\pm$' + str(round(perrD_AL2[0], 3)),
            'MB calyx: ' + str(round(poptD_calyx[0], 3)) + '$\pm$' + str(round(perrD_calyx[0], 3)),
            'LH: ' + str(round(poptD_LH[0], 3)) + '$\pm$' + str(round(perrD_LH[0], 3))], fontsize=13)
#plt.xlim(1, 75)
#plt.ylim(3, 1500)
#plt.tight_layout()
plt.text(27, 1800, '$r^{16/7}$', fontsize=13)
plt.text(11, 100, '$r^{4}$', fontsize=13)
plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("$L$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_neuropil_fixed_5.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Cluster scaling exponent calculation V2

def cons_check(val):
    val = sorted(set(val))
    gaps = [[s, e] for s, e in zip(val, val[1:]) if s+1 < e]
    edges = iter(val[:1] + sum(gaps, []) + val[-1:])
    return list(zip(edges, edges))
            
radiussize = np.logspace(-1, 2, 100)[0:99]
# radiussize = np.linspace(1, 100, 100)[0:99:3]

spheredist_calyx_sum = np.empty(len(radiussize))
spheredist_LH_sum = np.empty(len(radiussize))
spheredist_AL_sum = np.empty(len(radiussize))
spheredist_calyx_count1 = np.empty(len(radiussize))
spheredist_LH_count1 = np.empty(len(radiussize))
spheredist_AL_count1 = np.empty(len(radiussize))
spheredist_calyx_count2 = np.empty(len(radiussize))
spheredist_LH_count2 = np.empty(len(radiussize))
spheredist_AL_count2 = np.empty(len(radiussize))

for b in range(len(radiussize)):
    spheredist_calyx_temp = []
    spheredist_LH_temp = []
    spheredist_AL_temp = []
    spheredist_calyx_c_temp1 = []
    spheredist_LH_c_temp1 = []
    spheredist_AL_c_temp1 = []
    spheredist_calyx_c_temp2 = []
    spheredist_LH_c_temp2 = []
    spheredist_AL_c_temp2 = []
    
    for ib in range(len(MorphData.calyxdist)):
        inbound_calyx = np.where(np.sqrt(np.square(np.array(MorphData.calyxdist[ib])[:,0] - calyxCM[0]) +
                                         np.square(np.array(MorphData.calyxdist[ib])[:,1] - calyxCM[1]) +
                                         np.square(np.array(MorphData.calyxdist[ib])[:,2] - calyxCM[2])) <= radiussize[b])[0]
        dist_calyx = 0
        lenc = 0
        if len(inbound_calyx) > 1:
            valist = cons_check(inbound_calyx)
            for ibx in range(len(valist)):
                val = np.array(MorphData.calyxdist[ib])[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                x = val[:,0]
                y = val[:,1]
                z = val[:,2]
            
                xd = [j-i for i, j in zip(x[:-1], x[1:])]
                yd = [j-i for i, j in zip(y[:-1], y[1:])]
                zd = [j-i for i, j in zip(z[:-1], z[1:])]
                dist_calyx += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                if len(val) > 1:
                    lenc += len(val)
        spheredist_calyx_temp.append(dist_calyx)
        spheredist_calyx_c_temp1.append(lenc)
        spheredist_calyx_c_temp2.append(len(inbound_calyx))
        
    for ib in range(len(MorphData.LHdist)):
        inbound_LH = np.where(np.sqrt(np.square(np.array(MorphData.LHdist[ib])[:,0] - LHCM[0]) +
                                      np.square(np.array(MorphData.LHdist[ib])[:,1] - LHCM[1]) +
                                      np.square(np.array(MorphData.LHdist[ib])[:,2] - LHCM[2])) <= radiussize[b])[0]
        dist_LH = 0
        lenc = 0
        if len(inbound_LH) > 1:
            valist = cons_check(inbound_LH)
            for ibx in range(len(valist)):
                val = np.array(MorphData.LHdist[ib])[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                x = val[:,0]
                y = val[:,1]
                z = val[:,2]
            
                xd = [j-i for i, j in zip(x[:-1], x[1:])]
                yd = [j-i for i, j in zip(y[:-1], y[1:])]
                zd = [j-i for i, j in zip(z[:-1], z[1:])]
                dist_LH += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                if len(val) > 1:
                    lenc += len(val)
        spheredist_LH_temp.append(dist_LH)
        spheredist_LH_c_temp1.append(lenc)
        spheredist_LH_c_temp2.append(len(inbound_LH))
    
    for ib in range(len(MorphData.ALdist)):
        inbound_AL = np.where(np.sqrt(np.square(np.array(MorphData.ALdist[ib])[:,0] - ALCM[0]) +
                                      np.square(np.array(MorphData.ALdist[ib])[:,1] - ALCM[1]) +
                                      np.square(np.array(MorphData.ALdist[ib])[:,2] - ALCM[2])) <= radiussize[b])[0]
        dist_AL = 0
        lenc = 0
        if len(inbound_AL) > 1:
            valist = cons_check(inbound_AL)
            for ibx in range(len(valist)):
                val = np.array(MorphData.ALdist[ib])[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                x = val[:,0]
                y = val[:,1]
                z = val[:,2]
            
                xd = [j-i for i, j in zip(x[:-1], x[1:])]
                yd = [j-i for i, j in zip(y[:-1], y[1:])]
                zd = [j-i for i, j in zip(z[:-1], z[1:])]
                dist_AL += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                if len(val) > 1:
                    lenc += len(val)
        spheredist_AL_temp.append(dist_AL)
        spheredist_AL_c_temp1.append(lenc)
        spheredist_AL_c_temp2.append(len(inbound_AL))
        
    spheredist_calyx_sum[b] = np.sum(spheredist_calyx_temp)
    spheredist_LH_sum[b] = np.sum(spheredist_LH_temp)
    spheredist_AL_sum[b] = np.sum(spheredist_AL_temp)
    spheredist_calyx_count1[b] = np.sum(spheredist_calyx_c_temp1)
    spheredist_LH_count1[b] = np.sum(spheredist_LH_c_temp1)
    spheredist_AL_count1[b] = np.sum(spheredist_AL_c_temp1)
    spheredist_calyx_count2[b] = np.sum(spheredist_calyx_c_temp2)
    spheredist_LH_count2[b] = np.sum(spheredist_LH_c_temp2)
    spheredist_AL_count2[b] = np.sum(spheredist_AL_c_temp2)


#%% 
   
poptD_calyx_all = []
poptD_LH_all = []
poptD_AL_all = []

farg_calyx = np.where(np.abs(np.diff(np.log10(spheredist_calyx_sum[np.nonzero(spheredist_calyx_sum)]))) > 0.03)[0][-5]
iarg_calyx = np.where(np.abs(np.diff(np.log10(spheredist_calyx_sum[np.nonzero(spheredist_calyx_sum)]))) < 0.1)[0][0]

poptD_calyx, pcovD_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(radiussize[np.nonzero(spheredist_calyx_sum)][iarg_calyx:farg_calyx]), 
                                                    np.log10(spheredist_calyx_sum[np.nonzero(spheredist_calyx_sum)][iarg_calyx:farg_calyx]), 
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_calyx = np.sqrt(np.diag(pcovD_calyx))

farg_LH = np.where(np.abs(np.diff(np.log10(spheredist_LH_sum[np.nonzero(spheredist_LH_sum)]))) > 0.03)[0][-1]
iarg_LH = np.where(np.abs(np.diff(np.log10(spheredist_LH_sum[np.nonzero(spheredist_LH_sum)]))) < 0.1)[0][0]

poptD_LH, pcovD_LH = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[np.nonzero(spheredist_LH_sum)][iarg_LH:farg_LH]), 
                                              np.log10(spheredist_LH_sum[np.nonzero(spheredist_LH_sum)][iarg_LH:farg_LH]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_LH = np.sqrt(np.diag(pcovD_LH))

farg_AL1 = np.where(np.abs(np.diff(np.log10(spheredist_AL_sum[np.nonzero(spheredist_AL_sum)]))) > 0.03)[0][-1]
iarg_AL1 = np.where(np.abs(np.diff(np.log10(spheredist_AL_sum[np.nonzero(spheredist_AL_sum)]))) < 0.1)[0][0]

poptD_AL1, pcovD_AL1 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[np.nonzero(spheredist_AL_sum)][iarg_AL1:farg_AL1]), 
                                              np.log10(spheredist_AL_sum[np.nonzero(spheredist_AL_sum)][iarg_AL1:farg_AL1]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL1 = np.sqrt(np.diag(pcovD_AL1))

farg_AL2 = np.where(np.abs(np.diff(np.log10(spheredist_AL_sum[np.nonzero(spheredist_AL_sum)]))) > 0.03)[0][-15]
iarg_AL2 = np.where(np.abs(np.diff(np.log10(spheredist_AL_sum[np.nonzero(spheredist_AL_sum)]))) < 0.1)[0][0]

poptD_AL2, pcovD_AL2 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[np.nonzero(spheredist_AL_sum)][iarg_AL2:farg_AL2]), 
                                              np.log10(spheredist_AL_sum[np.nonzero(spheredist_AL_sum)][iarg_AL2:farg_AL2]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL2 = np.sqrt(np.diag(pcovD_AL2))

farg_AL3 = np.where(np.abs(np.diff(np.log10(spheredist_AL_sum[np.nonzero(spheredist_AL_sum)]))) > 0.03)[0][-15]
iarg_AL3 = np.where(np.abs(np.diff(np.log10(spheredist_AL_sum[np.nonzero(spheredist_AL_sum)]))) < 0.1)[0][0]

poptD_AL3, pcovD_AL3 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[np.nonzero(spheredist_AL_sum)][iarg_AL3:farg_AL3]), 
                                              np.log10(spheredist_AL_sum[np.nonzero(spheredist_AL_sum)][iarg_AL3:farg_AL3]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL3 = np.sqrt(np.diag(pcovD_AL3))


fitYD_calyx = objFuncPpow(radiussize, poptD_calyx[0], poptD_calyx[1])
fitYD_LH = objFuncPpow(radiussize, poptD_LH[0], poptD_LH[1])
fitYD_AL1 = objFuncPpow(radiussize, poptD_AL1[0], poptD_AL1[1])
fitYD_AL2 = objFuncPpow(radiussize, poptD_AL2[0], poptD_AL2[1])
fitYD_AL3 = objFuncPpow(radiussize, poptD_AL3[0], poptD_AL3[1])

fig = plt.figure(figsize=(8,6))

plt.scatter(radiussize[np.nonzero(spheredist_AL_sum)][iarg_AL2:], 
            spheredist_AL_sum[np.nonzero(spheredist_AL_sum)][iarg_AL2:], 
            color='tab:blue', facecolors='none')
plt.scatter(radiussize[np.nonzero(spheredist_calyx_sum)][iarg_calyx:], 
            spheredist_calyx_sum[np.nonzero(spheredist_calyx_sum)][iarg_calyx:], 
            color='tab:orange', facecolors='none')
plt.scatter(radiussize[np.nonzero(spheredist_LH_sum)][iarg_LH:], 
            spheredist_LH_sum[np.nonzero(spheredist_LH_sum)][iarg_LH:],
            color='tab:green', facecolors='none')


plt.plot(radiussize[np.nonzero(spheredist_AL_sum)][iarg_AL2:],
         fitYD_AL1[np.nonzero(spheredist_AL_sum)][iarg_AL2:],
         lw=2, color='tab:blue', alpha=0.5)
plt.plot(radiussize[np.nonzero(spheredist_AL_sum)][iarg_AL2-10:-27], 
          fitYD_AL2[np.nonzero(spheredist_AL_sum)][iarg_AL2-10:-27],
          lw=2, linestyle='--', color='tab:blue', alpha=0.5)
plt.plot(radiussize[np.nonzero(spheredist_AL_sum)][iarg_AL3:-57], 
          fitYD_AL3[np.nonzero(spheredist_AL_sum)][iarg_AL3:-57],
          lw=2, linestyle='dotted', color='tab:blue', alpha=0.5)
plt.plot(radiussize[np.nonzero(spheredist_calyx_sum)][iarg_calyx:],
         fitYD_calyx[np.nonzero(spheredist_calyx_sum)][iarg_calyx:],
         lw=2, color='tab:orange', alpha=0.5)
plt.plot(radiussize[np.nonzero(spheredist_LH_sum)][iarg_LH:],
         fitYD_LH[np.nonzero(spheredist_LH_sum)][iarg_LH:],
         lw=2, color='tab:green', alpha=0.5)
plt.yscale('log')
plt.xscale('log')
plt.legend(['AL1: ' + str(round(poptD_AL1[0], 3)) + '$\pm$' + str(round(perrD_AL1[0], 3)),
            'AL2: ' + str(round(poptD_AL2[0], 3)) + '$\pm$' + str(round(perrD_AL2[0], 3)),
            'AL3: ' + str(round(poptD_AL3[0], 3)) + '$\pm$' + str(round(perrD_AL3[0], 3)),
            'MB calyx: ' + str(round(poptD_calyx[0], 3)) + '$\pm$' + str(round(perrD_calyx[0], 3)),
            'LH: ' + str(round(poptD_LH[0], 3)) + '$\pm$' + str(round(perrD_LH[0], 3))], fontsize=13)
#plt.xlim(1, 75)
#plt.ylim(3, 1500)
#plt.tight_layout()
line1 = 2*np.power(radiussize, 16/7)
line2 = 1/60*np.power(radiussize, 4/1)
# line3 = 1/60*np.power(radiussize, 1)

plt.plot(radiussize[60:70], line1[60:70], lw=2, color='k')
plt.plot(radiussize[40:50], line2[40:50], lw=2, color='k')

plt.text(27, 1800, '$r^{16/7}$', fontsize=13)
plt.text(11, 100, '$r^{4}$', fontsize=13)

plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("$L$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_neuropil_fixed_6.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Moving window

Calyxmw = []
Calyxmwerr = []
LHmw = []
LHmwerr = []
ALmw = []
ALmwerr = []
mwx_calyx = []
mwx_LH = []
mwx_AL = []
shiftN = 10
for i in range(len(radiussize[np.nonzero(spheredist_calyx_sum)]) - shiftN):
    mwx_calyx.append(np.average(radiussize[np.nonzero(spheredist_calyx_sum)][i:i+shiftN]))
    
    poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(radiussize[np.nonzero(spheredist_calyx_sum)][i:i+shiftN]), 
                                                np.log10(spheredist_calyx_sum[np.nonzero(spheredist_calyx_sum)][i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    Calyxmw.append(poptmxc[0])
    Calyxmwerr.append(np.sqrt(np.diag(pcovmxc))[0])

for i in range(len(radiussize[np.nonzero(spheredist_LH_sum)]) - shiftN):    
    mwx_LH.append(np.average(radiussize[np.nonzero(spheredist_LH_sum)][i:i+shiftN]))
    poptmxl, pcovmxl = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(radiussize[np.nonzero(spheredist_LH_sum)][i:i+shiftN]), 
                                                np.log10(spheredist_LH_sum[np.nonzero(spheredist_LH_sum)][i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    LHmw.append(poptmxl[0])
    LHmwerr.append(np.sqrt(np.diag(pcovmxl))[0])
    
for i in range(len(radiussize[np.nonzero(spheredist_AL_sum)]) - shiftN):
    mwx_AL.append(np.average(radiussize[np.nonzero(spheredist_AL_sum)][i:i+shiftN]))
    poptmxa, pcovmxa = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(radiussize[np.nonzero(spheredist_AL_sum)][i:i+shiftN]), 
                                                np.log10(spheredist_AL_sum[np.nonzero(spheredist_AL_sum)][i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    ALmw.append(poptmxa[0])
    ALmwerr.append(np.sqrt(np.diag(pcovmxa))[0])
    

fig = plt.figure(figsize=(8,6))
plt.plot(np.array(mwx_AL), np.array(ALmw), lw=2)
plt.plot(np.array(mwx_calyx), np.array(Calyxmw), lw=2)
plt.plot(np.array(mwx_LH), np.array(LHmw), lw=2)
# plt.fill_between(mwx_AL, np.array(ALmw)-np.array(ALmwerr), np.array(ALmw)+np.array(ALmwerr), alpha=0.3)
# plt.fill_between(mwx_calyx, np.array(Calyxmw)-np.array(Calyxmwerr), np.array(Calyxmw)+np.array(Calyxmwerr), alpha=0.3)
# plt.fill_between(mwx_LH, np.array(LHmw)-np.array(LHmwerr), np.array(LHmw)+np.array(LHmwerr), alpha=0.3)

plt.xscale('log')
plt.legend(["AL", "MB calyx", "LH"], fontsize=13)
# plt.yscale('log')
#plt.xlim(1, 75)
plt.ylim(0, 4)
#plt.tight_layout()
plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("Slope", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_neuropil_mv_fixed_2.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Cluster scaling exponent calculation but with count instead


poptD_calyx_all = []
poptD_LH_all = []
poptD_AL_all = []

farg_calyx = np.where(np.abs(np.diff(np.log10(spheredist_calyx_count1[np.nonzero(spheredist_calyx_count1)]))) > 0.05)[0][-1]
iarg_calyx = np.where(np.abs(np.diff(np.log10(spheredist_calyx_count1[np.nonzero(spheredist_calyx_count1)]))) < 0.1)[0][0]

poptD_calyx, pcovD_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(radiussize[np.nonzero(spheredist_calyx_count1)][iarg_calyx:farg_calyx]), 
                                                    np.log10(spheredist_calyx_count1[np.nonzero(spheredist_calyx_count1)][iarg_calyx:farg_calyx]), 
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_calyx = np.sqrt(np.diag(pcovD_calyx))

farg_LH = np.where(np.abs(np.diff(np.log10(spheredist_LH_count1[np.nonzero(spheredist_LH_count1)]))) > 0.05)[0][-1]
iarg_LH = np.where(np.abs(np.diff(np.log10(spheredist_LH_count1[np.nonzero(spheredist_LH_count1)]))) < 0.1)[0][0]

poptD_LH, pcovD_LH = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[np.nonzero(spheredist_LH_count1)][iarg_LH:farg_LH]), 
                                              np.log10(spheredist_LH_count1[np.nonzero(spheredist_LH_count1)][iarg_LH:farg_LH]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_LH = np.sqrt(np.diag(pcovD_LH))

farg_AL = np.where(np.abs(np.diff(np.log10(spheredist_AL_count1[np.nonzero(spheredist_AL_count1)]))) > 0.05)[0][-1]
iarg_AL = np.where(np.abs(np.diff(np.log10(spheredist_AL_count1[np.nonzero(spheredist_AL_count1)]))) < 0.1)[0][0]

poptD_AL, pcovD_AL = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[np.nonzero(spheredist_AL_count1)][iarg_AL:farg_AL]), 
                                              np.log10(spheredist_AL_count1[np.nonzero(spheredist_AL_count1)][iarg_AL:farg_AL]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL = np.sqrt(np.diag(pcovD_AL))

fitYD_calyx = objFuncPpow(radiussize, poptD_calyx[0], poptD_calyx[1])
fitYD_LH = objFuncPpow(radiussize, poptD_LH[0], poptD_LH[1])
fitYD_AL = objFuncPpow(radiussize, poptD_AL[0], poptD_AL[1])

fig = plt.figure(figsize=(8,6))

plt.scatter(radiussize[np.nonzero(spheredist_AL_count1)][iarg_AL:], 
            spheredist_AL_count1[np.nonzero(spheredist_AL_count1)][iarg_AL:],
            color='tab:blue', facecolors='none', alpha=0.5)
plt.scatter(radiussize[np.nonzero(spheredist_calyx_count1)][iarg_calyx:], 
            spheredist_calyx_count1[np.nonzero(spheredist_calyx_count1)][iarg_calyx:],
            color='tab:orange', facecolors='none', alpha=0.5)
plt.scatter(radiussize[np.nonzero(spheredist_LH_count1)][iarg_LH:], 
            spheredist_LH_count1[np.nonzero(spheredist_LH_count1)][iarg_LH:],
            color='tab:green', facecolors='none', alpha=0.5)


plt.plot(radiussize[np.nonzero(spheredist_AL_count1)][iarg_AL:],
         fitYD_AL[np.nonzero(spheredist_AL_count1)][iarg_AL:],
         lw=2, color='tab:blue')
plt.plot(radiussize[np.nonzero(spheredist_calyx_count1)][iarg_calyx:], 
         fitYD_calyx[np.nonzero(spheredist_calyx_count1)][iarg_calyx:],
         lw=2, color='tab:orange')
plt.plot(radiussize[np.nonzero(spheredist_LH_count1)][iarg_LH:],
         fitYD_LH[np.nonzero(spheredist_LH_count1)][iarg_LH:],
         lw=2, color='tab:green')
plt.yscale('log')
plt.xscale('log')
plt.legend(['AL: ' + str(round(poptD_AL[0], 3)) + '$\pm$' + str(round(perrD_AL[0], 3)),
            'MB calyx: ' + str(round(poptD_calyx[0], 3)) + '$\pm$' + str(round(perrD_calyx[0], 3)),
            'LH: ' + str(round(poptD_LH[0], 3)) + '$\pm$' + str(round(perrD_LH[0], 3))], fontsize=13)
#plt.xlim(1, 75)
#plt.ylim(3, 1500)
#plt.tight_layout()
plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("Count", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_neuropil_count1_fixed_2.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Moving window

Calyxmw = []
Calyxmwerr = []
LHmw = []
LHmwerr = []
ALmw = []
ALmwerr = []
mwx_calyx = []
mwx_LH = []
mwx_AL = []
shiftN = 15
for i in range(len(radiussize[np.nonzero(spheredist_calyx_count1)]) - shiftN):
    mwx_calyx.append(np.average(radiussize[np.nonzero(spheredist_calyx_count1)][i:i+shiftN]))
    
    poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(radiussize[np.nonzero(spheredist_calyx_count1)][i:i+shiftN]), 
                                                np.log10(spheredist_calyx_count1[np.nonzero(spheredist_calyx_count1)][i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    Calyxmw.append(poptmxc[0])
    Calyxmwerr.append(np.sqrt(np.diag(pcovmxc))[0])

for i in range(len(radiussize[np.nonzero(spheredist_LH_count1)]) - shiftN):    
    mwx_LH.append(np.average(radiussize[np.nonzero(spheredist_LH_count1)][i:i+shiftN]))
    poptmxl, pcovmxl = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(radiussize[np.nonzero(spheredist_LH_count1)][i:i+shiftN]), 
                                                np.log10(spheredist_LH_count1[np.nonzero(spheredist_LH_count1)][i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    LHmw.append(poptmxl[0])
    LHmwerr.append(np.sqrt(np.diag(pcovmxl))[0])
    
for i in range(len(radiussize[np.nonzero(spheredist_AL_count1)]) - shiftN):
    mwx_AL.append(np.average(radiussize[np.nonzero(spheredist_AL_count1)][i:i+shiftN]))
    poptmxa, pcovmxa = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(radiussize[np.nonzero(spheredist_AL_count1)][i:i+shiftN]), 
                                                np.log10(spheredist_AL_count1[np.nonzero(spheredist_AL_count1)][i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    ALmw.append(poptmxa[0])
    ALmwerr.append(np.sqrt(np.diag(pcovmxa))[0])
    

fig = plt.figure(figsize=(8,6))
plt.plot(1/np.array(mwx_AL), 1/np.array(ALmw), lw=2)
plt.plot(1/np.array(mwx_calyx), 1/np.array(Calyxmw), lw=2)
plt.plot(1/np.array(mwx_LH), 1/np.array(LHmw), lw=2)
# plt.fill_between(mwx_AL, np.array(ALmw)-np.array(ALmwerr), np.array(ALmw)+np.array(ALmwerr), alpha=0.3)
# plt.fill_between(mwx_calyx, np.array(Calyxmw)-np.array(Calyxmwerr), np.array(Calyxmw)+np.array(Calyxmwerr), alpha=0.3)
# plt.fill_between(mwx_LH, np.array(LHmw)-np.array(LHmwerr), np.array(LHmw)+np.array(LHmwerr), alpha=0.3)

plt.xscale('log')
plt.legend(["AL", "MB calyx", "LH"], fontsize=13)
# plt.yscale('log')
#plt.xlim(1, 75)
plt.ylim(0, 1.5)
#plt.tight_layout()
plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("Slope", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_neuropil_count1_mv_fixed_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Cluster scaling exponent calculation per neuron

def cons_check(val):
    val = sorted(set(val))
    gaps = [[s, e] for s, e in zip(val, val[1:]) if s+1 < e]
    edges = iter(val[:1] + sum(gaps, []) + val[-1:])
    return list(zip(edges, edges))

radiussize = np.logspace(-1, 2, 100)[5:95:2]

un_calyx = np.unique(MorphData.calyxdist_trk)
un_LH = np.unique(MorphData.LHdist_trk)
un_AL = np.unique(MorphData.ALdist_trk)

spheredist_calyx_sum = np.empty((len(un_calyx),len(radiussize)))
spheredist_LH_sum = np.empty((len(un_LH),len(radiussize)))
spheredist_AL_sum = np.empty((len(un_AL), len(radiussize)))
spheredist_calyx_count1 = np.empty((len(un_calyx),len(radiussize)))
spheredist_LH_count1 = np.empty((len(un_LH),len(radiussize)))
spheredist_AL_count1 = np.empty((len(un_AL), len(radiussize)))
spheredist_calyx_count2 = np.empty((len(un_calyx),len(radiussize)))
spheredist_LH_count2 = np.empty((len(un_LH),len(radiussize)))
spheredist_AL_count2 = np.empty((len(un_AL), len(radiussize)))

for b in range(len(radiussize)):
    for n in range(len(un_calyx)):
        spheredist_calyx_temp = []
        spheredist_calyx_c_temp1 = []
        spheredist_calyx_c_temp2 = []
        
        trkd = np.where(MorphData.calyxdist_trk == un_calyx[n])[0]
        ncalyxdist_flat = np.array([item for sublist in np.array(MorphData.calyxdist, dtype=object)[trkd] for item in sublist])
        ncalyx_CM = np.average(ncalyxdist_flat, axis=0)
        
        for ib in trkd:
            inbound_calyx = np.where(np.sqrt(np.square(np.array(MorphData.calyxdist[ib])[:,0] - ncalyx_CM[0]) +
                                              np.square(np.array(MorphData.calyxdist[ib])[:,1] - ncalyx_CM[1]) +
                                              np.square(np.array(MorphData.calyxdist[ib])[:,2] - ncalyx_CM[2])) <= radiussize[b])[0]
            dist_calyx = 0
            lenc = 0
            if len(inbound_calyx) > 1:
                valist = cons_check(inbound_calyx)
                for ibx in range(len(valist)):
                    val = np.array(MorphData.calyxdist[ib])[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                    x = val[:,0]
                    y = val[:,1]
                    z = val[:,2]
                
                    xd = [j-i for i, j in zip(x[:-1], x[1:])]
                    yd = [j-i for i, j in zip(y[:-1], y[1:])]
                    zd = [j-i for i, j in zip(z[:-1], z[1:])]
                    dist_calyx += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                    if len(val) > 1:
                        lenc += len(val)
            spheredist_calyx_temp.append(dist_calyx)
            spheredist_calyx_c_temp1.append(lenc)
            spheredist_calyx_c_temp2.append(len(inbound_calyx))
            
        spheredist_calyx_sum[n][b] = np.sum(spheredist_calyx_temp)
        spheredist_calyx_count1[n][b] = np.sum(spheredist_calyx_c_temp1)
        spheredist_calyx_count2[n][b] = np.sum(spheredist_calyx_c_temp2)
    
    for n in range(len(un_LH)):
        spheredist_LH_temp = []
        spheredist_LH_c_temp1 = []
        spheredist_LH_c_temp2 = []
        
        trkd = np.where(MorphData.LHdist_trk == un_LH[n])[0]
        nLHdist_flat = np.array([item for sublist in np.array(MorphData.LHdist, dtype=object)[trkd] for item in sublist])
        nLH_CM = np.average(nLHdist_flat, axis=0)
        
        for ib in trkd:
            inbound_LH = np.where(np.sqrt(np.square(np.array(MorphData.LHdist[ib])[:,0] - nLH_CM[0]) +
                                              np.square(np.array(MorphData.LHdist[ib])[:,1] - nLH_CM[1]) +
                                              np.square(np.array(MorphData.LHdist[ib])[:,2] - nLH_CM[2])) <= radiussize[b])[0]
            dist_LH = 0
            lenc = 0
            if len(inbound_LH) > 1:
                valist = cons_check(inbound_LH)
                for ibx in range(len(valist)):
                    val = np.array(MorphData.LHdist[ib])[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                    x = val[:,0]
                    y = val[:,1]
                    z = val[:,2]
                
                    xd = [j-i for i, j in zip(x[:-1], x[1:])]
                    yd = [j-i for i, j in zip(y[:-1], y[1:])]
                    zd = [j-i for i, j in zip(z[:-1], z[1:])]
                    dist_LH += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                    if len(val) > 1:
                        lenc += len(val)
            spheredist_LH_temp.append(dist_LH)
            spheredist_LH_c_temp1.append(lenc)
            spheredist_LH_c_temp2.append(len(inbound_LH))
            
        spheredist_LH_sum[n][b] = np.sum(spheredist_LH_temp)
        spheredist_LH_count1[n][b] = np.sum(spheredist_LH_c_temp1)
        spheredist_LH_count2[n][b] = np.sum(spheredist_LH_c_temp2)
    
    for n in range(len(un_AL)):
        spheredist_AL_temp = []
        spheredist_AL_c_temp1 = []
        spheredist_AL_c_temp2 = []
        
        trkd = np.where(MorphData.ALdist_trk == un_AL[n])[0]
        nALdist_flat = np.array([item for sublist in np.array(MorphData.ALdist, dtype=object)[trkd] for item in sublist])
        nAL_CM = np.average(nALdist_flat, axis=0)
        
        for ib in trkd:
            inbound_AL = np.where(np.sqrt(np.square(np.array(MorphData.ALdist[ib])[:,0] - nAL_CM[0]) +
                                              np.square(np.array(MorphData.ALdist[ib])[:,1] - nAL_CM[1]) +
                                              np.square(np.array(MorphData.ALdist[ib])[:,2] - nAL_CM[2])) <= radiussize[b])[0]
            dist_AL = 0
            lenc = 0
            if len(inbound_AL) > 1:
                valist = cons_check(inbound_AL)
                for ibx in range(len(valist)):
                    val = np.array(MorphData.ALdist[ib])[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                    x = val[:,0]
                    y = val[:,1]
                    z = val[:,2]
                
                    xd = [j-i for i, j in zip(x[:-1], x[1:])]
                    yd = [j-i for i, j in zip(y[:-1], y[1:])]
                    zd = [j-i for i, j in zip(z[:-1], z[1:])]
                    dist_AL += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                    if len(val) > 1:
                        lenc += len(val)
            spheredist_AL_temp.append(dist_AL)
            spheredist_AL_c_temp1.append(lenc)
            spheredist_AL_c_temp2.append(len(inbound_AL))
            
        spheredist_AL_sum[n][b] = np.sum(spheredist_AL_temp)
        spheredist_AL_count1[n][b] = np.sum(spheredist_AL_c_temp1)
        spheredist_AL_count2[n][b] = np.sum(spheredist_AL_c_temp2)


spheredist_AL_sum = np.delete(spheredist_AL_sum, 73, 0)

spheredist_calyx_sum = np.delete(spheredist_calyx_sum, [40,41], 0)

spheredist_calyx_sum_avg = np.average(spheredist_calyx_sum, axis=0)
spheredist_calyx_sum_std = np.std(spheredist_calyx_sum, axis=0)
spheredist_LH_sum_avg = np.average(spheredist_LH_sum, axis=0)
spheredist_LH_sum_std = np.std(spheredist_LH_sum, axis=0)
spheredist_AL_sum_avg = np.average(spheredist_AL_sum, axis=0)
spheredist_AL_sum_std = np.std(spheredist_AL_sum, axis=0)

spheredist_calyx_count1_avg = np.average(spheredist_calyx_count1, axis=0)
spheredist_calyx_count1_std = np.std(spheredist_calyx_count1, axis=0)
spheredist_LH_count1_avg = np.average(spheredist_LH_count1, axis=0)
spheredist_LH_count1_std = np.std(spheredist_LH_count1, axis=0)
spheredist_AL_count1_avg = np.average(spheredist_AL_count1, axis=0)
spheredist_AL_count1_std = np.std(spheredist_AL_count1, axis=0)

spheredist_calyx_count2_avg = np.average(spheredist_calyx_count2, axis=0)
spheredist_calyx_count2_std = np.std(spheredist_calyx_count2, axis=0)
spheredist_LH_count2_avg = np.average(spheredist_LH_count2, axis=0)
spheredist_LH_count2_std = np.std(spheredist_LH_count2, axis=0)
spheredist_AL_count2_avg = np.average(spheredist_AL_count2, axis=0)
spheredist_AL_count2_std = np.std(spheredist_AL_count2, axis=0)




#%%

spheredist_calyx_sum_avg_nz = spheredist_calyx_sum_avg[np.nonzero(spheredist_calyx_sum_avg)]
spheredist_LH_sum_avg_nz = spheredist_LH_sum_avg[np.nonzero(spheredist_LH_sum_avg)]
spheredist_AL_sum_avg_nz = spheredist_AL_sum_avg[np.nonzero(spheredist_AL_sum_avg)]

farg = np.where(np.abs(np.diff(np.log10(spheredist_calyx_sum_avg))) > 0.14)[0][-1]
iarg = np.where(np.abs(np.diff(np.log10(spheredist_calyx_sum_avg))) > 0.14)[0][0]

poptD_calyx, pcovD_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(radiussize[np.nonzero(spheredist_calyx_sum_avg)][iarg:farg]), 
                                                    np.log10(spheredist_calyx_sum_avg_nz[iarg:farg]), 
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_calyx = np.sqrt(np.diag(pcovD_calyx))

farg = np.where(np.abs(np.diff(np.log10(spheredist_LH_sum_avg))) > 0.14)[0][-1]
iarg = np.where(np.abs(np.diff(np.log10(spheredist_LH_sum_avg))) > 0.14)[0][0]

poptD_LH, pcovD_LH = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[np.nonzero(spheredist_LH_sum_avg)][iarg:farg]), 
                                              np.log10(spheredist_LH_sum_avg_nz[iarg:farg]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_LH = np.sqrt(np.diag(pcovD_LH))

farg = np.where(np.abs(np.diff(np.log10(spheredist_AL_sum_avg))) > 0.14)[0][-1]
iarg = np.where(np.abs(np.diff(np.log10(spheredist_AL_sum_avg))) > 0.14)[0][0]

poptD_AL, pcovD_AL = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[np.nonzero(spheredist_AL_sum_avg)][iarg:farg]), 
                                              np.log10(spheredist_AL_sum_avg_nz[iarg:farg]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL = np.sqrt(np.diag(pcovD_AL))

fitYD_calyx = objFuncPpow(radiussize, poptD_calyx[0], poptD_calyx[1])
fitYD_LH = objFuncPpow(radiussize, poptD_LH[0], poptD_LH[1])
fitYD_AL = objFuncPpow(radiussize, poptD_AL[0], poptD_AL[1])

fig = plt.figure(figsize=(8,6))

plt.scatter(radiussize[np.nonzero(spheredist_AL_sum_avg)], spheredist_AL_sum_avg_nz, color='tab:blue', facecolors='none')
# plt.errorbar(radiussize, spheredist_AL_sum_avg, scipy.stats.sem(spheredist_AL_sum, axis=0), capsize=2)
plt.scatter(radiussize[np.nonzero(spheredist_calyx_sum_avg)], spheredist_calyx_sum_avg_nz, color='tab:orange', facecolors='none')
# plt.errorbar(radiussize, spheredist_calyx_sum_avg, scipy.stats.sem(spheredist_calyx_sum, axis=0), capsize=2)
plt.scatter(radiussize[np.nonzero(spheredist_LH_sum_avg)], spheredist_LH_sum_avg_nz, color='tab:green', facecolors='none')
# plt.errorbar(radiussize, spheredist_LH_sum_avg, scipy.stats.sem(spheredist_LH_sum, axis=0), capsize=2)


plt.plot(radiussize, fitYD_AL, lw=2, color='tab:blue', alpha=0.5)
plt.plot(radiussize, fitYD_calyx, lw=2, color='tab:orange', alpha=0.5)
plt.plot(radiussize, fitYD_LH, lw=2, color='tab:green', alpha=0.5)
plt.yscale('log')
plt.xscale('log')
plt.legend(['AL: ' + str(round(poptD_AL[0], 3)) + '$\pm$' + str(round(perrD_AL[0], 3)),
            'MB calyx: ' + str(round(poptD_calyx[0], 3)) + '$\pm$' + str(round(perrD_calyx[0], 3)),
            'LH: ' + str(round(poptD_LH[0], 3)) + '$\pm$' + str(round(perrD_LH[0], 3))], fontsize=13)
#plt.xlim(1, 75)
# plt.ylim(10e-3, 10e2)
#plt.tight_layout()
plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("$L$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_neuropil_per_n_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(8,6))

plt.scatter(np.tile(radiussize, (len(spheredist_AL_sum),1)), spheredist_AL_sum, color='tab:blue', facecolors='none', alpha=0.5)
seaborn.kdeplot(x=np.tile(radiussize, len(spheredist_AL_sum))[np.nonzero(spheredist_AL_sum.flatten())],
                y=spheredist_AL_sum.flatten()[np.nonzero(spheredist_AL_sum.flatten())], log_scale=True, color='tab:blue')

plt.yscale('log')
plt.xscale('log')
plt.xlim(1e-1, 1e2)
plt.ylim(1e-2, 1e4)
#plt.tight_layout()
plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("$L$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_AL_per_n_1.png', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(8,6))

plt.scatter(np.tile(radiussize, (len(spheredist_calyx_sum),1)), spheredist_calyx_sum, color='tab:orange', facecolors='none', alpha=0.5)
seaborn.kdeplot(x=np.tile(radiussize, len(spheredist_calyx_sum))[np.nonzero(spheredist_calyx_sum.flatten())],
                y=spheredist_calyx_sum.flatten()[np.nonzero(spheredist_calyx_sum.flatten())], log_scale=True, color='tab:orange')

plt.yscale('log')
plt.xscale('log')
plt.xlim(1e-1, 1e2)
plt.ylim(1e-2, 1e4)
#plt.tight_layout()
plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("$L$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_calyx_per_n_1.png', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(8,6))

plt.scatter(np.tile(radiussize, (len(spheredist_LH_sum),1)), spheredist_LH_sum, color='tab:green', facecolors='none', alpha=0.5)
seaborn.kdeplot(x=np.tile(radiussize, len(spheredist_LH_sum))[np.nonzero(spheredist_LH_sum.flatten())],
                y=spheredist_LH_sum.flatten()[np.nonzero(spheredist_LH_sum.flatten())], log_scale=True, color='tab:green')

plt.yscale('log')
plt.xscale('log')
plt.xlim(1e-1, 1e2)
plt.ylim(1e-2, 1e4)
#plt.tight_layout()
plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("$L$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_LH_per_n_1.png', dpi=300, bbox_inches='tight')
plt.show()


#%% 

farg = np.where(np.abs(np.diff(np.log10(spheredist_calyx_count1_avg))) > 0.14)[0][-1]
iarg =  np.where(np.abs(np.diff(np.log10(spheredist_calyx_count1_avg))) > 0.14)[0][0]

poptD_calyx, pcovD_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(radiussize[iarg:farg]), 
                                                    np.log10(spheredist_calyx_count1_avg[iarg:farg]), 
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_calyx = np.sqrt(np.diag(pcovD_calyx))

farg = np.where(np.abs(np.diff(np.log10(spheredist_LH_count1_avg))) > 0.14)[0][-1]
iarg =  np.where(np.abs(np.diff(np.log10(spheredist_LH_count1_avg))) > 0.14)[0][0]

poptD_LH, pcovD_LH = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[iarg:farg]), 
                                              np.log10(spheredist_LH_count1_avg[iarg:farg]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_LH = np.sqrt(np.diag(pcovD_LH))

farg = np.where(np.abs(np.diff(np.log10(spheredist_AL_count1_avg))) > 0.14)[0][-1]
iarg =  np.where(np.abs(np.diff(np.log10(spheredist_AL_count1_avg))) > 0.14)[0][0]

poptD_AL, pcovD_AL = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[iarg:farg]), 
                                              np.log10(spheredist_AL_count1_avg[iarg:farg]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL = np.sqrt(np.diag(pcovD_AL))

fitYD_calyx = objFuncPpow(radiussize, poptD_calyx[0], poptD_calyx[1])
fitYD_LH = objFuncPpow(radiussize, poptD_LH[0], poptD_LH[1])
fitYD_AL = objFuncPpow(radiussize, poptD_AL[0], poptD_AL[1])

fig = plt.figure(figsize=(8,6))

plt.scatter(radiussize, 
                    spheredist_AL_count1_avg, color='tab:blue', facecolors='none', alpha=0.5)
plt.scatter(radiussize, 
                    spheredist_calyx_count1_avg, color='tab:orange', facecolors='none', alpha=0.5)
plt.scatter(radiussize, 
                    spheredist_LH_count1_avg, color='tab:green', facecolors='none', alpha=0.5)


plt.plot(radiussize, fitYD_AL, lw=2, color='tab:blue')
plt.plot(radiussize, fitYD_calyx, lw=2, color='tab:orange')
plt.plot(radiussize, fitYD_LH, lw=2, color='tab:green')
plt.yscale('log')
plt.xscale('log')
plt.legend(['AL: ' + str(round(poptD_AL[0], 3)) + '$\pm$' + str(round(perrD_AL[0], 3)),
            'MB calyx: ' + str(round(poptD_calyx[0], 3)) + '$\pm$' + str(round(perrD_calyx[0], 3)),
            'LH: ' + str(round(poptD_LH[0], 3)) + '$\pm$' + str(round(perrD_LH[0], 3))], fontsize=13)
#plt.xlim(1, 75)
#plt.ylim(3, 1500)
#plt.tight_layout()
plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("Count", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_neuropil_per_n_count1.pdf', dpi=300, bbox_inches='tight')
plt.show()



farg = np.where(np.abs(np.diff(np.log10(spheredist_calyx_count2_avg))) > 0.14)[0][-1]
iarg =  np.where(np.abs(np.diff(np.log10(spheredist_calyx_count2_avg))) > 0.14)[0][0]


poptD_calyx, pcovD_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(radiussize[iarg:farg]), 
                                                    np.log10(spheredist_calyx_count2_avg[iarg:farg]), 
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_calyx = np.sqrt(np.diag(pcovD_calyx))

farg = np.where(np.abs(np.diff(np.log10(spheredist_LH_count2_avg))) > 0.14)[0][-1]
iarg =  np.where(np.abs(np.diff(np.log10(spheredist_LH_count2_avg))) > 0.14)[0][0]

poptD_LH, pcovD_LH = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[iarg:farg]), 
                                              np.log10(spheredist_LH_count2_avg[iarg:farg]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_LH = np.sqrt(np.diag(pcovD_LH))

farg = np.where(np.abs(np.diff(np.log10(spheredist_AL_count2_avg))) > 0.14)[0][-1]
iarg =  np.where(np.abs(np.diff(np.log10(spheredist_AL_count2_avg))) > 0.14)[0][0]

poptD_AL, pcovD_AL = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize[iarg:farg]), 
                                              np.log10(spheredist_AL_count2_avg[iarg:farg]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL = np.sqrt(np.diag(pcovD_AL))

fitYD_calyx = objFuncPpow(radiussize, poptD_calyx[0], poptD_calyx[1])
fitYD_LH = objFuncPpow(radiussize, poptD_LH[0], poptD_LH[1])
fitYD_AL = objFuncPpow(radiussize, poptD_AL[0], poptD_AL[1])

fig = plt.figure(figsize=(8,6))

plt.scatter(radiussize, 
                    spheredist_AL_count2_avg, color='tab:blue', facecolors='none', alpha=0.5)
plt.scatter(radiussize, 
                    spheredist_calyx_count2_avg, color='tab:orange', facecolors='none', alpha=0.5)
plt.scatter(radiussize, 
                    spheredist_LH_count2_avg, color='tab:green', facecolors='none', alpha=0.5)


plt.plot(radiussize, fitYD_AL, lw=2, color='tab:blue')
plt.plot(radiussize, fitYD_calyx, lw=2, color='tab:orange')
plt.plot(radiussize, fitYD_LH, lw=2, color='tab:green')
plt.yscale('log')
plt.xscale('log')
plt.legend(['AL: ' + str(round(poptD_AL[0], 3)) + '$\pm$' + str(round(perrD_AL[0], 3)),
            'MB calyx: ' + str(round(poptD_calyx[0], 3)) + '$\pm$' + str(round(perrD_calyx[0], 3)),
            'LH: ' + str(round(poptD_LH[0], 3)) + '$\pm$' + str(round(perrD_LH[0], 3))], fontsize=13)
#plt.xlim(1, 75)
#plt.ylim(3, 1500)
#plt.tight_layout()
plt.xlabel("Radius $r$", fontsize=15)
plt.ylabel("Count", fontsize=15)
# plt.savefig(Parameter.outputdir + '/density_scale_neuropil_per_n_count2.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Moving window

Calyxmw = []
Calyxmwerr = []
LHmw = []
LHmwerr = []
ALmw = []
ALmwerr = []
mwx = []
shiftN = 10
for i in range(len(radiussize) - shiftN):
    mwx.append(np.average(radiussize[i:i+shiftN]))
    
    poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(radiussize[i:i+shiftN]), 
                                                np.log10(spheredist_calyx_sum_avg[i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    Calyxmw.append(poptmxc[0])
    Calyxmwerr.append(np.sqrt(np.diag(pcovmxc))[0])
    
    poptmxl, pcovmxl = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(radiussize[i:i+shiftN]), 
                                                np.log10(spheredist_LH_sum_avg[i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    LHmw.append(poptmxl[0])
    LHmwerr.append(np.sqrt(np.diag(pcovmxl))[0])
    
    poptmxa, pcovmxa = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(radiussize[i:i+shiftN]), 
                                                np.log10(spheredist_AL_sum_avg[i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    ALmw.append(poptmxa[0])
    ALmwerr.append(np.sqrt(np.diag(pcovmxa))[0])
    

fig = plt.figure(figsize=(12,8))
plt.plot(mwx[:49-shiftN], Calyxmw[:49-shiftN], lw=2)
plt.plot(mwx[:53-shiftN], LHmw[:53-shiftN], lw=2)
plt.plot(mwx, ALmw, lw=2)
plt.fill_between(mwx[:49-shiftN], np.array(Calyxmw[:49-shiftN])-np.array(Calyxmwerr[:49-shiftN]), 
                 np.array(Calyxmw[:49-shiftN])+np.array(Calyxmwerr[:49-shiftN]), alpha=0.3)
plt.fill_between(mwx[:53-shiftN], np.array(LHmw[:53-shiftN])-np.array(LHmwerr[:53-shiftN]),
                 np.array(LHmw[:53-shiftN])+np.array(LHmwerr[:53-shiftN]), alpha=0.3)
plt.fill_between(mwx, np.array(ALmw)-np.array(ALmwerr), np.array(ALmw)+np.array(ALmwerr), alpha=0.3)
plt.xscale('log')
plt.legend(["Calyx", "LH", "AL"], fontsize=15)
# plt.yscale('log')
#plt.xlim(1, 75)
#plt.ylim(3, 1500)
#plt.tight_layout()
plt.xlabel("Radius", fontsize=15)
plt.ylabel("Slope", fontsize=15)
plt.show()



#%%
 
radiussize_all = np.logspace(1, 3, 100)[20:69]

spheredist_all_sum = np.empty((len(MorphData.neuron_id), len(radiussize_all)))

for m in range(len(MorphData.neuron_id)):
    for b in range(len(radiussize_all)):
        spheredist_all_temp = []
        
        for ib in range(len(BranchData.branch_dist[m])):
            inbound_all = np.where(np.sqrt(np.square(np.array(BranchData.branch_dist[m][ib])[:,0] - fullCM[0]) +
                                        np.square(np.array(BranchData.branch_dist[m][ib])[:,1] - fullCM[1]) +
                                        np.square(np.array(BranchData.branch_dist[m][ib])[:,2] - fullCM[2])) <= radiussize_all[b])[0]
            
            if len(inbound_all) > 1:
                val = np.array(BranchData.branch_dist[m][ib])[inbound_all]
                x = val[:,0]
                y = val[:,1]
                z = val[:,2]
                
                xd = [j-i for i, j in zip(x[:-1], x[1:])]
                yd = [j-i for i, j in zip(y[:-1], y[1:])]
                zd = [j-i for i, j in zip(z[:-1], z[1:])]
                dist_all = np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                spheredist_all_temp.append(dist_all)
            else:
                spheredist_all_temp.append(0)

        spheredist_all_sum[m][b] = np.sum(spheredist_all_temp)



#%%
   
radiussize_all_inv = np.divide(1, 4/3*np.pi*np.power(radiussize_all, 3))

spheredist_all_sum[spheredist_all_sum == 0] = np.nan

spheredist_all_sum_avg = np.nanmean(spheredist_all_sum, axis=0)

spheredist_all_sum_avg = spheredist_all_sum_avg[np.count_nonzero(~np.isnan(spheredist_all_sum), axis=0) >= 10]

poptD_all, pcovD_all = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(radiussize_all_inv[:30]), 
                                                    np.log10(spheredist_all_sum_avg[:30]),
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_all = np.sqrt(np.diag(pcovD_all))

fitYD_all = objFuncPpow(radiussize_all_inv, poptD_all[0], poptD_all[1])

fig = plt.figure(figsize=(12,8))

plt.scatter(radiussize_all_inv, 
                    spheredist_all_sum_avg, c='tab:blue')

plt.plot(radiussize_all_inv, fitYD_all, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['All: ' + str(round(poptD_all[0], 3)) + '$\pm$' + str(round(perrD_all[0], 3))], fontsize=15)
plt.xlim(1e-8, 5e-5)
#plt.tight_layout()
plt.xlabel("Density", fontsize=15)
plt.ylabel("Length", fontsize=15)
plt.show()


#%%


#m = 6
#
#fig = plt.figure(figsize=(24, 16))
#ax = plt.axes(projection='3d')
#ax.set_xlim(400, 600)
#ax.set_ylim(150, 400)
#ax.set_zlim(50, 200)
#cmap = cm.get_cmap('viridis', len(MorphData.morph_id))
#
#tararr = np.array(MorphData.morph_dist[m])
#somaIdx = np.where(np.array(MorphData.morph_parent[m]) < 0)[0]
#for p in range(len(MorphData.morph_parent[m])):
#    if MorphData.morph_parent[m][p] < 0:
#        pass
#    else:
#        morph_line = np.vstack((MorphData.morph_dist[m][MorphData.morph_id[m].index(MorphData.morph_parent[m][p])], MorphData.morph_dist[m][p]))
#        ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], lw=1, color=cmap(m))
#ax.scatter3D(calyxCM[0], calyxCM[1], calyxCM[2], s=200)
#ax.scatter3D(LHCM[0], LHCM[1], LHCM[2], s=200)
#ax.scatter3D(ALCM[0], ALCM[1], ALCM[2], s=200)
#ax.legend(['Calyx', 'LH', 'AL'], fontsize=15)
#leg = ax.get_legend()
#leg.legendHandles[0].set_color('tab:blue')
#leg.legendHandles[1].set_color('tab:orange')
#leg.legendHandles[2].set_color('tab:green')
#plt.show()




#%% Dimension calculation

radiussize = np.multiply(2, np.logspace(-1, 2, 100)[0:99:3])

dist_len_dim = np.empty((len(radiussize), len(MorphData.neuron_id)))

for r in range(len(radiussize)):
    for b in range(len(BranchData.branch_dist)):
        dist_len_dim_temp = []
        for bd in range(len(BranchData.branch_dist[b])):
            bdi = 0
            taridx = 0
            
            while bdi != len(BranchData.branch_dist[b][bd]):
                rhs = BranchData.branch_dist[b][bd][taridx]
                lhs = BranchData.branch_dist[b][bd][bdi]
                dist = np.linalg.norm(np.subtract(rhs, lhs))
                if dist >= radiussize[r]:
                    taridx = bdi
                    dist_len_dim_temp.append(dist)
                bdi += 1
        dist_len_dim[r][b] = np.sum(dist_len_dim_temp)

#dist_len_dim[dist_len_dim == 0] = np.nan

#%%

dist_len_dim_avg = np.nanmean(dist_len_dim, axis=1)

poptDim_all, pcovDim_all = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(radiussize[10:26]), 
                                                    np.log10(dist_len_dim_avg[10:26]),
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrDim_all = np.sqrt(np.diag(pcovDim_all))

fitYDim_all = objFuncPpow(radiussize, poptDim_all[0], poptDim_all[1])

fig = plt.figure(figsize=(12,8))
#for i in range(len(MorphData.neuron_id)):
#    plt.scatter(radiussize, dist_len_dim[:,i])
plt.scatter(radiussize[:26], dist_len_dim_avg[:26])
plt.plot(radiussize[:26], fitYDim_all[:26], lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['All: ' + str(round(poptDim_all[0], 3)) + '$\pm$' + str(round(perrDim_all[0], 3))], fontsize=15)
#plt.ylim(3, 1500)
#plt.tight_layout()
plt.xlabel("Diameter", fontsize=15)
plt.ylabel("Length", fontsize=15)
plt.show()


#%%

radiussize = np.multiply(2, np.logspace(-1, 1, 100)[0:99:3])

dist_len_calyx_dim = np.empty((len(radiussize), len(MorphData.calyxdist)))
dist_len_LH_dim = np.empty((len(radiussize), len(MorphData.LHdist)))
dist_len_AL_dim = np.empty((len(radiussize), len(MorphData.ALdist)))

for r in range(len(radiussize)):
    for c in range(len(MorphData.calyxdist)):
        dist_len_calyx_dim_temp = []
        cdi = 0
        taridx = 0
        
        while cdi != len(MorphData.calyxdist[c]):
            rhs = MorphData.calyxdist[c][taridx]
            lhs = MorphData.calyxdist[c][cdi]
            dist = np.linalg.norm(np.subtract(rhs, lhs))
            if dist >= radiussize[r]:
                taridx = cdi
                dist_len_calyx_dim_temp.append(dist)
            cdi += 1
        dist_len_calyx_dim[r][c] = np.sum(dist_len_calyx_dim_temp)
    for l in range(len(MorphData.LHdist)):
        dist_len_LH_dim_temp = []
        ldi = 0
        taridx = 0
        
        while ldi != len(MorphData.LHdist[l]):
            rhs = MorphData.LHdist[l][taridx]
            lhs = MorphData.LHdist[l][ldi]
            dist = np.linalg.norm(np.subtract(rhs, lhs))
            if dist >= radiussize[r]:
                taridx = ldi
                dist_len_LH_dim_temp.append(dist)
            ldi += 1
        dist_len_LH_dim[r][l] = np.sum(dist_len_LH_dim_temp)
    for a in range(len(MorphData.ALdist)):
        dist_len_AL_dim_temp = []
        adi = 0
        taridx = 0
        
        while adi != len(MorphData.ALdist[a]):
            rhs = MorphData.ALdist[a][taridx]
            lhs = MorphData.ALdist[a][adi]
            dist = np.linalg.norm(np.subtract(rhs, lhs))
            if dist >= radiussize[r]:
                taridx = adi
                dist_len_AL_dim_temp.append(dist)
            adi += 1
        dist_len_AL_dim[r][a] = np.sum(dist_len_AL_dim_temp)

#dist_len_calyx_dim[dist_len_calyx_dim == 0] = np.nan
#dist_len_LH_dim[dist_len_LH_dim == 0] = np.nan
#dist_len_AL_dim[dist_len_AL_dim == 0] = np.nan

#%%

dist_len_calyx_dim_avg = np.nanmean(dist_len_calyx_dim, axis=1)
dist_len_LH_dim_avg = np.nanmean(dist_len_LH_dim, axis=1)
dist_len_AL_dim_avg = np.nanmean(dist_len_AL_dim, axis=1)

poptDim_calyx, pcovDim_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(radiussize[13:23]), 
                                                        np.log10(dist_len_calyx_dim_avg[13:23]),
                                                        p0=[-0.1, 0.1], 
                                                        maxfev=10000)
perrDim_calyx = np.sqrt(np.diag(pcovDim_calyx))

poptDim_LH, pcovDim_LH = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(radiussize[13:23]), 
                                                        np.log10(dist_len_LH_dim_avg[13:23]),
                                                        p0=[-0.1, 0.1], 
                                                        maxfev=10000)
perrDim_LH = np.sqrt(np.diag(pcovDim_LH))

poptDim_AL, pcovDim_AL = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(radiussize[13:23]), 
                                                        np.log10(dist_len_AL_dim_avg[13:23]),
                                                        p0=[-0.1, 0.1], 
                                                        maxfev=10000)
perrDim_AL = np.sqrt(np.diag(pcovDim_AL))

fitYDim_calyx = objFuncPpow(radiussize, poptDim_calyx[0], poptDim_calyx[1])
fitYDim_LH = objFuncPpow(radiussize, poptDim_LH[0], poptDim_LH[1])
fitYDim_AL = objFuncPpow(radiussize, poptDim_AL[0], poptDim_AL[1])

fig = plt.figure(figsize=(12,8))
#for i in range(len(MorphData.neuron_id)):
#    plt.scatter(radiussize, dist_len_dim[:,i])
plt.scatter(radiussize[:23], dist_len_calyx_dim_avg[:23])
plt.scatter(radiussize[:23], dist_len_LH_dim_avg[:23])
plt.scatter(radiussize[:23], dist_len_AL_dim_avg[:23])
plt.plot(radiussize[:23], fitYDim_calyx[:23], lw=2, linestyle='--')
plt.plot(radiussize[:23], fitYDim_LH[:23], lw=2, linestyle='--')
plt.plot(radiussize[:23], fitYDim_AL[:23], lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['Calyx: ' + str(round(poptDim_calyx[0], 3)) + '$\pm$' + str(round(perrDim_calyx[0], 3)),
            'LH: ' + str(round(poptDim_LH[0], 3)) + '$\pm$' + str(round(perrDim_LH[0], 3)),
            'AL: ' + str(round(poptDim_AL[0], 3)) + '$\pm$' + str(round(perrDim_AL[0], 3))], fontsize=15)
#plt.xlim(0.1, 10)
#plt.tight_layout()
plt.xlabel("Diameter", fontsize=15)
plt.ylabel("Length", fontsize=15)
plt.show()

t8 = time.time()

print('checkpoint 8: ' + str(t8-t7))

#%% Fractal dimension using binary box counting


binsize = np.logspace(-1, 3, 100)[13:99:3]

xmax_all = np.max(MorphData.morph_dist_flat[:,0])
xmin_all = np.min(MorphData.morph_dist_flat[:,0])
ymax_all = np.max(MorphData.morph_dist_flat[:,1])
ymin_all = np.min(MorphData.morph_dist_flat[:,1])
zmax_all = np.max(MorphData.morph_dist_flat[:,2])
zmin_all = np.min(MorphData.morph_dist_flat[:,2])

hlist = []
hlist_count = []
hlist_numbox = []

for b in range(len(binsize)):
    xbin = np.arange(xmin_all-0.01, xmax_all+binsize[b], binsize[b])
    ybin = np.arange(ymin_all-0.01, ymax_all+binsize[b], binsize[b])
    zbin = np.arange(zmin_all-0.01, zmax_all+binsize[b], binsize[b])
    if len(xbin) == 1:
        xbin = [-1000, 1000]
    if len(ybin) == 1:
        ybin = [-1000, 1000]
    if len(zbin) == 1:
        zbin = [-1000, 1000]
        
    h, e = np.histogramdd(MorphData.morph_dist_flat, 
                          bins=[xbin, 
                                ybin,
                                zbin])
    # hlist.append(h)
    hlist_count.append(np.count_nonzero(h))
    # hlist_numbox.append((len(xbin)-1)*
    #                     (len(ybin)-1)*
    #                     (len(zbin)-1))


#%%

farg = np.where(np.abs(np.diff(np.log10(hlist_count))) > 0.22)[0][-1]#np.argwhere(np.array(hlist_count) > 1)[-1][0] + 2
iarg =  np.where(np.abs(np.diff(np.log10(hlist_count))) > 0.22)[0][0]

poptBcount_all, pcovBcount_all = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_count[iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_all = np.sqrt(np.diag(pcovBcount_all))

fitYBcount_all = objFuncPpow(binsize, poptBcount_all[0], poptBcount_all[1])

fig = plt.figure(figsize=(8,6))
plt.scatter(1/binsize, hlist_count)
plt.plot(1/binsize, fitYBcount_all, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['FD: ' + str(round(-poptBcount_all[0], 3)) + '$\pm$' + str(round(-perrBcount_all[0], 3))], fontsize=13)
#plt.xlim(0.1, 20)
#plt.tight_layout()
plt.xlabel("Box Size $l$", fontsize=15)
plt.ylabel("Count", fontsize=15)
# plt.savefig(Parameter.outputdir + '/fd_whole_2.pdf', dpi=300, bbox_inches='tight')
plt.show()



#%% Fractal dimension using binary box counting for each region

binsize = np.logspace(-1, 3, 100)[13:90:3]

hlist_calyx_count = []
hlist_LH_count = []
hlist_AL_count = []

for b in range(len(binsize)):
    xbin_calyx = np.arange(xmin_calyx-0.01, xmax_calyx+binsize[b], binsize[b])
    ybin_calyx = np.arange(ymin_calyx-0.01, ymax_calyx+binsize[b], binsize[b])
    zbin_calyx = np.arange(zmin_calyx-0.01, zmax_calyx+binsize[b], binsize[b])
    if len(xbin_calyx) == 1:
        xbin_calyx = [-1000, 1000]
    if len(ybin_calyx) == 1:
        ybin_calyx = [-1000, 1000]
    if len(zbin_calyx) == 1:
        zbin_calyx = [-1000, 1000]
    
    hc, e = np.histogramdd(calyx_dist_flat, 
                          bins=[xbin_calyx, 
                                ybin_calyx,
                                zbin_calyx])
    hlist_calyx_count.append(np.count_nonzero(hc))
    
    xbin_LH = np.arange(xmin_LH-0.01, xmax_LH+binsize[b], binsize[b])
    ybin_LH = np.arange(ymin_LH-0.01, ymax_LH+binsize[b], binsize[b])
    zbin_LH = np.arange(zmin_LH-0.01, zmax_LH+binsize[b], binsize[b])
    if len(xbin_LH) == 1:
        xbin_LH = [-1000, 1000]
    if len(ybin_LH) == 1:
        ybin_LH = [-1000, 1000]
    if len(zbin_LH) == 1:
        zbin_LH = [-1000, 1000]
    
    hh, e = np.histogramdd(LH_dist_flat, 
                          bins=[xbin_LH, 
                                ybin_LH,
                                zbin_LH])
    hlist_LH_count.append(np.count_nonzero(hh))
    
    xbin_AL = np.arange(xmin_AL-0.01, xmax_AL+binsize[b], binsize[b])
    ybin_AL = np.arange(ymin_AL-0.01, ymax_AL+binsize[b], binsize[b])
    zbin_AL = np.arange(zmin_AL-0.01, zmax_AL+binsize[b], binsize[b])
    if len(xbin_AL) == 1:
        xbin_AL = [-1000, 1000]
    if len(ybin_AL) == 1:
        ybin_AL = [-1000, 1000]
    if len(zbin_AL) == 1:
        zbin_AL = [-1000, 1000]
        
    ha, e = np.histogramdd(AL_dist_flat, 
                          bins=[xbin_AL, 
                                ybin_AL,
                                zbin_AL])
    hlist_AL_count.append(np.count_nonzero(ha))




#%%

farg = np.where(np.abs(np.diff(np.log10(hlist_calyx_count))) > 0.22)[0][-1]#np.argwhere(np.array(hlist_calyx_count) > 1)[-1][0] + 2
iarg = np.where(np.abs(np.diff(np.log10(hlist_calyx_count))) > 0.22)[0][0]
poptBcount_calyx, pcovBcount_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_calyx_count[iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_calyx = np.sqrt(np.diag(pcovBcount_calyx))

farg = np.where(np.abs(np.diff(np.log10(hlist_LH_count))) > 0.22)[0][-1]#np.argwhere(np.array(hlist_LH_count) > 1)[-1][0] + 2
iarg = np.where(np.abs(np.diff(np.log10(hlist_LH_count))) > 0.22)[0][0]
poptBcount_LH, pcovBcount_LH = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_LH_count[iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_LH = np.sqrt(np.diag(pcovBcount_LH))

farg = np.where(np.abs(np.diff(np.log10(hlist_AL_count))) > 0.22)[0][-1]#np.argwhere(np.array(hlist_AL_count) > 1)[-1][0] + 2
iarg = np.where(np.abs(np.diff(np.log10(hlist_AL_count))) > 0.22)[0][0]
poptBcount_AL, pcovBcount_AL = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_AL_count[iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_AL = np.sqrt(np.diag(pcovBcount_AL))

fitYBcount_calyx = objFuncPpow(binsize, poptBcount_calyx[0], poptBcount_calyx[1])
fitYBcount_LH = objFuncPpow(binsize, poptBcount_LH[0], poptBcount_LH[1])
fitYBcount_AL = objFuncPpow(binsize, poptBcount_AL[0], poptBcount_AL[1])
    
fig = plt.figure(figsize=(12,8))
plt.scatter(binsize, hlist_calyx_count)
plt.scatter(binsize, hlist_LH_count)
plt.scatter(binsize, hlist_AL_count)
plt.plot(binsize, fitYBcount_calyx, lw=2, linestyle='--')
plt.plot(binsize, fitYBcount_LH, lw=2, linestyle='--')
plt.plot(binsize, fitYBcount_AL, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['Calyx: ' + str(round(poptBcount_calyx[0], 3)) + '$\pm$' + str(round(perrBcount_calyx[0], 3)),
            'LH: ' + str(round(poptBcount_LH[0], 3)) + '$\pm$' + str(round(perrBcount_LH[0], 3)),
            'AL: ' + str(round(poptBcount_AL[0], 3)) + '$\pm$' + str(round(perrBcount_AL[0], 3))], fontsize=15)
#plt.xlim(0.1, 20)
#plt.tight_layout()
plt.xlabel("Box Size", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()


#%% Fractal dimension using binary box counting for each region in sub-physiological scale

binsize = np.logspace(-2, 3, 100)[20:90:3]

sp_l1 = 0

max_calyx_b = calyxCM + sp_l1 + [(xmax_calyx - xmin_calyx)/3, (ymax_calyx - ymin_calyx)/3, (zmax_calyx - zmin_calyx)/3]
min_calyx_b = calyxCM - sp_l1 - [(xmax_calyx - xmin_calyx)/3, (ymax_calyx - ymin_calyx)/3, (zmax_calyx - zmin_calyx)/3]

max_LH_b = LHCM + sp_l1 + [(xmax_LH - xmin_LH)/3, (ymax_LH - ymin_LH)/3, (zmax_LH - zmin_LH)/3]
min_LH_b = LHCM - sp_l1 - [(xmax_LH - xmin_LH)/3, (ymax_LH - ymin_LH)/3, (zmax_LH - zmin_LH)/3]

max_AL_b = ALCM + sp_l1 + [(xmax_AL - xmin_AL)/3, (ymax_AL - ymin_AL)/3, (zmax_AL - zmin_AL)/3]
min_AL_b = ALCM - sp_l1 - [(xmax_AL - xmin_AL)/3, (ymax_AL - ymin_AL)/3, (zmax_AL - zmin_AL)/3]

hlist_calyx_b = []
hlist_calyx_b_count = []
hlist_calyx_b_numbox = []
hlist_LH_b = []
hlist_LH_b_count = []
hlist_LH_b_numbox = []
hlist_AL_b = []
hlist_AL_b_count = []
hlist_AL_b_numbox = []


for b in range(len(binsize)):
    xbin_calyx_b = np.arange(min_calyx_b[0], max_calyx_b[0]+binsize[b], binsize[b])
    ybin_calyx_b = np.arange(min_calyx_b[1], max_calyx_b[1]+binsize[b], binsize[b])
    zbin_calyx_b = np.arange(min_calyx_b[2], max_calyx_b[2]+binsize[b], binsize[b])
    if len(xbin_calyx_b) == 1:
        xbin_calyx_b = [-1000, 1000]
    if len(ybin_calyx_b) == 1:
        ybin_calyx_b = [-1000, 1000]
    if len(zbin_calyx_b) == 1:
        zbin_calyx_b = [-1000, 1000]
    
    hc, e = np.histogramdd(calyx_dist_flat, 
                          bins=[xbin_calyx_b, 
                                ybin_calyx_b,
                                zbin_calyx_b])
    hlist_calyx_b.append(hc)
    hlist_calyx_b_count.append(np.count_nonzero(hc))
    hlist_calyx_b_numbox.append((len(xbin_calyx_b)-1)*
                              (len(ybin_calyx_b)-1)*
                              (len(zbin_calyx_b)-1))
    
    xbin_LH_b = np.arange(min_LH_b[0], max_LH_b[0]+binsize[b], binsize[b])
    ybin_LH_b = np.arange(min_LH_b[1], max_LH_b[1]+binsize[b], binsize[b])
    zbin_LH_b = np.arange(min_LH_b[2], max_LH_b[2]+binsize[b], binsize[b])
    if len(xbin_LH_b) == 1:
        xbin_LH_b = [-1000, 1000]
    if len(ybin_LH_b) == 1:
        ybin_LH_b = [-1000, 1000]
    if len(zbin_LH_b) == 1:
        zbin_LH_b = [-1000, 1000]
    
    hh, e = np.histogramdd(LH_dist_flat, 
                          bins=[xbin_LH_b, 
                                ybin_LH_b,
                                zbin_LH_b])
    hlist_LH_b.append(hh)
    hlist_LH_b_count.append(np.count_nonzero(hh))
    hlist_LH_b_numbox.append((len(xbin_LH_b)-1)*
                           (len(ybin_LH_b)-1)*
                           (len(zbin_LH_b)-1))
    
    xbin_AL_b = np.arange(min_AL_b[0], max_AL_b[0]+binsize[b], binsize[b])
    ybin_AL_b = np.arange(min_AL_b[1], max_AL_b[1]+binsize[b], binsize[b])
    zbin_AL_b = np.arange(min_AL_b[2], max_AL_b[2]+binsize[b], binsize[b])
    if len(xbin_AL_b) == 1:
        xbin_AL_b = [-1000, 1000]
    if len(ybin_AL_b) == 1:
        ybin_AL_b = [-1000, 1000]
    if len(zbin_AL_b) == 1:
        zbin_AL_b = [-1000, 1000]
        
    ha, e = np.histogramdd(AL_dist_flat, 
                          bins=[xbin_AL_b, 
                                ybin_AL_b,
                                zbin_AL_b])
    hlist_AL_b.append(ha)
    hlist_AL_b_count.append(np.count_nonzero(ha))
    hlist_AL_b_numbox.append((len(xbin_AL_b)-1)*
                           (len(ybin_AL_b)-1)*
                           (len(zbin_AL_b)-1))




#%%
    
farg = np.where(np.abs(np.diff(np.log10(hlist_calyx_b_count))) > 0.22)[0][-1]#np.argwhere(np.array(hlist_calyx_b_count) > 1)[-1][0] + 2
iarg =  np.where(np.abs(np.diff(np.log10(hlist_calyx_b_count))) > 0.22)[0][0]
poptBcount_calyx_b, pcovBcount_calyx_b = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_calyx_b_count[iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_calyx_b = np.sqrt(np.diag(pcovBcount_calyx_b))

farg = np.where(np.abs(np.diff(np.log10(hlist_LH_b_count))) > 0.22)[0][-1]#np.argwhere(np.array(hlist_LH_b_count) > 1)[-1][0] + 2
iarg = np.where(np.abs(np.diff(np.log10(hlist_LH_b_count))) > 0.22)[0][0]
poptBcount_LH_b, pcovBcount_LH_b = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_LH_b_count[iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_LH_b = np.sqrt(np.diag(pcovBcount_LH_b))

farg = np.where(np.abs(np.diff(np.log10(hlist_AL_b_count))) > 0.22)[0][-1]#np.argwhere(np.array(hlist_AL_b_count) > 1)[-1][0] + 2
iarg = np.where(np.abs(np.diff(np.log10(hlist_AL_b_count))) > 0.22)[0][0]
poptBcount_AL_b, pcovBcount_AL_b = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_AL_b_count[iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_AL_b = np.sqrt(np.diag(pcovBcount_AL_b))

fitYBcount_calyx_b = objFuncPpow(binsize, poptBcount_calyx_b[0], poptBcount_calyx_b[1])
fitYBcount_LH_b = objFuncPpow(binsize, poptBcount_LH_b[0], poptBcount_LH_b[1])
fitYBcount_AL_b = objFuncPpow(binsize, poptBcount_AL_b[0], poptBcount_AL_b[1])
    
fig = plt.figure(figsize=(8,6))
plt.scatter(1/binsize, hlist_AL_b_count)
plt.scatter(1/binsize, hlist_calyx_b_count)
plt.scatter(1/binsize, hlist_LH_b_count)
plt.plot(1/binsize, fitYBcount_AL_b, lw=2, linestyle='--')
plt.plot(1/binsize, fitYBcount_calyx_b, lw=2, linestyle='--')
plt.plot(1/binsize, fitYBcount_LH_b, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['AL: ' + str(round(-poptBcount_AL_b[0], 3)) + '$\pm$' + str(round(perrBcount_AL_b[0], 3)),
            'MB calyx: ' + str(round(-poptBcount_calyx_b[0], 3)) + '$\pm$' + str(round(perrBcount_calyx_b[0], 3)),
            'LH: ' + str(round(-poptBcount_LH_b[0], 3)) + '$\pm$' + str(round(perrBcount_LH_b[0], 3))], fontsize=13)
# plt.xlim(3e-3, 4)
#plt.tight_layout()
plt.xlabel("$1/l$", fontsize=15)
plt.ylabel("Count", fontsize=15)
# plt.savefig(Parameter.outputdir + '/fd_neuropil_fixed_4.pdf', dpi=300, bbox_inches='tight')
plt.show()



#%% Binary Box-counting for Sub-physiological Region Length Scale

binsize = np.logspace(-1, 3, 100)[13:85:2]

sp_l = np.arange(10, 51, 2.5)
bbr = 1

hlist_calyx_b_count = np.empty((bbr, len(sp_l), len(binsize)), dtype=int)
hlist_LH_b_count = np.empty((bbr, len(sp_l), len(binsize)), dtype=int)
hlist_AL_b_count = np.empty((bbr, len(sp_l), len(binsize)), dtype=int)

for r in range(bbr):
    for l in range(len(sp_l)):
        
        calyx_rand = np.array([np.random.uniform(-(xmax_calyx - xmin_calyx)/20, (xmax_calyx - xmin_calyx)/20), 
                               np.random.uniform(-(ymax_calyx - ymin_calyx)/20, (ymax_calyx - ymin_calyx)/20),
                               np.random.uniform(-(zmax_calyx - zmin_calyx)/20, (zmax_calyx - zmin_calyx)/20)])
        max_calyx_b = calyxCM + sp_l[l]# + calyx_rand
        min_calyx_b = calyxCM - sp_l[l]# + calyx_rand
        
        LH_rand = np.array([np.random.uniform(-(xmax_LH - xmin_LH)/20, (xmax_LH - xmin_LH)/20), 
                            np.random.uniform(-(ymax_LH - ymin_LH)/20, (ymax_LH - ymin_LH)/20),
                            np.random.uniform(-(zmax_LH - zmin_LH)/20, (zmax_LH - zmin_LH)/20)])
        
        max_LH_b = LHCM + sp_l[l]# + LH_rand
        min_LH_b = LHCM - sp_l[l]# + LH_rand
        
        AL_rand = np.array([np.random.uniform(-(xmax_AL - xmin_AL)/20, (xmax_AL - xmin_AL)/20), 
                            np.random.uniform(-(ymax_AL - ymin_AL)/20, (ymax_AL - ymin_AL)/20),
                            np.random.uniform(-(zmax_AL - zmin_AL)/20, (zmax_AL - zmin_AL)/20)])
        
        max_AL_b = ALCM + sp_l[l]# + AL_rand
        min_AL_b = ALCM - sp_l[l]# + AL_rand
        
        for b in range(len(binsize)):
            xbin_calyx_b = np.arange(min_calyx_b[0], max_calyx_b[0]+binsize[b], binsize[b])
            ybin_calyx_b = np.arange(min_calyx_b[1], max_calyx_b[1]+binsize[b], binsize[b])
            zbin_calyx_b = np.arange(min_calyx_b[2], max_calyx_b[2]+binsize[b], binsize[b])
            if len(xbin_calyx_b) == 1:
                xbin_calyx_b = [-1000, 1000]
            if len(ybin_calyx_b) == 1:
                ybin_calyx_b = [-1000, 1000]
            if len(zbin_calyx_b) == 1:
                zbin_calyx_b = [-1000, 1000]
            
            hc, e = np.histogramdd(calyx_dist_flat, 
                                  bins=[xbin_calyx_b, 
                                        ybin_calyx_b,
                                        zbin_calyx_b])
            hlist_calyx_b_count[r][l][b] = np.count_nonzero(hc)
            
            xbin_LH_b = np.arange(min_LH_b[0], max_LH_b[0]+binsize[b], binsize[b])
            ybin_LH_b = np.arange(min_LH_b[1], max_LH_b[1]+binsize[b], binsize[b])
            zbin_LH_b = np.arange(min_LH_b[2], max_LH_b[2]+binsize[b], binsize[b])
            if len(xbin_LH_b) == 1:
                xbin_LH_b = [-1000, 1000]
            if len(ybin_LH_b) == 1:
                ybin_LH_b = [-1000, 1000]
            if len(zbin_LH_b) == 1:
                zbin_LH_b = [-1000, 1000]
            
            hh, e = np.histogramdd(LH_dist_flat, 
                                  bins=[xbin_LH_b, 
                                        ybin_LH_b,
                                        zbin_LH_b])
            hlist_LH_b_count[r][l][b] = np.count_nonzero(hh)
            
            xbin_AL_b = np.arange(min_AL_b[0], max_AL_b[0]+binsize[b], binsize[b])
            ybin_AL_b = np.arange(min_AL_b[1], max_AL_b[1]+binsize[b], binsize[b])
            zbin_AL_b = np.arange(min_AL_b[2], max_AL_b[2]+binsize[b], binsize[b])
            if len(xbin_AL_b) == 1:
                xbin_AL_b = [-1000, 1000]
            if len(ybin_AL_b) == 1:
                ybin_AL_b = [-1000, 1000]
            if len(zbin_AL_b) == 1:
                zbin_AL_b = [-1000, 1000]
                
            ha, e = np.histogramdd(AL_dist_flat, 
                                  bins=[xbin_AL_b, 
                                        ybin_AL_b,
                                        zbin_AL_b])
            hlist_AL_b_count[r][l][b] = np.count_nonzero(ha)




#%%

poptBcount_calyx_b = np.empty((bbr, len(sp_l), 2))
perrBcount_calyx_b = np.empty((bbr, len(sp_l), 2))
poptBcount_LH_b = np.empty((bbr, len(sp_l), 2))
perrBcount_LH_b = np.empty((bbr, len(sp_l), 2))
poptBcount_AL_b = np.empty((bbr, len(sp_l), 2))
perrBcount_AL_b = np.empty((bbr, len(sp_l), 2))
fitYBcount_calyx_b = np.empty((bbr, len(sp_l), len(binsize)))
fitYBcount_LH_b = np.empty((bbr, len(sp_l), len(binsize)))
fitYBcount_AL_b = np.empty((bbr, len(sp_l), len(binsize)))

for r in range(bbr):
    for l in range(len(sp_l)):
        farg = np.where(np.abs(np.diff(np.log10(hlist_calyx_b_count[r][l]))) > 0.15)[0][-1]#np.argwhere(np.array(hlist_calyx_b_count[r][l]) > 1)[-1][0] + 2
        iarg = np.where(np.abs(np.diff(np.log10(hlist_calyx_b_count[r][l]))) > 0.15)[0][0]
        if iarg < 0:
            iarg = 0
        poptBcount_calyx_b_t, pcovBcount_calyx_b_t = scipy.optimize.curve_fit(objFuncGL, 
                                                                np.log10(binsize[iarg:farg]), 
                                                                np.log10(hlist_calyx_b_count[r][l][iarg:farg]),
                                                                p0=[0.1, 0.1], 
                                                                maxfev=10000)
        perrBcount_calyx_b_t = np.sqrt(np.diag(pcovBcount_calyx_b_t))
        
        farg = np.where(np.abs(np.diff(np.log10(hlist_LH_b_count[r][l]))) > 0.15)[0][-1]#np.argwhere(np.array(hlist_LH_b_count[r][l]) > 1)[-1][0] + 2
        iarg = np.where(np.abs(np.diff(np.log10(hlist_LH_b_count[r][l]))) > 0.15)[0][0]
        if iarg < 0:
            iarg = 0
        poptBcount_LH_b_t, pcovBcount_LH_b_t = scipy.optimize.curve_fit(objFuncGL, 
                                                                np.log10(binsize[iarg:farg]), 
                                                                np.log10(hlist_LH_b_count[r][l][iarg:farg]),
                                                                p0=[0.1, 0.1], 
                                                                maxfev=10000)
        perrBcount_LH_b_t = np.sqrt(np.diag(pcovBcount_LH_b_t))
        
        farg = np.where(np.abs(np.diff(np.log10(hlist_AL_b_count[r][l]))) > 0.15)[0][-1]#np.argwhere(np.array(hlist_AL_b_count[r][l]) > 1)[-1][0] + 2
        iarg = np.where(np.abs(np.diff(np.log10(hlist_AL_b_count[r][l]))) > 0.15)[0][0]
        if iarg < 0:
            iarg = 0
        poptBcount_AL_b_t, pcovBcount_AL_b_t = scipy.optimize.curve_fit(objFuncGL, 
                                                                np.log10(binsize[iarg:farg]), 
                                                                np.log10(hlist_AL_b_count[r][l][iarg:farg]),
                                                                p0=[0.1, 0.1], 
                                                                maxfev=10000)
        perrBcount_AL_b_t = np.sqrt(np.diag(pcovBcount_AL_b_t))
    
        fitYBcount_calyx_b[r][l] = objFuncPpow(binsize, poptBcount_calyx_b_t[0], poptBcount_calyx_b_t[1])
        fitYBcount_LH_b[r][l] = objFuncPpow(binsize, poptBcount_LH_b_t[0], poptBcount_LH_b_t[1])
        fitYBcount_AL_b[r][l] = objFuncPpow(binsize, poptBcount_AL_b_t[0], poptBcount_AL_b_t[1])
        
        poptBcount_calyx_b[r][l] = poptBcount_calyx_b_t
        perrBcount_calyx_b[r][l] = perrBcount_calyx_b_t
        poptBcount_LH_b[r][l] = poptBcount_LH_b_t
        perrBcount_LH_b[r][l] = perrBcount_LH_b_t
        poptBcount_AL_b[r][l] = poptBcount_AL_b_t
        perrBcount_AL_b[r][l] = perrBcount_AL_b_t
    
fig = plt.figure(figsize=(8,6))
for r in range(bbr):
    for l in range(len(sp_l)):
        plt.scatter(binsize, hlist_calyx_b_count[r][l], color='tab:blue')
        plt.scatter(binsize, hlist_LH_b_count[r][l], color='tab:orange')
        plt.scatter(binsize, hlist_AL_b_count[r][l], color='tab:green')
        plt.plot(binsize, fitYBcount_calyx_b[r][l], lw=1, linestyle='--', color='tab:blue')
        plt.plot(binsize, fitYBcount_LH_b[r][l], lw=1, linestyle='--', color='tab:orange')
        plt.plot(binsize, fitYBcount_AL_b[r][l], lw=1, linestyle='--', color='tab:green')
plt.yscale('log')
plt.xscale('log')
# plt.legend(['Calyx: ' + str(round(poptBcount_calyx_b[0], 3)) + '$\pm$' + str(round(perrBcount_calyx_b[0], 3)),
#             'LH: ' + str(round(poptBcount_LH_b[0], 3)) + '$\pm$' + str(round(perrBcount_LH_b[0], 3)),
#             'AL: ' + str(round(poptBcount_AL_b[0], 3)) + '$\pm$' + str(round(perrBcount_AL_b[0], 3))], fontsize=15)
plt.xlim(0.2, 350)
#plt.tight_layout()
plt.xlabel("Box Size", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()


#%%

poptBcount_calyx_b_avg = np.average(poptBcount_calyx_b[:,:,0], axis=0)
poptBcount_LH_b_avg = np.average(poptBcount_LH_b[:,:,0], axis=0)
poptBcount_AL_b_avg = np.average(poptBcount_AL_b[:,:,0], axis=0)

perrBcount_calyx_b_avg = np.sqrt(np.sum(np.square(perrBcount_calyx_b[:,:,0]), axis=0))/bbr
perrBcount_LH_b_avg = np.sqrt(np.sum(np.square(perrBcount_LH_b[:,:,0]), axis=0))/bbr
perrBcount_AL_b_avg = np.sqrt(np.sum(np.square(perrBcount_AL_b[:,:,0]), axis=0))/bbr

fig = plt.figure(figsize=(8,6))
plt.plot(2*sp_l, poptBcount_AL_b_avg, lw=2, linestyle='--', color='tab:blue')
plt.plot(2*sp_l, poptBcount_calyx_b_avg, lw=2, linestyle='--', color='tab:orange')
plt.plot(2*sp_l, poptBcount_LH_b_avg, lw=2, linestyle='--', color='tab:green')
plt.fill_between(2*sp_l, poptBcount_AL_b_avg-perrBcount_AL_b_avg, poptBcount_AL_b_avg+perrBcount_AL_b_avg, alpha=0.3)
plt.fill_between(2*sp_l, poptBcount_calyx_b_avg-perrBcount_calyx_b_avg, poptBcount_calyx_b_avg+perrBcount_calyx_b_avg, alpha=0.3)
plt.fill_between(2*sp_l, poptBcount_LH_b_avg-perrBcount_LH_b_avg, poptBcount_LH_b_avg+perrBcount_LH_b_avg, alpha=0.3)
plt.legend(['AL', 'MB calyx', 'LH'], fontsize=13)
#plt.tight_layout()
plt.xlabel("Sampled Bounding Cube Side Length $L$", fontsize=15)
plt.ylabel("Fractal Dimension", fontsize=15)
# plt.savefig(Parameter.outputdir + '/fd_neuropil_mv_win_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

t9 = time.time()

print('checkpoint 9: ' + str(t9-t8))

#%% Single Neuron Dimension Calculation using Binary Box-counting

binsize = np.logspace(-2, 3, 100)[25:95:2]

hlist_single_count = np.empty((len(MorphData.morph_dist), len(binsize)))

for i in range(len(MorphData.morph_dist)):
    morph_dist_single = np.array(MorphData.morph_dist[i])
    xmax_single = np.max(morph_dist_single[:,0])
    xmin_single = np.min(morph_dist_single[:,0])
    ymax_single = np.max(morph_dist_single[:,1])
    ymin_single = np.min(morph_dist_single[:,1])
    zmax_single = np.max(morph_dist_single[:,2])
    zmin_single = np.min(morph_dist_single[:,2])
    
    for b in range(len(binsize)):
        xbin = np.arange(xmin_single-0.01, xmax_single+binsize[b], binsize[b])
        ybin = np.arange(ymin_single-0.01, ymax_single+binsize[b], binsize[b])
        zbin = np.arange(zmin_single-0.01, zmax_single+binsize[b], binsize[b])
        if len(xbin) < 2:
            xbin = [-1000, 1000]
        if len(ybin) < 2:
            ybin = [-1000, 1000]
        if len(zbin) < 2:
            zbin = [-1000, 1000]
            
        h, e = np.histogramdd(morph_dist_single, bins=[xbin, ybin, zbin])
        hlist_single_count[i][b] = np.count_nonzero(h)
   


#%%

cmap = cm.get_cmap('viridis', len(MorphData.morph_dist))

poptBcount_single_list = []
pcovBcount_single_list = []

fig = plt.figure(figsize=(12,8))
for i in range(len(MorphData.morph_dist)):
    if len(MorphData.morph_dist[i]) > 700:
        farg = np.where(np.abs(np.diff(np.log10(hlist_single_count[i]))) > 0.14)[0][-1]#np.argwhere(hlist_single_count[i] > 1)[-1][0] + 2
        iarg = np.where(np.abs(np.diff(np.log10(hlist_single_count[i]))) > 0.14)[0][0]
        if iarg < 0:
            iarg = 0
        poptBcount_single, pcovBcount_single = scipy.optimize.curve_fit(objFuncGL, 
                                                            np.log10(binsize[iarg:farg]), 
                                                            np.log10(hlist_single_count[i][iarg:farg]),
                                                            p0=[0.1, 0.1], 
                                                            maxfev=10000)
        perrBcount_single = np.sqrt(np.diag(pcovBcount_single))
        
        poptBcount_single_list.append(poptBcount_single)
        pcovBcount_single_list.append(perrBcount_single)
        
        fitYBcount_single = objFuncPpow(binsize, poptBcount_single[0], poptBcount_single[1])
        plt.scatter(binsize, hlist_single_count[i], color=cmap(i))
        plt.plot(binsize, fitYBcount_single, lw=2, linestyle='--', color=cmap(i))
    
#plt.legend([str(i) + ': ' + str(round(poptBcount_single[0], 3)) + '$\pm$' + str(round(perrBcount_single[0], 3))], fontsize=15)    
plt.yscale('log')
plt.xscale('log')
#plt.xlim(0.1, 1000)
plt.ylim(0.1, 50000)
#plt.tight_layout()
plt.xlabel("Box Count", fontsize=15)
plt.ylabel("Count", fontsize=15)
#plt.savefig(Parameter.outputdir + '/fd_whole.pdf', dpi=300, bbox_inches='tight')
plt.show()

poptBcount_single_all = np.abs(np.sort(np.array(poptBcount_single_list)[:,0]))
xval = np.abs(np.linspace(min(poptBcount_single_all)-0.1, max(poptBcount_single_all)+0.1, 300))

kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.05).fit(poptBcount_single_all.reshape((len(poptBcount_single_all),1)))

log_dens = kde.score_samples(xval.reshape((len(xval),1)))

fig = plt.figure(figsize=(8,6))
plt.hist(poptBcount_single_all, bins=int(len(hlist_single_count)/5), density=True)
plt.plot(xval, np.exp(log_dens), lw=3)
plt.vlines(np.mean(poptBcount_single_all), 
           0, 
           7.5, 
           linestyle='--', 
           label="Mean: " + str(round(np.mean(poptBcount_single_all), 3)), 
           color='tab:red', lw=3)
plt.ylim(0, 7.5)
plt.legend(fontsize=13)
#plt.tight_layout()
plt.xlabel("Fractal Dimension", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
# plt.savefig(Parameter.outputdir + '/fd_single_whole_dist_1.pdf', dpi=300, bbox_inches='tight')
plt.show()



# t10 = time.time()

# print('checkpoint 10: ' + str(t10-t9))

#%% Single Neuron Dimension Calculation by calyx, LH, and AL using Binary Box-counting

MorphData.calyxdist_neuron = []
MorphData.LHdist_neuron = []
MorphData.ALdist_neuron = []

for i in range(len(MorphData.morph_dist)):
    calyxdist_neuron_t = []
    LHdist_neuron_t = []
    ALdist_neuron_t = []
    for j in range(len(MorphData.morph_dist[i])):
        
        branch_dist_temp2_rot = roty.apply(MorphData.morph_dist[i][j])
        branch_dist_temp2_rot2 = rotx.apply(MorphData.morph_dist[i][j])
        branch_dist_temp2_rot3 = rotz.apply(MorphData.morph_dist[i][j])
        
        # if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
        #     (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
        #     (np.array(branch_dist_temp2_rot)[2] > 354.95).all() and (np.array(branch_dist_temp2_rot)[2] < 399.06).all()):
        if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
            (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
            (np.array(branch_dist_temp2_rot3)[2] > 434.08).all() and (np.array(branch_dist_temp2_rot3)[2] < 496.22).all()):
            calyxdist_neuron_t.append(MorphData.morph_dist[i][j])
        # elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and
        #       (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 278.76).all() and
        #       (np.array(branch_dist_temp2_rot)[2] < 345.93).all()):
        elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and
              (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 286.78).all() and
              (np.array(branch_dist_temp2_rot)[2] < 343.93).all()):
            LHdist_neuron_t.append(MorphData.morph_dist[i][j])
        # elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 516.38).all() and 
        #       (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 361.12).all() and
        #       (np.array(branch_dist_temp2_rot2)[2] < -77.84).all()):
        elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 533.42).all() and 
              (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 363.12).all() and
              (np.array(branch_dist_temp2_rot2)[2] < 180.77).all()):
            ALdist_neuron_t.append(MorphData.morph_dist[i][j])
    if len(calyxdist_neuron_t) > 700:
        MorphData.calyxdist_neuron.append(calyxdist_neuron_t)
    if len(LHdist_neuron_t) > 700:
        MorphData.LHdist_neuron.append(LHdist_neuron_t)
    if len(ALdist_neuron_t) > 700:
        MorphData.ALdist_neuron.append(ALdist_neuron_t)


#%% Single Neuron Dimension Calculation by calyx

binsize = np.logspace(-2, 3, 100)[15:90:2]

hlist_single_count_calyx = np.empty((len(MorphData.calyxdist_neuron), len(binsize)))

for i in range(len(MorphData.calyxdist_neuron)):
    morph_dist_single = np.array(MorphData.calyxdist_neuron[i])
    xmax_single = np.max(morph_dist_single[:,0])
    xmin_single = np.min(morph_dist_single[:,0])
    ymax_single = np.max(morph_dist_single[:,1])
    ymin_single = np.min(morph_dist_single[:,1])
    zmax_single = np.max(morph_dist_single[:,2])
    zmin_single = np.min(morph_dist_single[:,2])
    
    for b in range(len(binsize)):
        xbin = np.arange(xmin_single-0.01, xmax_single+binsize[b], binsize[b])
        ybin = np.arange(ymin_single-0.01, ymax_single+binsize[b], binsize[b])
        zbin = np.arange(zmin_single-0.01, zmax_single+binsize[b], binsize[b])
        if len(xbin) < 2:
            xbin = [-1000, 1000]
        if len(ybin) < 2:
            ybin = [-1000, 1000]
        if len(zbin) < 2:
            zbin = [-1000, 1000]
            
        h, e = np.histogramdd(morph_dist_single, bins=[xbin, ybin, zbin])
        hlist_single_count_calyx[i][b] = np.count_nonzero(h)
   

#%%

cmap = cm.get_cmap('viridis', len(MorphData.calyxdist_neuron))

poptBcount_single_list_calyx = []
pcovBcount_single_list_calyx = []

fig = plt.figure(figsize=(12,8))
for i in range(len(MorphData.calyxdist_neuron)):
    farg = np.where(np.abs(np.diff(np.log10(hlist_single_count_calyx[i]))) > 0.15)[0][-1]#np.argwhere(hlist_single_count_calyx[i] > 1)[-1][0] + 2
    iarg = np.where(np.abs(np.diff(np.log10(hlist_single_count_calyx[i]))) > 0.15)[0][0]
    if iarg < 0:
        iarg = 0
    poptBcount_single_calyx, pcovBcount_single_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_single_count_calyx[i][iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
    perrBcount_single_calyx = np.sqrt(np.diag(pcovBcount_single_calyx))
    
    poptBcount_single_list_calyx.append(poptBcount_single_calyx)
    pcovBcount_single_list_calyx.append(perrBcount_single_calyx)
    
    fitYBcount_single_calyx = objFuncPpow(binsize, poptBcount_single_calyx[0], poptBcount_single_calyx[1])
    plt.scatter(binsize, hlist_single_count_calyx[i], color=cmap(i))
    plt.plot(binsize, fitYBcount_single_calyx, lw=2, linestyle='--', color=cmap(i))
    
#plt.legend([str(i) + ': ' + str(round(poptBcount_single[0], 3)) + '$\pm$' + str(round(perrBcount_single[0], 3))], fontsize=15)    
plt.yscale('log')
plt.xscale('log')
#plt.xlim(0.1, 1000)
#plt.ylim(0.1, 100000)
#plt.tight_layout()
plt.xlabel("Box Count", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()

poptBcount_single_all_calyx = np.abs(np.sort(np.array(poptBcount_single_list_calyx)[:,0]))
xval_calyx = np.linspace(min(poptBcount_single_all_calyx)-0.1, max(poptBcount_single_all_calyx)+0.1, 300)

kde_calyx = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.05).fit(poptBcount_single_all_calyx.reshape((len(poptBcount_single_all_calyx),1)))

log_dens_calyx = kde_calyx.score_samples(xval_calyx.reshape((len(xval_calyx),1)))

fig = plt.figure(figsize=(8,6))
plt.hist(poptBcount_single_all_calyx, bins=int(len(hlist_single_count_calyx)/5), density=True)
plt.plot(xval_calyx, np.exp(log_dens_calyx), lw=3)
plt.vlines(np.mean(poptBcount_single_all_calyx), 
           0, 
           5,
           linestyle='--',
           label="Mean: " + str(round(np.mean(poptBcount_single_all_calyx), 3)),
           color='tab:red',
           lw=3)
plt.ylim(0, 4.5)
plt.legend(fontsize=13)
plt.xlabel("Fractal Dimension", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
# plt.savefig(Parameter.outputdir + '/fd_single_calyx_dist_2.pdf', dpi=300, bbox_inches='tight')
plt.show()



#%% Single Neuron Dimension Calculation by LH

binsize = np.logspace(-2, 3, 100)[15:90:2]

hlist_single_count_LH = np.empty((len(MorphData.LHdist_neuron), len(binsize)))

for i in range(len(MorphData.LHdist_neuron)):
    morph_dist_single = np.array(MorphData.LHdist_neuron[i])
    xmax_single = np.max(morph_dist_single[:,0])
    xmin_single = np.min(morph_dist_single[:,0])
    ymax_single = np.max(morph_dist_single[:,1])
    ymin_single = np.min(morph_dist_single[:,1])
    zmax_single = np.max(morph_dist_single[:,2])
    zmin_single = np.min(morph_dist_single[:,2])
    
    for b in range(len(binsize)):
        xbin = np.arange(xmin_single-0.01, xmax_single+binsize[b], binsize[b])
        ybin = np.arange(ymin_single-0.01, ymax_single+binsize[b], binsize[b])
        zbin = np.arange(zmin_single-0.01, zmax_single+binsize[b], binsize[b])
        if len(xbin) < 2:
            xbin = [-1000, 1000]
        if len(ybin) < 2:
            ybin = [-1000, 1000]
        if len(zbin) < 2:
            zbin = [-1000, 1000]
            
        h, e = np.histogramdd(morph_dist_single, bins=[xbin, ybin, zbin])
        hlist_single_count_LH[i][b] = np.count_nonzero(h)
   


#%%

cmap = cm.get_cmap('viridis', len(MorphData.LHdist_neuron))

poptBcount_single_list_LH = []
pcovBcount_single_list_LH = []

fig = plt.figure(figsize=(12,8))
for i in range(len(MorphData.LHdist_neuron)):
    farg = np.where(np.abs(np.diff(np.log10(hlist_single_count_LH[i]))) > 0.15)[0][-1]#np.argwhere(hlist_single_count_LH[i] > 1)[-1][0] + 2
    iarg = np.where(np.abs(np.diff(np.log10(hlist_single_count_LH[i]))) > 0.15)[0][0]
    if iarg < 0:
        iarg = 0
    poptBcount_single_LH, pcovBcount_single_LH = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_single_count_LH[i][iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
    perrBcount_single_LH = np.sqrt(np.diag(pcovBcount_single_LH))
    
    poptBcount_single_list_LH.append(poptBcount_single_LH)
    pcovBcount_single_list_LH.append(perrBcount_single_LH)
    
    fitYBcount_single_LH = objFuncPpow(binsize, poptBcount_single_LH[0], poptBcount_single_LH[1])
    plt.scatter(binsize, hlist_single_count_LH[i], color=cmap(i))
    plt.plot(binsize, fitYBcount_single_LH, lw=2, linestyle='--', color=cmap(i))
    
#plt.legend([str(i) + ': ' + str(round(poptBcount_single[0], 3)) + '$\pm$' + str(round(perrBcount_single[0], 3))], fontsize=15)    
plt.yscale('log')
plt.xscale('log')
#plt.xlim(0.1, 1000)
#plt.ylim(0.1, 100000)
#plt.tight_layout()
plt.xlabel("Box Count", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()

poptBcount_single_all_LH = np.abs(np.sort(np.array(poptBcount_single_list_LH)[:,0]))
xval_LH = np.linspace(min(poptBcount_single_all_LH)-0.1, max(poptBcount_single_all_LH)+0.1, 300)

kde_LH = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.05).fit(poptBcount_single_all_LH.reshape((len(poptBcount_single_all_LH),1)))

log_dens_LH = kde_LH.score_samples(xval_LH.reshape((len(xval_LH),1)))

fig = plt.figure(figsize=(8,6))
plt.hist(poptBcount_single_all_LH, bins=int(len(hlist_single_count_LH)/5), density=True)
plt.plot(xval_LH, np.exp(log_dens_LH), lw=3)
plt.vlines(np.mean(poptBcount_single_all_LH), 
           0, 
           5, 
           linestyle='--', 
           label="Mean: " + str(round(np.mean(poptBcount_single_all_LH), 3)),
           color='tab:red', 
           lw=3)
plt.ylim(0, 4)
plt.legend(fontsize=13)
plt.xlabel("Fractal Dimension", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
# plt.savefig(Parameter.outputdir + '/fd_single_LH_dist_2.pdf', dpi=300, bbox_inches='tight')
plt.show()



#%% Single Neuron Dimension Calculation by AL

binsize = np.logspace(-2, 3, 100)[20:90:2]

hlist_single_count_AL = np.empty((len(MorphData.ALdist_neuron), len(binsize)))

for i in range(len(MorphData.ALdist_neuron)):
    morph_dist_single = np.array(MorphData.ALdist_neuron[i])
    xmax_single = np.max(morph_dist_single[:,0])
    xmin_single = np.min(morph_dist_single[:,0])
    ymax_single = np.max(morph_dist_single[:,1])
    ymin_single = np.min(morph_dist_single[:,1])
    zmax_single = np.max(morph_dist_single[:,2])
    zmin_single = np.min(morph_dist_single[:,2])
    
    for b in range(len(binsize)):
        xbin = np.arange(xmin_single-0.01, xmax_single+binsize[b], binsize[b])
        ybin = np.arange(ymin_single-0.01, ymax_single+binsize[b], binsize[b])
        zbin = np.arange(zmin_single-0.01, zmax_single+binsize[b], binsize[b])
        if len(xbin) < 2:
            xbin = [-1000, 1000]
        if len(ybin) < 2:
            ybin = [-1000, 1000]
        if len(zbin) < 2:
            zbin = [-1000, 1000]
            
        h, e = np.histogramdd(morph_dist_single, bins=[xbin, ybin, zbin])
        hlist_single_count_AL[i][b] = np.count_nonzero(h)
   


#%%

cmap = cm.get_cmap('viridis', len(MorphData.ALdist_neuron))

poptBcount_single_list_AL = []
pcovBcount_single_list_AL = []

fig = plt.figure(figsize=(12,8))
for i in range(len(MorphData.ALdist_neuron)):
    farg = np.where(np.abs(np.diff(np.log10(hlist_single_count_AL[i]))) > 0.15)[0][-1]#np.argwhere(hlist_single_count_AL[i] > 1)[-1][0] + 2
    iarg = np.where(np.abs(np.diff(np.log10(hlist_single_count_AL[i]))) > 0.15)[0][0]
    if iarg < 0:
        iarg = 0
    poptBcount_single_AL, pcovBcount_single_AL = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_single_count_AL[i][iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
    perrBcount_single_AL = np.sqrt(np.diag(pcovBcount_single_AL))
    
    poptBcount_single_list_AL.append(poptBcount_single_AL)
    pcovBcount_single_list_AL.append(perrBcount_single_AL)
    
    fitYBcount_single_AL = objFuncPpow(binsize, poptBcount_single_AL[0], poptBcount_single_AL[1])
    plt.scatter(binsize, hlist_single_count_AL[i], color=cmap(i))
    plt.plot(binsize, fitYBcount_single_AL, lw=2, linestyle='--', color=cmap(i))
    
#plt.legend([str(i) + ': ' + str(round(poptBcount_single[0], 3)) + '$\pm$' + str(round(perrBcount_single[0], 3))], fontsize=15)    
plt.yscale('log')
plt.xscale('log')
#plt.xlim(0.1, 1000)
#plt.ylim(0.1, 100000)
#plt.tight_layout()
plt.xlabel("Box Count", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()

poptBcount_single_all_AL = np.abs(np.sort(np.array(poptBcount_single_list_AL)[:,0]))
xval_AL = np.linspace(min(poptBcount_single_all_AL)-0.1, max(poptBcount_single_all_AL)+0.1, 300)

kde_AL = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.05).fit(poptBcount_single_all_AL.reshape((len(poptBcount_single_all_AL),1)))

log_dens_AL = kde_AL.score_samples(xval_AL.reshape((len(xval_AL),1)))

fig = plt.figure(figsize=(8,6))
plt.hist(poptBcount_single_all_AL, bins=int(len(hlist_single_count_AL)/5), density=True)
plt.plot(xval_AL, np.exp(log_dens_AL), lw=3)
plt.vlines(np.mean(poptBcount_single_all_AL), 
           0,
           5,
           linestyle='--', 
           label="Mean: " + str(round(np.mean(poptBcount_single_all_AL), 3)), 
           color='tab:red',
           lw=3)
plt.ylim(0, 4.5)
plt.legend(fontsize=13)
plt.xlabel("Fractal Dimension", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
# plt.savefig(Parameter.outputdir + '/fd_single_AL_dist_2.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Branching point and tip coordinate collection

BranchData.branchP_dist = []
MorphData.endP_dist = []
        
for i in range(len(BranchData.branchP)):
    branchP_dist_t = []
    for j in range(len(BranchData.branchP[i])):
        branchP_dist_t.append(MorphData.morph_dist[i][MorphData.morph_id[i].index(BranchData.branchP[i][j])])
    if len(branchP_dist_t) > 0:
        BranchData.branchP_dist.append(branchP_dist_t)
    else:
        BranchData.branchP_dist.append([])
    
for i in range(len(MorphData.endP)):
    endP_dist_t = []
    for j in range(len(MorphData.endP[i])):
        endP_dist_t.append(MorphData.morph_dist[i][MorphData.morph_id[i].index(MorphData.endP[i][j])])
    if len(endP_dist_t) > 0:
        MorphData.endP_dist.append(endP_dist_t)
    else:
        MorphData.endP_dist.append([])

branchP_dist_flat = np.array([item for sublist in BranchData.branchP_dist for item in sublist])
endP_dist_flat = np.array([item for sublist in MorphData.endP_dist for item in sublist])



#%%

fig = plt.figure(figsize=(24, 16))
ax = plt.axes(projection='3d')
cmap = cm.get_cmap('viridis', len(BranchData.branchP_dist))
for f in range(len(BranchData.branchP_dist)):
    ax.scatter3D(np.array(BranchData.branchP_dist[f])[:,0], 
                 np.array(BranchData.branchP_dist[f])[:,1], 
                 np.array(BranchData.branchP_dist[f])[:,2], color=cmap(f), marker='o')
plt.show()


fig = plt.figure(figsize=(24, 16))
ax = plt.axes(projection='3d')
cmap = cm.get_cmap('viridis', len(MorphData.endP_dist))
for f in range(len(MorphData.endP_dist)):
    ax.scatter3D(np.array(MorphData.endP_dist[f])[:,0], 
                 np.array(MorphData.endP_dist[f])[:,1], 
                 np.array(MorphData.endP_dist[f])[:,2], color=cmap(f), marker='o')
plt.show()



#%% Binary Box-counting for all branching points


binsize = np.logspace(-1, 3, 100)[13:99:3]

xmax_bp = np.max(branchP_dist_flat[:,0])
xmin_bp = np.min(branchP_dist_flat[:,0])
ymax_bp = np.max(branchP_dist_flat[:,1])
ymin_bp = np.min(branchP_dist_flat[:,1])
zmax_bp = np.max(branchP_dist_flat[:,2])
zmin_bp = np.min(branchP_dist_flat[:,2])

hlist_bp_count = []

for b in range(len(binsize)):
    xbin = np.arange(xmin_bp, xmax_bp+binsize[b], binsize[b])
    ybin = np.arange(ymin_bp, ymax_bp+binsize[b], binsize[b])
    zbin = np.arange(zmin_bp, zmax_bp+binsize[b], binsize[b])
    if len(xbin) == 1:
        xbin = [-1000, 1000]
    if len(ybin) == 1:
        ybin = [-1000, 1000]
    if len(zbin) == 1:
        zbin = [-1000, 1000]
        
    h, e = np.histogramdd(branchP_dist_flat, 
                          bins=[xbin, 
                                ybin,
                                zbin])
    hlist_bp_count.append(np.count_nonzero(h))


#%%

poptBcount_bp_all, pcovBcount_bp_all = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[8:21]), 
                                                        np.log10(hlist_bp_count[8:21]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_bp_all = np.sqrt(np.diag(pcovBcount_bp_all))

fitYBcount_bp_all = objFuncPpow(binsize, poptBcount_bp_all[0], poptBcount_bp_all[1])
    
fig = plt.figure(figsize=(12,8))
plt.scatter(binsize, hlist_bp_count)
plt.plot(binsize, fitYBcount_bp_all, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['All: ' + str(round(poptBcount_bp_all[0], 3)) + '$\pm$' + str(round(perrBcount_bp_all[0], 3))], fontsize=15)
#plt.xlim(0.1, 20)
#plt.tight_layout()
plt.xlabel("Box Size", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()


#%% Binary Box-counting for all tips


binsize = np.logspace(-1, 3, 100)[13:99:3]

xmax_endP = np.max(endP_dist_flat[:,0])
xmin_endP = np.min(endP_dist_flat[:,0])
ymax_endP = np.max(endP_dist_flat[:,1])
ymin_endP = np.min(endP_dist_flat[:,1])
zmax_endP = np.max(endP_dist_flat[:,2])
zmin_endP = np.min(endP_dist_flat[:,2])

hlist_endP_count = []

for b in range(len(binsize)):
    xbin = np.arange(xmin_endP, xmax_endP+binsize[b], binsize[b])
    ybin = np.arange(ymin_endP, ymax_endP+binsize[b], binsize[b])
    zbin = np.arange(zmin_endP, zmax_endP+binsize[b], binsize[b])
    if len(xbin) == 1:
        xbin = [-1000, 1000]
    if len(ybin) == 1:
        ybin = [-1000, 1000]
    if len(zbin) == 1:
        zbin = [-1000, 1000]
        
    h, e = np.histogramdd(endP_dist_flat, 
                          bins=[xbin, 
                                ybin,
                                zbin])
    hlist_endP_count.append(np.count_nonzero(h))


#%%

poptBcount_endP_all, pcovBcount_endP_all = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[8:21]), 
                                                        np.log10(hlist_endP_count[8:21]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_endP_all = np.sqrt(np.diag(pcovBcount_endP_all))

fitYBcount_endP_all = objFuncPpow(binsize, poptBcount_endP_all[0], poptBcount_endP_all[1])
    
fig = plt.figure(figsize=(12,8))
plt.scatter(binsize, hlist_endP_count)
plt.plot(binsize, fitYBcount_endP_all, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['All: ' + str(round(poptBcount_endP_all[0], 3)) + '$\pm$' + str(round(perrBcount_endP_all[0], 3))], fontsize=15)
#plt.xlim(0.1, 20)
#plt.tight_layout()
plt.xlabel("Box Size", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()


#%% Categorization for branching points and tips

branchP_calyx_dist = []
branchP_LH_dist = []
branchP_AL_dist = []

for i in range(len(branchP_dist_flat)):
    
    branch_dist_temp2_rot = roty.apply(branchP_dist_flat[i])
    branch_dist_temp2_rot2 = rotx.apply(branchP_dist_flat[i])
    branch_dist_temp2_rot3 = rotz.apply(branchP_dist_flat[i])
    
    # if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
    #     (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
    #     (np.array(branch_dist_temp2_rot)[2] > 354.95).all() and (np.array(branch_dist_temp2_rot)[2] < 399.06).all()):
    if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
        (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
        (np.array(branch_dist_temp2_rot3)[2] > 434.08).all() and (np.array(branch_dist_temp2_rot3)[2] < 496.22).all()):
        branchP_calyx_dist.append(branchP_dist_flat[i])
    # elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and
    #       (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 278.76).all() and
    #       (np.array(branch_dist_temp2_rot)[2] < 345.93).all()):
    elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and
          (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 286.78).all() and
          (np.array(branch_dist_temp2_rot)[2] < 343.93).all()):
        branchP_LH_dist.append(branchP_dist_flat[i])
    # elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 516.38).all() and 
    #       (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 361.12).all() and
    #       (np.array(branch_dist_temp2_rot2)[2] < -77.84).all()):
    elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 533.42).all() and 
          (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 363.12).all() and
          (np.array(branch_dist_temp2_rot2)[2] < 180.77).all()):
        branchP_AL_dist.append(branchP_dist_flat[i])


endP_calyx_dist = []
endP_LH_dist = []
endP_AL_dist = []

for i in range(len(endP_dist_flat)):
    
    branch_dist_temp2_rot = roty.apply(endP_dist_flat[i])
    branch_dist_temp2_rot2 = rotx.apply(endP_dist_flat[i])
    branch_dist_temp2_rot3 = rotz.apply(endP_dist_flat[i])
    
    # if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
    #     (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
    #     (np.array(branch_dist_temp2_rot)[2] > 354.95).all() and (np.array(branch_dist_temp2_rot)[2] < 399.06).all()):
    if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
        (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
        (np.array(branch_dist_temp2_rot3)[2] > 434.08).all() and (np.array(branch_dist_temp2_rot3)[2] < 496.22).all()):
        endP_calyx_dist.append(endP_dist_flat[i])
    # elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and
    #       (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 278.76).all() and
    #       (np.array(branch_dist_temp2_rot)[2] < 345.93).all()):
    elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and
          (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 286.78).all() and
          (np.array(branch_dist_temp2_rot)[2] < 343.93).all()):
        endP_LH_dist.append(endP_dist_flat[i])
    # elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 516.38).all() and 
    #       (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 361.12).all() and
    #       (np.array(branch_dist_temp2_rot2)[2] < -77.84).all()):
    elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 533.42).all() and 
          (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 363.12).all() and
          (np.array(branch_dist_temp2_rot2)[2] < 180.77).all()):
        endP_AL_dist.append(endP_dist_flat[i])



#%% Branching point region fractal dimension calculation

binsize = np.logspace(-1, 3, 100)[13:90:3]

branchP_calyx_dist = np.array(branchP_calyx_dist)
branchP_LH_dist = np.array(branchP_LH_dist)
branchP_AL_dist = np.array(branchP_AL_dist)

xmax_bp_calyx = np.max(branchP_calyx_dist[:,0])
xmin_bp_calyx = np.min(branchP_calyx_dist[:,0])
ymax_bp_calyx = np.max(branchP_calyx_dist[:,1])
ymin_bp_calyx = np.min(branchP_calyx_dist[:,1])
zmax_bp_calyx = np.max(branchP_calyx_dist[:,2])
zmin_bp_calyx = np.min(branchP_calyx_dist[:,2])

xmax_bp_LH = np.max(branchP_LH_dist[:,0])
xmin_bp_LH = np.min(branchP_LH_dist[:,0])
ymax_bp_LH = np.max(branchP_LH_dist[:,1])
ymin_bp_LH = np.min(branchP_LH_dist[:,1])
zmax_bp_LH = np.max(branchP_LH_dist[:,2])
zmin_bp_LH = np.min(branchP_LH_dist[:,2])

xmax_bp_AL = np.max(branchP_AL_dist[:,0])
xmin_bp_AL = np.min(branchP_AL_dist[:,0])
ymax_bp_AL = np.max(branchP_AL_dist[:,1])
ymin_bp_AL = np.min(branchP_AL_dist[:,1])
zmax_bp_AL = np.max(branchP_AL_dist[:,2])
zmin_bp_AL = np.min(branchP_AL_dist[:,2])

hlist_calyx_bp = []
hlist_calyx_bp_count = []
hlist_calyx_bp_numbox = []
hlist_LH_bp = []
hlist_LH_bp_count = []
hlist_LH_bp_numbox = []
hlist_AL_bp = []
hlist_AL_bp_count = []
hlist_AL_bp_numbox = []

for b in range(len(binsize)):
    xbin_calyx = np.arange(xmin_bp_calyx, xmax_bp_calyx+binsize[b], binsize[b])
    ybin_calyx = np.arange(ymin_bp_calyx, ymax_bp_calyx+binsize[b], binsize[b])
    zbin_calyx = np.arange(zmin_bp_calyx, zmax_bp_calyx+binsize[b], binsize[b])
    if len(xbin_calyx) == 1:
        xbin_calyx = [-1000, 1000]
    if len(ybin_calyx) == 1:
        ybin_calyx = [-1000, 1000]
    if len(zbin_calyx) == 1:
        zbin_calyx = [-1000, 1000]
    
    hc, e = np.histogramdd(branchP_calyx_dist, 
                          bins=[xbin_calyx, 
                                ybin_calyx,
                                zbin_calyx])
    hlist_calyx_bp_count.append(np.count_nonzero(hc))
    
    xbin_LH = np.arange(xmin_bp_LH, xmax_bp_LH+binsize[b], binsize[b])
    ybin_LH = np.arange(ymin_bp_LH, ymax_bp_LH+binsize[b], binsize[b])
    zbin_LH = np.arange(zmin_bp_LH, zmax_bp_LH+binsize[b], binsize[b])
    if len(xbin_LH) == 1:
        xbin_LH = [-1000, 1000]
    if len(ybin_LH) == 1:
        ybin_LH = [-1000, 1000]
    if len(zbin_LH) == 1:
        zbin_LH = [-1000, 1000]
    
    hh, e = np.histogramdd(branchP_LH_dist, 
                          bins=[xbin_LH, 
                                ybin_LH,
                                zbin_LH])
    hlist_LH_bp_count.append(np.count_nonzero(hh))
    
    xbin_AL = np.arange(xmin_bp_AL, xmax_bp_AL+binsize[b], binsize[b])
    ybin_AL = np.arange(ymin_bp_AL, ymax_bp_AL+binsize[b], binsize[b])
    zbin_AL = np.arange(zmin_bp_AL, zmax_bp_AL+binsize[b], binsize[b])
    if len(xbin_AL) == 1:
        xbin_AL = [-1000, 1000]
    if len(ybin_AL) == 1:
        ybin_AL = [-1000, 1000]
    if len(zbin_AL) == 1:
        zbin_AL = [-1000, 1000]
        
    ha, e = np.histogramdd(branchP_AL_dist, 
                          bins=[xbin_AL, 
                                ybin_AL,
                                zbin_AL])
    hlist_AL_bp_count.append(np.count_nonzero(ha))




#%%
    
    
poptBcount_bp_calyx, pcovBcount_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_calyx_bp_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_bp_calyx = np.sqrt(np.diag(pcovBcount_calyx))

poptBcount_bp_LH, pcovBcount_LH = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_LH_bp_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_bp_LH = np.sqrt(np.diag(pcovBcount_LH))

poptBcount_bp_AL, pcovBcount_AL = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_AL_bp_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_bp_AL = np.sqrt(np.diag(pcovBcount_AL))

fitYBcount_bp_calyx = objFuncPpow(binsize, poptBcount_bp_calyx[0], poptBcount_bp_calyx[1])
fitYBcount_bp_LH = objFuncPpow(binsize, poptBcount_bp_LH[0], poptBcount_bp_LH[1])
fitYBcount_bp_AL = objFuncPpow(binsize, poptBcount_bp_AL[0], poptBcount_bp_AL[1])
    
fig = plt.figure(figsize=(12,8))
plt.scatter(binsize, hlist_calyx_bp_count)
plt.scatter(binsize, hlist_LH_bp_count)
plt.scatter(binsize, hlist_AL_bp_count)
plt.plot(binsize, fitYBcount_bp_calyx, lw=2, linestyle='--')
plt.plot(binsize, fitYBcount_bp_LH, lw=2, linestyle='--')
plt.plot(binsize, fitYBcount_bp_AL, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['Calyx: ' + str(round(poptBcount_bp_calyx[0], 3)) + '$\pm$' + str(round(perrBcount_bp_calyx[0], 3)),
            'LH: ' + str(round(poptBcount_bp_LH[0], 3)) + '$\pm$' + str(round(perrBcount_bp_LH[0], 3)),
            'AL: ' + str(round(poptBcount_bp_AL[0], 3)) + '$\pm$' + str(round(perrBcount_bp_AL[0], 3))], fontsize=15)
#plt.xlim(0.1, 20)
#plt.tight_layout()
plt.xlabel("Box Size", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()





#%% Tip region fractal dimension calculation

binsize = np.logspace(-1, 3, 100)[13:90:3]

endP_calyx_dist = np.array(endP_calyx_dist)
endP_LH_dist = np.array(endP_LH_dist)
endP_AL_dist = np.array(endP_AL_dist)

xmax_ep_calyx = np.max(endP_calyx_dist[:,0])
xmin_ep_calyx = np.min(endP_calyx_dist[:,0])
ymax_ep_calyx = np.max(endP_calyx_dist[:,1])
ymin_ep_calyx = np.min(endP_calyx_dist[:,1])
zmax_ep_calyx = np.max(endP_calyx_dist[:,2])
zmin_ep_calyx = np.min(endP_calyx_dist[:,2])

xmax_ep_LH = np.max(endP_LH_dist[:,0])
xmin_ep_LH = np.min(endP_LH_dist[:,0])
ymax_ep_LH = np.max(endP_LH_dist[:,1])
ymin_ep_LH = np.min(endP_LH_dist[:,1])
zmax_ep_LH = np.max(endP_LH_dist[:,2])
zmin_ep_LH = np.min(endP_LH_dist[:,2])

xmax_ep_AL = np.max(endP_AL_dist[:,0])
xmin_ep_AL = np.min(endP_AL_dist[:,0])
ymax_ep_AL = np.max(endP_AL_dist[:,1])
ymin_ep_AL = np.min(endP_AL_dist[:,1])
zmax_ep_AL = np.max(endP_AL_dist[:,2])
zmin_ep_AL = np.min(endP_AL_dist[:,2])

hlist_calyx_ep = []
hlist_calyx_ep_count = []
hlist_calyx_ep_numbox = []
hlist_LH_ep = []
hlist_LH_ep_count = []
hlist_LH_ep_numbox = []
hlist_AL_ep = []
hlist_AL_ep_count = []
hlist_AL_ep_numbox = []

for b in range(len(binsize)):
    xbin_calyx = np.arange(xmin_ep_calyx, xmax_ep_calyx+binsize[b], binsize[b])
    ybin_calyx = np.arange(ymin_ep_calyx, ymax_ep_calyx+binsize[b], binsize[b])
    zbin_calyx = np.arange(zmin_ep_calyx, zmax_ep_calyx+binsize[b], binsize[b])
    if len(xbin_calyx) == 1:
        xbin_calyx = [-1000, 1000]
    if len(ybin_calyx) == 1:
        ybin_calyx = [-1000, 1000]
    if len(zbin_calyx) == 1:
        zbin_calyx = [-1000, 1000]
    
    hc, e = np.histogramdd(endP_calyx_dist, 
                          bins=[xbin_calyx, 
                                ybin_calyx,
                                zbin_calyx])
    hlist_calyx_ep_count.append(np.count_nonzero(hc))
    
    xbin_LH = np.arange(xmin_ep_LH, xmax_ep_LH+binsize[b], binsize[b])
    ybin_LH = np.arange(ymin_ep_LH, ymax_ep_LH+binsize[b], binsize[b])
    zbin_LH = np.arange(zmin_ep_LH, zmax_ep_LH+binsize[b], binsize[b])
    if len(xbin_LH) == 1:
        xbin_LH = [-1000, 1000]
    if len(ybin_LH) == 1:
        ybin_LH = [-1000, 1000]
    if len(zbin_LH) == 1:
        zbin_LH = [-1000, 1000]
    
    hh, e = np.histogramdd(endP_LH_dist, 
                          bins=[xbin_LH, 
                                ybin_LH,
                                zbin_LH])
    hlist_LH_ep_count.append(np.count_nonzero(hh))
    
    xbin_AL = np.arange(xmin_ep_AL, xmax_ep_AL+binsize[b], binsize[b])
    ybin_AL = np.arange(ymin_ep_AL, ymax_ep_AL+binsize[b], binsize[b])
    zbin_AL = np.arange(zmin_ep_AL, zmax_ep_AL+binsize[b], binsize[b])
    if len(xbin_AL) == 1:
        xbin_AL = [-1000, 1000]
    if len(ybin_AL) == 1:
        ybin_AL = [-1000, 1000]
    if len(zbin_AL) == 1:
        zbin_AL = [-1000, 1000]
        
    ha, e = np.histogramdd(endP_AL_dist, 
                          bins=[xbin_AL, 
                                ybin_AL,
                                zbin_AL])
    hlist_AL_ep_count.append(np.count_nonzero(ha))




#%%
    
    
poptBcount_ep_calyx, pcovBcount_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_calyx_ep_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_ep_calyx = np.sqrt(np.diag(pcovBcount_calyx))

poptBcount_ep_LH, pcovBcount_LH = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_LH_ep_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_ep_LH = np.sqrt(np.diag(pcovBcount_LH))

poptBcount_ep_AL, pcovBcount_AL = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_AL_ep_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_ep_AL = np.sqrt(np.diag(pcovBcount_AL))

fitYBcount_ep_calyx = objFuncPpow(binsize, poptBcount_ep_calyx[0], poptBcount_ep_calyx[1])
fitYBcount_ep_LH = objFuncPpow(binsize, poptBcount_ep_LH[0], poptBcount_ep_LH[1])
fitYBcount_ep_AL = objFuncPpow(binsize, poptBcount_ep_AL[0], poptBcount_ep_AL[1])
    
fig = plt.figure(figsize=(12,8))
plt.scatter(binsize, hlist_calyx_ep_count)
plt.scatter(binsize, hlist_LH_ep_count)
plt.scatter(binsize, hlist_AL_ep_count)
plt.plot(binsize, fitYBcount_ep_calyx, lw=2, linestyle='--')
plt.plot(binsize, fitYBcount_ep_LH, lw=2, linestyle='--')
plt.plot(binsize, fitYBcount_ep_AL, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['Calyx: ' + str(round(poptBcount_ep_calyx[0], 3)) + '$\pm$' + str(round(perrBcount_ep_calyx[0], 3)),
            'LH: ' + str(round(poptBcount_ep_LH[0], 3)) + '$\pm$' + str(round(perrBcount_ep_LH[0], 3)),
            'AL: ' + str(round(poptBcount_ep_AL[0], 3)) + '$\pm$' + str(round(perrBcount_ep_AL[0], 3))], fontsize=15)
#plt.xlim(0.1, 20)
#plt.tight_layout()
plt.xlabel("Box Size", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()



#%% Branching point and tip combined region fractal dimension calculation

binsize = np.logspace(-1, 3, 100)[13:90:3]

baeP_calyx_dist = np.array(branchP_calyx_dist + endP_calyx_dist)
baeP_LH_dist = np.array(branchP_LH_dist + endP_LH_dist)
baeP_AL_dist = np.array(branchP_AL_dist + endP_AL_dist)

xmax_baep_calyx = np.max(baeP_calyx_dist[:,0])
xmin_baep_calyx = np.min(baeP_calyx_dist[:,0])
ymax_baep_calyx = np.max(baeP_calyx_dist[:,1])
ymin_baep_calyx = np.min(baeP_calyx_dist[:,1])
zmax_baep_calyx = np.max(baeP_calyx_dist[:,2])
zmin_baep_calyx = np.min(baeP_calyx_dist[:,2])

xmax_baep_LH = np.max(baeP_LH_dist[:,0])
xmin_baep_LH = np.min(baeP_LH_dist[:,0])
ymax_baep_LH = np.max(baeP_LH_dist[:,1])
ymin_baep_LH = np.min(baeP_LH_dist[:,1])
zmax_baep_LH = np.max(baeP_LH_dist[:,2])
zmin_baep_LH = np.min(baeP_LH_dist[:,2])

xmax_baep_AL = np.max(baeP_AL_dist[:,0])
xmin_baep_AL = np.min(baeP_AL_dist[:,0])
ymax_baep_AL = np.max(baeP_AL_dist[:,1])
ymin_baep_AL = np.min(baeP_AL_dist[:,1])
zmax_baep_AL = np.max(baeP_AL_dist[:,2])
zmin_baep_AL = np.min(baeP_AL_dist[:,2])

hlist_calyx_baep = []
hlist_calyx_baep_count = []
hlist_calyx_baep_numbox = []
hlist_LH_baep = []
hlist_LH_baep_count = []
hlist_LH_baep_numbox = []
hlist_AL_baep = []
hlist_AL_baep_count = []
hlist_AL_baep_numbox = []

for b in range(len(binsize)):
    xbin_calyx = np.arange(xmin_baep_calyx, xmax_baep_calyx+binsize[b], binsize[b])
    ybin_calyx = np.arange(ymin_baep_calyx, ymax_baep_calyx+binsize[b], binsize[b])
    zbin_calyx = np.arange(zmin_baep_calyx, zmax_baep_calyx+binsize[b], binsize[b])
    if len(xbin_calyx) == 1:
        xbin_calyx = [-1000, 1000]
    if len(ybin_calyx) == 1:
        ybin_calyx = [-1000, 1000]
    if len(zbin_calyx) == 1:
        zbin_calyx = [-1000, 1000]
    
    hc, e = np.histogramdd(baeP_calyx_dist, 
                          bins=[xbin_calyx, 
                                ybin_calyx,
                                zbin_calyx])
    hlist_calyx_baep_count.append(np.count_nonzero(hc))
    
    xbin_LH = np.arange(xmin_baep_LH, xmax_baep_LH+binsize[b], binsize[b])
    ybin_LH = np.arange(ymin_baep_LH, ymax_baep_LH+binsize[b], binsize[b])
    zbin_LH = np.arange(zmin_baep_LH, zmax_baep_LH+binsize[b], binsize[b])
    if len(xbin_LH) == 1:
        xbin_LH = [-1000, 1000]
    if len(ybin_LH) == 1:
        ybin_LH = [-1000, 1000]
    if len(zbin_LH) == 1:
        zbin_LH = [-1000, 1000]
    
    hh, e = np.histogramdd(baeP_LH_dist, 
                          bins=[xbin_LH, 
                                ybin_LH,
                                zbin_LH])
    hlist_LH_baep_count.append(np.count_nonzero(hh))
    
    xbin_AL = np.arange(xmin_baep_AL, xmax_baep_AL+binsize[b], binsize[b])
    ybin_AL = np.arange(ymin_baep_AL, ymax_baep_AL+binsize[b], binsize[b])
    zbin_AL = np.arange(zmin_baep_AL, zmax_baep_AL+binsize[b], binsize[b])
    if len(xbin_AL) == 1:
        xbin_AL = [-1000, 1000]
    if len(ybin_AL) == 1:
        ybin_AL = [-1000, 1000]
    if len(zbin_AL) == 1:
        zbin_AL = [-1000, 1000]
        
    ha, e = np.histogramdd(baeP_AL_dist, 
                          bins=[xbin_AL, 
                                ybin_AL,
                                zbin_AL])
    hlist_AL_baep_count.append(np.count_nonzero(ha))




#%%
    
    
poptBcount_baep_calyx, pcovBcount_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_calyx_baep_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_baep_calyx = np.sqrt(np.diag(pcovBcount_calyx))

poptBcount_baep_LH, pcovBcount_LH = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_LH_baep_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_baep_LH = np.sqrt(np.diag(pcovBcount_LH))

poptBcount_baep_AL, pcovBcount_AL = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_AL_baep_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_baep_AL = np.sqrt(np.diag(pcovBcount_AL))

fitYBcount_baep_calyx = objFuncPpow(binsize, poptBcount_baep_calyx[0], poptBcount_baep_calyx[1])
fitYBcount_baep_LH = objFuncPpow(binsize, poptBcount_baep_LH[0], poptBcount_baep_LH[1])
fitYBcount_baep_AL = objFuncPpow(binsize, poptBcount_baep_AL[0], poptBcount_baep_AL[1])
    
fig = plt.figure(figsize=(12,8))
plt.scatter(binsize, hlist_calyx_baep_count)
plt.scatter(binsize, hlist_LH_baep_count)
plt.scatter(binsize, hlist_AL_baep_count)
plt.plot(binsize, fitYBcount_baep_calyx, lw=2, linestyle='--')
plt.plot(binsize, fitYBcount_baep_LH, lw=2, linestyle='--')
plt.plot(binsize, fitYBcount_baep_AL, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['Calyx: ' + str(round(poptBcount_baep_calyx[0], 3)) + '$\pm$' + str(round(perrBcount_baep_calyx[0], 3)),
            'LH: ' + str(round(poptBcount_baep_LH[0], 3)) + '$\pm$' + str(round(perrBcount_baep_LH[0], 3)),
            'AL: ' + str(round(poptBcount_baep_AL[0], 3)) + '$\pm$' + str(round(perrBcount_baep_AL[0], 3))], fontsize=15)
#plt.xlim(0.1, 20)
#plt.tight_layout()
plt.xlabel("Box Size", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()


t11 = time.time()



#%% Radius of Gyration for minmax contour

poptR, pcovR = scipy.optimize.curve_fit(objFuncGL, 
                                        np.log10(LengthData.length_total), 
                                        np.log10(rGy), 
                                        p0=[1., 0.], 
                                        maxfev=100000)
perrR = np.sqrt(np.diag(pcovR))
fitYR = objFuncPpow(LengthData.length_total, poptR[0], poptR[1])

fig = plt.figure(figsize=(8,6))
plt.scatter(LengthData.length_total, rGy)
plt.plot(LengthData.length_total, fitYR, color='tab:red')
plt.yscale('log')
plt.xscale('log')
#    plt.xlim(10, 10000)
#    plt.ylim(7, 4000)
plt.legend([str(round(poptR[0], 3)) + '$\pm$' + str(round(perrR[0], 3))], fontsize=15)
plt.title(r"Scaling Behavior of $R_{g}$ to Length", fontsize=20)
plt.xlabel(r"Length ($\lambda N$)", fontsize=15)
plt.ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
plt.tight_layout()
plt.show()


#%% Radius of Gyration for minmax coordinate (end-to-end)

LengthData.length_ee = []

for i in range(len(MorphData.morph_dist)):
    LengthData.length_ee.append(np.max(scipy.spatial.distance.cdist(MorphData.morph_dist[i], MorphData.morph_dist[i])))
    
poptR_ee, pcovR_ee = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(LengthData.length_ee), 
                                              np.log10(rGy), 
                                              p0=[1., 0.], 
                                              maxfev=100000)
perrR_ee = np.sqrt(np.diag(pcovR_ee))
fitYR_ee = objFuncPpow(LengthData.length_ee, poptR_ee[0], poptR_ee[1])

fig = plt.figure(figsize=(8,6))
plt.scatter(LengthData.length_ee, rGy)
plt.plot(LengthData.length_ee, fitYR_ee, color='tab:red')
plt.yscale('log')
plt.xscale('log')
#    plt.xlim(10, 10000)
#    plt.ylim(7, 4000)
plt.legend([str(round(poptR_ee[0], 3)) + '$\pm$' + str(round(perrR_ee[0], 3))], fontsize=15)
plt.title(r"Scaling Behavior of $R_{g}$ to Maximal Length", fontsize=20)
plt.xlabel(r"Length ($\lambda N$)", fontsize=15)
plt.ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
plt.tight_layout()
plt.show()



#%% Radius of Gyration for calyx, LH, and AL

LengthData.length_calyx_total = []
LengthData.length_LH_total = []
LengthData.length_AL_total = []

calyxdist_per_n_flat = []
LHdist_per_n_flat = []
ALdist_per_n_flat = []

calyxdist_per_n_count = []
LHdist_per_n_count = []
ALdist_per_n_count = []

un_calyx = np.unique(MorphData.calyxdist_trk)
un_LH = np.unique(MorphData.LHdist_trk)
un_AL = np.unique(MorphData.ALdist_trk)

for i in range(len(un_calyx)):
    idx = np.where(MorphData.calyxdist_trk == un_calyx[i])[0]
    tarval = np.array(MorphData.calyxdist,dtype=object)[idx]
    calyxdist_per_n_flat_t = [item for sublist in tarval for item in sublist]
    sumval = np.sum(LengthData.length_calyx[un_calyx[i]])
    if calyxdist_per_n_flat_t:# and sumval > 1:# and sumval < 5000:
        calyxdist_per_n_flat.append(calyxdist_per_n_flat_t)
        calyxdist_per_n_count.append(len(calyxdist_per_n_flat_t))
        LengthData.length_calyx_total.append(sumval)

(rGy_calyx, cML_calyx) = utils.radiusOfGyration(calyxdist_per_n_flat)

for i in range(len(un_LH)):
    idx = np.where(MorphData.LHdist_trk == un_LH[i])[0]
    tarval = np.array(MorphData.LHdist,dtype=object)[idx]
    LHdist_per_n_flat_t = [item for sublist in tarval for item in sublist]
    sumval = np.sum(LengthData.length_LH[un_LH[i]])
    if LHdist_per_n_flat_t:# and sumval > 1:# and sumval < 5000:
        LHdist_per_n_flat.append(LHdist_per_n_flat_t)
        LHdist_per_n_count.append(len(LHdist_per_n_flat_t))
        LengthData.length_LH_total.append(sumval)

(rGy_LH, cML_LH) = utils.radiusOfGyration(LHdist_per_n_flat)

for i in range(len(un_AL)):
    idx = np.where(MorphData.ALdist_trk == un_AL[i])[0]
    tarval = np.array(MorphData.ALdist,dtype=object)[idx]
    ALdist_per_n_flat_t = [item for sublist in tarval for item in sublist]
    sumval = np.sum(LengthData.length_AL[un_AL[i]])
    if ALdist_per_n_flat_t:# and sumval > 1:# and sumval < 5000:
        ALdist_per_n_flat.append(ALdist_per_n_flat_t)
        ALdist_per_n_count.append(len(ALdist_per_n_flat_t))
        LengthData.length_AL_total.append(sumval)

(rGy_AL, cML_AL) = utils.radiusOfGyration(ALdist_per_n_flat)

#%%

xvallog1 = np.logspace(0, 4)

poptR_calyx, pcovR_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                        np.log10(LengthData.length_calyx_total)[np.argsort(np.log10(LengthData.length_calyx_total))[4:]], 
                                        np.log10(rGy_calyx)[np.argsort(np.log10(LengthData.length_calyx_total))[4:]], 
                                        p0=[1., 0.], 
                                        maxfev=100000)
perrR_calyx = np.sqrt(np.diag(pcovR_calyx))
fitYR_calyx = objFuncPpow(xvallog1, poptR_calyx[0], poptR_calyx[1])

poptR_LH, pcovR_LH = scipy.optimize.curve_fit(objFuncGL, 
                                        np.log10(LengthData.length_LH_total), 
                                        np.log10(rGy_LH), 
                                        p0=[1., 0.], 
                                        maxfev=100000)
perrR_LH = np.sqrt(np.diag(pcovR_LH))
fitYR_LH = objFuncPpow(xvallog1, poptR_LH[0], poptR_LH[1])

poptR_AL, pcovR_AL = scipy.optimize.curve_fit(objFuncGL, 
                                        np.log10(LengthData.length_AL_total), 
                                        np.log10(rGy_AL), 
                                        p0=[1., 0.], 
                                        maxfev=100000)
perrR_AL = np.sqrt(np.diag(pcovR_AL))
fitYR_AL = objFuncPpow(xvallog1, poptR_AL[0], poptR_AL[1])


fig = plt.figure(figsize=(8,6))
plt.scatter(LengthData.length_AL_total, rGy_AL, color='tab:blue', facecolors='none')
plt.scatter(LengthData.length_calyx_total, rGy_calyx, color='tab:orange', facecolors='none')
plt.scatter(LengthData.length_LH_total, rGy_LH, color='tab:green', facecolors='none')
plt.plot(xvallog1, fitYR_AL, ls='dashed', lw=3)
plt.plot(xvallog1, fitYR_calyx, ls='dashed', lw=3)
plt.plot(xvallog1, fitYR_LH, ls='dashed', lw=3)
plt.yscale('log')
plt.xscale('log')
plt.xlim(1, 10000)
# plt.ylim(7, 4000)
plt.legend(['AL: ' + str(round(poptR_AL[0], 3)) + '$\pm$' + str(round(perrR_AL[0], 3)),
            'MB calyx: ' + str(round(poptR_calyx[0], 3)) + '$\pm$' + str(round(perrR_calyx[0], 3)),
            'LH: ' + str(round(poptR_LH[0], 3)) + '$\pm$' + str(round(perrR_LH[0], 3))], fontsize=15)
# plt.title(r"MB calyx", fontsize=20)
plt.xlabel(r"$L$", fontsize=15)
plt.ylabel(r"$R_{g}$", fontsize=15)
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/rgy_neuropil_fixed_4.pdf', dpi=300, bbox_inches='tight')
plt.show()

print('MB Calyx: ' + str(np.corrcoef(np.log10(LengthData.length_calyx_total), np.log10(rGy_calyx))[0][1]))
print('LH: ' + str(np.corrcoef(np.log10(LengthData.length_LH_total), np.log10(rGy_LH))[0][1]))
print('AL: ' + str(np.corrcoef(np.log10(LengthData.length_AL_total), np.log10(rGy_AL))[0][1]))


#%% Radius of Gyration for calyx, LH, and AL per segment

length_calyx_nempty = copy.deepcopy([x for x in LengthData.length_calyx if x != []])
length_LH_nempty = copy.deepcopy([x for x in LengthData.length_LH if x != []])
length_AL_nempty = copy.deepcopy([x for x in LengthData.length_AL if x != []])

calyx_btrk = copy.deepcopy([x for x in BranchData.calyx_branchTrk if x != []])
calyx_toex = [i for i,x in enumerate(LengthData.length_calyx) if not x]
calyx_bP = copy.deepcopy([element for i, element in enumerate(BranchData.calyx_branchP) if i not in calyx_toex])

LH_btrk = copy.deepcopy([x for x in BranchData.LH_branchTrk if x != []])
LH_toex = [i for i,x in enumerate(LengthData.length_LH) if not x]
LH_bP = copy.deepcopy([element for i, element in enumerate(BranchData.LH_branchP) if i not in LH_toex])

AL_btrk = copy.deepcopy([x for x in BranchData.AL_branchTrk if x != []])
AL_toex = [i for i,x in enumerate(LengthData.length_AL) if not x]
AL_bP = copy.deepcopy([element for i, element in enumerate(BranchData.AL_branchP) if i not in AL_toex])

calyxdist_per_seg_count = []
LHdist_per_seg_count = []
ALdist_per_seg_count = []

rGy_calyx_per_seg = []
rGy_LH_per_seg = []
rGy_AL_per_seg = []

rGy_calyx_per_bP = []
length_calyx_bP = []
rGy_LH_per_bP = []
length_LH_bP = []
rGy_AL_per_bP = []
length_AL_bP = []

un_calyx = np.unique(MorphData.calyxdist_trk)
un_LH = np.unique(MorphData.LHdist_trk)
un_AL = np.unique(MorphData.ALdist_trk)

for i in range(len(un_calyx)):
    idx = np.where(MorphData.calyxdist_trk == un_calyx[i])[0]
    tarval = np.array(MorphData.calyxdist)[idx]
    (rGy_t, cML_t) = utils.radiusOfGyration(tarval)
    rGy_calyx_per_seg.append(rGy_t)

for i in range(len(un_LH)):
    idx = np.where(MorphData.LHdist_trk == un_LH[i])[0]
    tarval = np.array(MorphData.LHdist)[idx]
    (rGy_t, cML_t) = utils.radiusOfGyration(tarval)
    rGy_LH_per_seg.append(rGy_t)
    
for i in range(len(un_AL)):
    idx = np.where(MorphData.ALdist_trk == un_AL[i])[0]
    tarval = np.array(MorphData.ALdist)[idx]
    (rGy_t, cML_t) = utils.radiusOfGyration(tarval)
    rGy_AL_per_seg.append(rGy_t)

#%%

xvallog1 = np.logspace(-3, 3)

poptR_calyx_per_seg = []
perrR_calyx_per_seg = []
fitYR_calyx_per_seg = []

poptR_LH_per_seg = []
perrR_LH_per_seg = []
fitYR_LH_per_seg = []

poptR_AL_per_seg = []
perrR_AL_per_seg = []
fitYR_AL_per_seg = []

for i in range(len(length_calyx_nempty)):
    if len(length_calyx_nempty[i]) > 1:
        popt, pcov = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(length_calyx_nempty[i]), 
                                                np.log10(rGy_calyx_per_seg[i]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
        poptR_calyx_per_seg.append(popt)
        perrR_calyx_per_seg.append(np.sqrt(np.diag(pcov)))
        fitYR_calyx_per_seg.append(objFuncPpow(xvallog1, popt[0], popt[1]))

for i in range(len(length_LH_nempty)):
    if len(length_LH_nempty[i]) > 1:
        if 0 in rGy_LH_per_seg[i]:
            pass
        else:
            popt, pcov = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(length_LH_nempty[i]), 
                                                    np.log10(rGy_LH_per_seg[i]), 
                                                    p0=[1., 0.], 
                                                    maxfev=100000)
            poptR_LH_per_seg.append(popt)
            perrR_LH_per_seg.append(np.sqrt(np.diag(pcov)))
            fitYR_LH_per_seg.append(objFuncPpow(xvallog1, popt[0], popt[1]))

for i in range(len(length_AL_nempty)):
    if len(length_AL_nempty[i]) > 1:
        if 0 in rGy_AL_per_seg[i]:
            pass
        else:
            popt, pcov = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(length_AL_nempty[i]), 
                                                    np.log10(rGy_AL_per_seg[i]), 
                                                    p0=[1., 0.], 
                                                    maxfev=100000)
            poptR_AL_per_seg.append(popt)
            perrR_AL_per_seg.append(np.sqrt(np.diag(pcov)))
            fitYR_AL_per_seg.append(objFuncPpow(xvallog1, popt[0], popt[1]))

fig = plt.figure(figsize=(8,6))
for i in range(len(length_calyx_nempty)):
    plt.scatter(length_calyx_nempty[i], rGy_calyx_per_seg[i], s=0.5, color='tab:orange')
for i in range(len(length_LH_nempty)):
    plt.scatter(length_LH_nempty[i], rGy_LH_per_seg[i], s=0.5, color='tab:green')
for i in range(len(length_AL_nempty)):
    plt.scatter(length_AL_nempty[i], rGy_AL_per_seg[i], s=0.5, color='tab:blue')
plt.scatter(LengthData.length_AL_total, rGy_AL, color='tab:blue', facecolors='none')
plt.scatter(LengthData.length_calyx_total, rGy_calyx, color='tab:orange', facecolors='none')
plt.scatter(LengthData.length_LH_total, rGy_LH, color='tab:green', facecolors='none')
plt.scatter(LengthData.length_total, rGy, color='tab:red')

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r"$L$", fontsize=15)
plt.ylabel(r"$R_{g}$", fontsize=15)
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/rgy_neuropil_segment_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


s1 = set(un_calyx) 
s2 = set(un_LH) 
s3 = set(un_AL) 

set1 = s1.intersection(s2)
result_set = set1.intersection(s3) 
all_list = list(result_set)

poptlist = []

for i in np.array(all_list):
    cidx = np.where(un_calyx == i)[0][0]
    lidx = np.where(un_LH == i)[0][0]
    aidx = np.where(un_AL == i)[0][0]
    lenall = length_calyx_nempty[cidx] + length_LH_nempty[lidx] + length_AL_nempty[aidx]
    lenall.append(LengthData.length_AL_total[aidx])
    lenall.append(LengthData.length_calyx_total[cidx])
    lenall.append(LengthData.length_LH_total[lidx])
    lenall.append(LengthData.length_total[i])
    lenall = np.array(lenall)
    rall = rGy_calyx_per_seg[cidx].tolist() + rGy_LH_per_seg[lidx].tolist() + rGy_AL_per_seg[aidx].tolist()
    rall.append(rGy_AL[aidx])
    rall.append(rGy_calyx[cidx])
    rall.append(rGy_LH[lidx])
    rall.append(rGy[i])
    rall = np.array(rall)
    taridx = np.where(lenall > 40)
    popt, pcov = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(lenall[taridx[0]]), 
                                                    np.log10(rall[taridx[0]]), 
                                                    p0=[1., 0.], 
                                                    maxfev=100000)
    poptlist.append(popt)


#%% Radius of Gyration for calyx, LH, and AL per branching point

for i in range(len(calyx_bP)):
    rGy_calyx_per_bP_temp = []
    length_calyx_bP_temp = []
    idx = np.where(np.array(MorphData.calyxdist_trk) == un_calyx[i])[0]
    tarval = np.array(MorphData.calyxdist)[idx]
    for j in range(len(calyx_bP[i])):
        temp_tar = []
        temp_len = 0
        for k in range(len(calyx_btrk[i])):
            if calyx_btrk[i][k][0] == calyx_bP[i][j] or calyx_btrk[i][k][-1] == calyx_bP[i][j]:
                temp_tar.append(tarval[k])
                temp_len += length_calyx_nempty[i][k]
                
        (rGy_t, cML_t) = utils.radiusOfGyration([[item for sublist in temp_tar for item in sublist]])
        rGy_calyx_per_bP_temp.append(rGy_t)
        length_calyx_bP_temp.append(temp_len)
    
    rGy_calyx_per_bP.append(np.squeeze(rGy_calyx_per_bP_temp))
    length_calyx_bP.append(length_calyx_bP_temp)

for i in range(len(LH_bP)):
    rGy_LH_per_bP_temp = []
    length_LH_bP_temp = []
    idx = np.where(np.array(MorphData.LHdist_trk) == un_LH[i])[0]
    tarval = np.array(MorphData.LHdist)[idx]
    for j in range(len(LH_bP[i])):
        temp_tar = []
        temp_len = 0
        for k in range(len(LH_btrk[i])):
            if LH_btrk[i][k][0] == LH_bP[i][j] or LH_btrk[i][k][-1] == LH_bP[i][j]:
                temp_tar.append(tarval[k])
                temp_len += length_LH_nempty[i][k]
                
        (rGy_t, cML_t) = utils.radiusOfGyration([[item for sublist in temp_tar for item in sublist]])
        rGy_LH_per_bP_temp.append(rGy_t)
        length_LH_bP_temp.append(temp_len)
    rGy_LH_per_bP.append(np.squeeze(rGy_LH_per_bP_temp))
    length_LH_bP.append(length_LH_bP_temp)

for i in range(len(AL_bP)):
    rGy_AL_per_bP_temp = []
    length_AL_bP_temp = []
    idx = np.where(np.array(MorphData.ALdist_trk) == un_AL[i])[0]
    tarval = np.array(MorphData.ALdist)[idx]
    for j in range(len(AL_bP[i])):
        temp_tar = []
        temp_len = 0
        for k in range(len(AL_btrk[i])):
            if AL_btrk[i][k][0] == AL_bP[i][j] or AL_btrk[i][k][-1] == AL_bP[i][j]:
                temp_tar.append(tarval[k])
                temp_len += length_AL_nempty[i][k]
                
        (rGy_t, cML_t) = utils.radiusOfGyration([[item for sublist in temp_tar for item in sublist]])
        rGy_AL_per_bP_temp.append(rGy_t)
        length_AL_bP_temp.append(temp_len)
    rGy_AL_per_bP.append(np.squeeze(rGy_AL_per_bP_temp))
    length_AL_bP.append(length_AL_bP_temp)



#%% Length study

fig = plt.figure(figsize=(8,6))
plt.hist(LengthData.length_AL_total, 
          bins=int((np.max(LengthData.length_AL_total) - np.min(LengthData.length_AL_total))/100),
          density=True,
          alpha=0.5)

plt.hist(LengthData.length_calyx_total, 
          bins=int((np.max(LengthData.length_calyx_total) - np.min(LengthData.length_calyx_total))/100),
          density=True,
          alpha=0.5)

plt.hist(LengthData.length_LH_total, 
          bins=int((np.max(LengthData.length_LH_total) - np.min(LengthData.length_LH_total))/100),
          density=True,
          alpha=0.5)

plt.hist(LengthData.length_total, 
          bins=int((np.max(LengthData.length_total) - np.min(LengthData.length_total))/100),
          density=True,
          alpha=0.5)

plt.ylabel("Probability", fontsize=15)
plt.xlabel("L", fontsize=15)
plt.legend(['AL', 'MB calyx', 'LH', 'Total'], fontsize=13)
plt.xlim(0, 4500)
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/total_length_hist_comb_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(4, 1, figsize=(6,10))
ax[0].hist(LengthData.length_AL_total, 
          bins=int((np.max(LengthData.length_AL_total) - np.min(LengthData.length_AL_total))/100),
          density=True,
          color='tab:blue')

ax[1].hist(LengthData.length_calyx_total, 
          bins=int((np.max(LengthData.length_calyx_total) - np.min(LengthData.length_calyx_total))/100),
          density=True,
          color='tab:orange')

ax[2].hist(LengthData.length_LH_total, 
          bins=int((np.max(LengthData.length_LH_total) - np.min(LengthData.length_LH_total))/100),
          density=True,
          color='tab:green')

ax[3].hist(LengthData.length_total, 
          bins=int((np.max(LengthData.length_total) - np.min(LengthData.length_total))/100),
          density=True,
          color='tab:red')

ax[0].set_ylabel("AL", fontsize=15)
ax[1].set_ylabel("MB calyx", fontsize=15)
ax[2].set_ylabel("LH", fontsize=15)
ax[3].set_ylabel("Total", fontsize=15)
ax[3].set_xlabel("L", fontsize=15)
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[2].set_xticklabels([])
ax[0].set_xlim(0, 4500)
ax[1].set_xlim(0, 4500)
ax[2].set_xlim(0, 4500)
ax[3].set_xlim(0, 4500)

plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/total_length_hist_sep_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(8,6))
lab = ['AL', 'MB calyx', 'LH', 'Total']
medianprops = dict(linestyle='-', linewidth=1.5, color='k')
bp = ax.boxplot([LengthData.length_AL_total, LengthData.length_calyx_total, 
             LengthData.length_LH_total, LengthData.length_total], 
            notch=False, vert=False, patch_artist=True, labels=lab,
            medianprops=medianprops, positions=[4,3,2,1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax.tick_params(axis = 'y', which = 'minor', labelsize = 15)
ax.set_xlabel('L', fontsize=15)
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/total_length_box_2.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Regional dist categorization

glo_info = pd.read_excel(os.path.join(Parameter.PATH, '../all_skeletons_type_list_180919.xls'))

glo_list = []
glo_idx = []

for f in range(len(MorphData.neuron_id)):
    idx = np.where(glo_info.skid == int(MorphData.neuron_id[f]))[0][0]
    if 'glomerulus' in glo_info['old neuron name'][idx]:
        if glo_info['type'][idx] != 'unknown glomerulus': # One neuron in this glomerulus that does not project to LH
            if glo_info['type'][idx] == 'DP1l, VL2p': # Neuron with both DP1l and VL2p label
                glo_name = 'VL2p' # Neuron seems to have more similar spetrum as VL2p
            else:
                glo_name = glo_info['type'][idx]
                
            if glo_name in glo_list:
                glo_idx[glo_list.index(glo_name)].append(f)
            else:
                glo_list.append(glo_name)
                glo_idx.append([f])

glo_len = [len(arr) for arr in glo_idx]
glo_lb = [sum(glo_len[0:i]) for i in range(len(glo_len)+1)]
glo_lbs = np.subtract(glo_lb, glo_lb[0])
glo_float = np.divide(glo_lbs, glo_lbs[-1])

glo_list_neuron = np.repeat(glo_list, glo_len)
glo_lb_idx = []

for i in range(len(glo_lb)-1):
    glo_lb_idx.append(np.arange(glo_lb[i],glo_lb[i+1]))

morph_dist_calyx = []
morph_dist_LH = []
morph_dist_AL = []

morph_dist_calyx_bp = []
morph_dist_LH_bp = []
morph_dist_AL_bp = []

morph_dist_calyx_ep = []
morph_dist_LH_ep = []
morph_dist_AL_ep = []

for i in range(len(glo_list)):
    morph_dist_calyx_temp = []
    morph_dist_LH_temp = []
    morph_dist_AL_temp = []
    morph_dist_calyx_bp_temp = []
    morph_dist_LH_bp_temp = []
    morph_dist_AL_bp_temp = []
    morph_dist_calyx_ep_temp = []
    morph_dist_LH_ep_temp = []
    morph_dist_AL_ep_temp = []
    for j in range(len(glo_idx[i])):
        morph_dist_calyx_temp2 = []
        morph_dist_LH_temp2 = []
        morph_dist_AL_temp2 = []
        morph_dist_calyx_bp_temp2 = []
        morph_dist_LH_bp_temp2 = []
        morph_dist_AL_bp_temp2 = []
        morph_dist_calyx_ep_temp2 = []
        morph_dist_LH_ep_temp2 = []
        morph_dist_AL_ep_temp2 = []
        
        for p in range(len(MorphData.morph_dist[glo_idx[i][j]])):
            
            branch_dist_temp2_rot = roty.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            branch_dist_temp2_rot2 = rotx.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            branch_dist_temp2_rot3 = rotz.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))

            # if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
            #     (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
            #     (np.array(branch_dist_temp2_rot)[2] > 354.95).all() and (np.array(branch_dist_temp2_rot)[2] < 399.06).all()):
            if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
                (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
                (np.array(branch_dist_temp2_rot3)[2] > 434.08).all() and (np.array(branch_dist_temp2_rot3)[2] < 496.22).all()):
                morph_dist_calyx_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
            # elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and
            #       (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 278.76).all() and
            #       (np.array(branch_dist_temp2_rot)[2] < 345.93).all()):
            elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and
                  (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 286.78).all() and
                  (np.array(branch_dist_temp2_rot)[2] < 343.93).all()):
                morph_dist_LH_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
            # elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 516.38).all() and 
            #       (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 361.12).all() and
            #       (np.array(branch_dist_temp2_rot2)[2] < -77.84).all()):
            elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 533.42).all() and 
                  (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 363.12).all() and
                  (np.array(branch_dist_temp2_rot2)[2] < 180.77).all()):
                morph_dist_AL_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
        
        for q in range(len(BranchData.branchP_dist[glo_idx[i][j]])):
            
            branch_dist_temp2_rot = roty.apply(np.array(BranchData.branchP_dist[glo_idx[i][j]][q]))
            branch_dist_temp2_rot2 = rotx.apply(np.array(BranchData.branchP_dist[glo_idx[i][j]][q]))
            branch_dist_temp2_rot3 = rotz.apply(np.array(BranchData.branchP_dist[glo_idx[i][j]][q]))
            
            # if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
            #     (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
            #     (np.array(branch_dist_temp2_rot)[2] > 354.95).all() and (np.array(branch_dist_temp2_rot)[2] < 399.06).all()):
            if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
                (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
                (np.array(branch_dist_temp2_rot3)[2] > 434.08).all() and (np.array(branch_dist_temp2_rot3)[2] < 496.22).all()):
                morph_dist_calyx_bp_temp2.append(np.array(BranchData.branchP_dist[glo_idx[i][j]][q]))
            # elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and
            #       (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 278.76).all() and
            #       (np.array(branch_dist_temp2_rot)[2] < 345.93).all()):
            elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and
                  (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 286.78).all() and
                  (np.array(branch_dist_temp2_rot)[2] < 343.93).all()):
                morph_dist_LH_bp_temp2.append(np.array(BranchData.branchP_dist[glo_idx[i][j]][q]))
            # elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 516.38).all() and 
            #       (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 361.12).all() and
            #       (np.array(branch_dist_temp2_rot2)[2] < -77.84).all()):
            elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 533.42).all() and 
                  (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 363.12).all() and
                  (np.array(branch_dist_temp2_rot2)[2] < 180.77).all()):
                morph_dist_AL_bp_temp2.append(np.array(BranchData.branchP_dist[glo_idx[i][j]][q]))
        
        for r in range(len(MorphData.endP_dist[glo_idx[i][j]])):
            
            branch_dist_temp2_rot = roty.apply(np.array(MorphData.endP_dist[glo_idx[i][j]][r]))
            branch_dist_temp2_rot2 = rotx.apply(np.array(MorphData.endP_dist[glo_idx[i][j]][r]))
            branch_dist_temp2_rot3 = rotz.apply(np.array(MorphData.endP_dist[glo_idx[i][j]][r]))
            
            # if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
            #     (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
            #     (np.array(branch_dist_temp2_rot)[2] > 354.95).all() and (np.array(branch_dist_temp2_rot)[2] < 399.06).all()):
            if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
                (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
                (np.array(branch_dist_temp2_rot3)[2] > 434.08).all() and (np.array(branch_dist_temp2_rot3)[2] < 496.22).all()):
                morph_dist_calyx_ep_temp2.append(np.array(MorphData.endP_dist[glo_idx[i][j]][r]))
            # elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 190.71).all() and
            #       (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 278.76).all() and
            #       (np.array(branch_dist_temp2_rot)[2] < 345.93).all()):
            elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and
                  (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 286.78).all() and
                  (np.array(branch_dist_temp2_rot)[2] < 343.93).all()):
                morph_dist_LH_ep_temp2.append(np.array(MorphData.endP_dist[glo_idx[i][j]][r]))
            # elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 516.38).all() and 
            #       (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 361.12).all() and
            #       (np.array(branch_dist_temp2_rot2)[2] < -77.84).all()):
            elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 533.42).all() and 
                  (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 363.12).all() and
                  (np.array(branch_dist_temp2_rot2)[2] < 180.77).all()):
                morph_dist_AL_ep_temp2.append(np.array(MorphData.endP_dist[glo_idx[i][j]][r]))
        
        morph_dist_calyx_temp.append(morph_dist_calyx_temp2)
        morph_dist_LH_temp.append(morph_dist_LH_temp2)
        morph_dist_AL_temp.append(morph_dist_AL_temp2)
        morph_dist_calyx_bp_temp.append(morph_dist_calyx_bp_temp2)
        morph_dist_LH_bp_temp.append(morph_dist_LH_bp_temp2)
        morph_dist_AL_bp_temp.append(morph_dist_AL_bp_temp2)
        morph_dist_calyx_ep_temp.append(morph_dist_calyx_ep_temp2)
        morph_dist_LH_ep_temp.append(morph_dist_LH_ep_temp2)
        morph_dist_AL_ep_temp.append(morph_dist_AL_ep_temp2)
                
    morph_dist_calyx.append(morph_dist_calyx_temp)
    morph_dist_LH.append(morph_dist_LH_temp)
    morph_dist_AL.append(morph_dist_AL_temp)
    morph_dist_calyx_bp.append(morph_dist_calyx_bp_temp)
    morph_dist_LH_bp.append(morph_dist_LH_bp_temp)
    morph_dist_AL_bp.append(morph_dist_AL_bp_temp)
    morph_dist_calyx_ep.append(morph_dist_calyx_ep_temp)
    morph_dist_LH_ep.append(morph_dist_LH_ep_temp)
    morph_dist_AL_ep.append(morph_dist_AL_ep_temp)
    

#%% Radius of Gyration on neurons spanning AL, MB calyx, and LH

glo_idx_flat = np.sort([item for sublist in glo_idx for item in sublist])



poptR, pcovR = scipy.optimize.curve_fit(objFuncGL, 
                                        np.log10(MorphData.morph_dist_len[glo_idx_flat]), 
                                        np.log10(rGy[glo_idx_flat]), 
                                        p0=[1., 0.], 
                                        maxfev=100000)
perrR = np.sqrt(np.diag(pcovR))
fitYR = objFuncPpow(MorphData.morph_dist_len[glo_idx_flat], poptR[0], poptR[1])

fig = plt.figure(figsize=(8,6))
plt.scatter(MorphData.morph_dist_len[glo_idx_flat], rGy[glo_idx_flat])
plt.plot(MorphData.morph_dist_len[glo_idx_flat], fitYR, color='tab:red')
plt.yscale('log')
plt.xscale('log')
#    plt.xlim(10, 10000)
#    plt.ylim(7, 4000)
plt.legend([str(round(poptR[0], 3)) + '$\pm$' + str(round(perrR[0], 3))], fontsize=15)
plt.title(r"Scaling Behavior of $R_{g}$ to Length", fontsize=20)
plt.xlabel(r"Length ($\lambda N$)", fontsize=15)
plt.ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
plt.tight_layout()
plt.show()



#%%

morph_dist_calyx_CM = []
morph_dist_LH_CM = []
morph_dist_AL_CM = []

morph_dist_calyx_std = []
morph_dist_LH_std = []
morph_dist_AL_std = []

for i in range(len(morph_dist_AL)):
    morph_dist_calyx_CM_temp = []
    morph_dist_LH_CM_temp = []
    morph_dist_AL_CM_temp = []
    
    morph_dist_calyx_std_temp = []
    morph_dist_LH_std_temp = []
    morph_dist_AL_std_temp = []
    
    for j in range(len(morph_dist_AL[i])):
        morph_dist_calyx_CM_temp.append(np.average(np.array(morph_dist_calyx[i][j]), axis=0))
        morph_dist_LH_CM_temp.append(np.average(np.array(morph_dist_LH[i][j]), axis=0))
        morph_dist_AL_CM_temp.append(np.average(np.array(morph_dist_AL[i][j]), axis=0))
        
        morph_dist_calyx_std_temp.append(np.std(np.array(morph_dist_calyx[i][j]), axis=0))
        morph_dist_LH_std_temp.append(np.std(np.array(morph_dist_LH[i][j]), axis=0))
        morph_dist_AL_std_temp.append(np.std(np.array(morph_dist_AL[i][j]), axis=0))
    
    morph_dist_calyx_CM.append(morph_dist_calyx_CM_temp)
    morph_dist_LH_CM.append(morph_dist_LH_CM_temp)
    morph_dist_AL_CM.append(morph_dist_AL_CM_temp)
    
    morph_dist_LH_std.append(morph_dist_LH_std_temp)
    morph_dist_calyx_std.append(morph_dist_calyx_std_temp)
    morph_dist_AL_std.append(morph_dist_AL_std_temp)
    
    
#%% Calculate convex hull

from scipy.spatial import ConvexHull

morph_dist_calyx_flt = [item for sublist in morph_dist_calyx for item in sublist]
morph_dist_calyx_flat = [item for sublist in morph_dist_calyx_flt for item in sublist]

mdcalyx_xmax = np.max(np.array(morph_dist_calyx_flat)[:,0])
mdcalyx_xmin = np.min(np.array(morph_dist_calyx_flat)[:,0])
mdcalyx_ymax = np.max(np.array(morph_dist_calyx_flat)[:,1])
mdcalyx_ymin = np.min(np.array(morph_dist_calyx_flat)[:,1])
mdcalyx_zmax = np.max(np.array(morph_dist_calyx_flat)[:,2])
mdcalyx_zmin = np.min(np.array(morph_dist_calyx_flat)[:,2])

morph_dist_LH_flt = [item for sublist in morph_dist_LH for item in sublist]
morph_dist_LH_flat = [item for sublist in morph_dist_LH_flt for item in sublist]

mdLH_xmax = np.max(np.array(morph_dist_LH_flat)[:,0])
mdLH_xmin = np.min(np.array(morph_dist_LH_flat)[:,0])
mdLH_ymax = np.max(np.array(morph_dist_LH_flat)[:,1])
mdLH_ymin = np.min(np.array(morph_dist_LH_flat)[:,1])
mdLH_zmax = np.max(np.array(morph_dist_LH_flat)[:,2])
mdLH_zmin = np.min(np.array(morph_dist_LH_flat)[:,2])

morph_dist_AL_flt = [item for sublist in morph_dist_AL for item in sublist]
morph_dist_AL_flat = [item for sublist in morph_dist_AL_flt for item in sublist]

mdAL_xmax = np.max(np.array(morph_dist_AL_flat)[:,0])
mdAL_xmin = np.min(np.array(morph_dist_AL_flat)[:,0])
mdAL_ymax = np.max(np.array(morph_dist_AL_flat)[:,1])
mdAL_ymin = np.min(np.array(morph_dist_AL_flat)[:,1])
mdAL_zmax = np.max(np.array(morph_dist_AL_flat)[:,2])
mdAL_zmin = np.min(np.array(morph_dist_AL_flat)[:,2])

hull_calyx = ConvexHull(np.array(morph_dist_calyx_flat))
calyx_vol = hull_calyx.volume
calyx_area = hull_calyx.area
calyx_density_l = np.sum(LengthData.length_calyx_total)/calyx_vol
calyx_density_c = len(calyx_dist_flat)/calyx_vol

hull_LH = ConvexHull(np.array(morph_dist_LH_flat))
LH_vol = hull_LH.volume
LH_area = hull_LH.area
LH_density_l = np.sum(LengthData.length_LH_total)/LH_vol
LH_density_c = len(LH_dist_flat)/LH_vol

hull_AL = ConvexHull(np.array(morph_dist_AL_flat))
AL_vol = hull_AL.volume
AL_area = hull_AL.area
AL_density_l = np.sum(LengthData.length_AL_total)/AL_vol
AL_density_c = len(AL_dist_flat)/AL_vol

    
#%% Scatterplot of CM based on glomeruli ID

markerlist = ["o", "v", "^", "<", ">", "1", "2", "s", "p", "P", "*", "H", "+", 
              "x", "D", "o", "v", "^", "<", ">", "1", "2", "s", "p", "P", "*", 
              "H", "+", "x", "D", "o", "v", "^", "<", ">", "1", "2", "s", "p", 
              "P", "*", "H", "+", "x", "D", "o", "v", "^", "<", ">", "1", "2",
              "s", "p", "P", "*", "H", "+", "x", "D"]

morph_dist_calyx_CMCM = []
morph_dist_LH_CMCM = []
morph_dist_AL_CMCM = []

fig = plt.figure(figsize=(24, 16))
ax = plt.axes(projection='3d')
cmap = cm.get_cmap('tab20', len(glo_list))
for f in range(len(glo_list)):
    calyxtemp = np.array(morph_dist_calyx_CM[f])
    ax.scatter(calyxtemp[:,0], 
               calyxtemp[:,1], 
               calyxtemp[:,2], 
               color=cmap(f), 
               marker=markerlist[f],
               label=str(glo_list[f]),
               depthshade=False)
    morph_dist_calyx_CMCM.append(np.average(np.array(calyxtemp), axis=0))
ax.legend()
for f in range(len(glo_list)):
    LHtemp = np.array(morph_dist_LH_CM[f])
    ax.scatter(LHtemp[:,0], 
               LHtemp[:,1], 
               LHtemp[:,2], 
               color=cmap(f), 
               marker=markerlist[f],
               label=str(glo_list[f]),
               depthshade=False)
    ALtemp = np.array(morph_dist_AL_CM[f])
    ax.scatter(ALtemp[:,0], 
               ALtemp[:,1], 
               ALtemp[:,2], 
               color=cmap(f), 
               marker=markerlist[f],
               label=str(glo_list[f]),
               depthshade=False)
    morph_dist_LH_CMCM.append(np.average(np.array(LHtemp), axis=0))
    morph_dist_AL_CMCM.append(np.average(np.array(ALtemp), axis=0))
ax.set_xlim(430, 580)
ax.set_ylim(210, 350)
ax.set_zlim(40, 170)
plt.show()


#%% Plot calyx per glomerulus

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
cmap = cm.get_cmap('jet', len(morph_dist_calyx))
for i in range(len(MorphData.calyxdist)):
    glo_n = MorphData.calyxdist_trk[i]
    isglo = [i for i, idx in enumerate(glo_idx) if glo_n in idx]
    listOfPoints = MorphData.calyxdist[i]
    if len(isglo) > 0:
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(isglo[0]), lw=0.25)
    else:
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color='gray', lw=0.25)
ax.grid(True)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlim(490, 550)
ax.set_ylim(350, 150)
ax.set_zlim(160, 190)

# plt.savefig(os.path.join(Parameter.outputdir, 'neurons_calyx_7'), dpi=300, bbox_inches='tight')
plt.show()


#%% Plot LH per glomerulus

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
cmap = cm.get_cmap('jet', len(morph_dist_LH))
for i in range(len(MorphData.LHdist)):
    glo_n = MorphData.LHdist_trk[i]
    isglo = [i for i, idx in enumerate(glo_idx) if glo_n in idx]
    listOfPoints = MorphData.LHdist[i]
    if len(isglo) > 0:
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(isglo[0]), lw=0.25)
    else:
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color='gray', lw=0.25)
ax.grid(True)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlim(410, 480)
ax.set_ylim(350, 150)
ax.set_zlim(135, 175)

# plt.savefig(os.path.join(Parameter.outputdir, 'neurons_LH_7'), dpi=300, bbox_inches='tight')
plt.show()


#%% Plot AL per glomerulus

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
cmap = cm.get_cmap('jet', len(morph_dist_AL))
for i in range(len(MorphData.ALdist)):
    glo_n = MorphData.ALdist_trk[i]
    isglo = [i for i, idx in enumerate(glo_idx) if glo_n in idx]
    listOfPoints = MorphData.ALdist[i]
    if len(isglo) > 0:
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(isglo[0]), lw=0.25)
    else:
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color='gray', lw=0.25)
ax.grid(True)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlim(485, 585)
ax.set_ylim(430, 70)
ax.set_zlim(25, 75)

# plt.savefig(os.path.join(Parameter.outputdir, 'neurons_AL_7'), dpi=300, bbox_inches='tight')
plt.show()


#%% Cluster quantification

morph_dist_calyx_CM_flat = np.array([item for sublist in morph_dist_calyx_CM for item in sublist])
morph_dist_LH_CM_flat = np.array([item for sublist in morph_dist_LH_CM for item in sublist])
morph_dist_AL_CM_flat = np.array([item for sublist in morph_dist_AL_CM for item in sublist])

morph_dist_calyx_r = scipy.spatial.distance.cdist(morph_dist_calyx_CM_flat, morph_dist_calyx_CM_flat)
morph_dist_LH_r = scipy.spatial.distance.cdist(morph_dist_LH_CM_flat, morph_dist_LH_CM_flat)
morph_dist_AL_r = scipy.spatial.distance.cdist(morph_dist_AL_CM_flat, morph_dist_AL_CM_flat)

calyxdist_cluster_u_full = []
calyxdist_noncluster_u_full = []

for i in range(len(glo_list)):
    calyx_sq = morph_dist_calyx_r[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i]:glo_lbs[i+1]]
    calyx_sq_tri = calyx_sq[np.triu_indices_from(calyx_sq, k=1)]
    calyx_nc = morph_dist_calyx_r[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i+1]:]
        
    if len(calyx_sq_tri) > 0:
        calyxdist_cluster_u_full.append(calyx_sq_tri)
    calyxdist_noncluster_u_full.append(calyx_nc.flatten())        

calyxdist_cluster_u_full_flat = [item for sublist in calyxdist_cluster_u_full for item in sublist]
calyxdist_noncluster_u_full_flat = [item for sublist in calyxdist_noncluster_u_full for item in sublist]

# calyxdist_cluster_u_full_flat = np.divide(calyxdist_cluster_u_full_flat, np.cbrt(calyx_vol))
# calyxdist_noncluster_u_full_flat = np.divide(calyxdist_noncluster_u_full_flat, np.cbrt(calyx_vol))


LHdist_cluster_u_full = []
LHdist_noncluster_u_full = []

for i in range(len(glo_list)):
    LH_sq = morph_dist_LH_r[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i]:glo_lbs[i+1]]
    LH_sq_tri = LH_sq[np.triu_indices_from(LH_sq, k=1)]
    LH_nc = morph_dist_LH_r[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i+1]:]
        
    if len(LH_sq_tri) > 0:
        LHdist_cluster_u_full.append(LH_sq_tri)
    LHdist_noncluster_u_full.append(LH_nc.flatten())        

LHdist_cluster_u_full_flat = [item for sublist in LHdist_cluster_u_full for item in sublist]
LHdist_noncluster_u_full_flat = [item for sublist in LHdist_noncluster_u_full for item in sublist]

# LHdist_cluster_u_full_flat = np.divide(LHdist_cluster_u_full_flat, np.cbrt(LH_vol))
# LHdist_noncluster_u_full_flat = np.divide(LHdist_noncluster_u_full_flat, np.cbrt(LH_vol))


ALdist_cluster_u_full = []
ALdist_noncluster_u_full = []

for i in range(len(glo_list)):
    AL_sq = morph_dist_AL_r[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i]:glo_lbs[i+1]]
    AL_sq_tri = AL_sq[np.triu_indices_from(AL_sq, k=1)]
    AL_nc = morph_dist_AL_r[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i+1]:]
        
    if len(AL_sq_tri) > 0:
        ALdist_cluster_u_full.append(AL_sq_tri)
    ALdist_noncluster_u_full.append(AL_nc.flatten())        

ALdist_cluster_u_full_flat = [item for sublist in ALdist_cluster_u_full for item in sublist]
ALdist_noncluster_u_full_flat = [item for sublist in ALdist_noncluster_u_full for item in sublist]

# ALdist_cluster_u_full_flat = np.divide(ALdist_cluster_u_full_flat, np.cbrt(AL_vol))
# ALdist_noncluster_u_full_flat = np.divide(ALdist_noncluster_u_full_flat, np.cbrt(AL_vol))


print("Calyx cluster Mean: " + str(np.mean(calyxdist_cluster_u_full_flat)) + ", STD: " + str(np.std(calyxdist_cluster_u_full_flat)))
print("Calyx noncluster Mean: " + str(np.mean(calyxdist_noncluster_u_full_flat)) + ", STD: " + str(np.std(calyxdist_noncluster_u_full_flat)))

print("LH cluster Mean: " + str(np.mean(LHdist_cluster_u_full_flat)) + ", STD: " + str(np.std(LHdist_cluster_u_full_flat)))
print("LH noncluster Mean: " + str(np.mean(LHdist_noncluster_u_full_flat)) + ", STD: " + str(np.std(LHdist_noncluster_u_full_flat)))

print("AL cluster Mean: " + str(np.mean(ALdist_cluster_u_full_flat)) + ", STD: " + str(np.std(ALdist_cluster_u_full_flat)))
print("AL noncluster Mean: " + str(np.mean(ALdist_noncluster_u_full_flat)) + ", STD: " + str(np.std(ALdist_noncluster_u_full_flat)))


#%% Cluster quantification using new distance metric

morph_dist_calyx_r_new = np.zeros((len(morph_dist_calyx_CM_flat), len(morph_dist_calyx_CM_flat)))
morph_dist_LH_r_new = np.zeros((len(morph_dist_LH_CM_flat), len(morph_dist_LH_CM_flat)))
morph_dist_AL_r_new = np.zeros((len(morph_dist_AL_CM_flat), len(morph_dist_AL_CM_flat)))

for i in range(len(morph_dist_calyx_CM_flat)):
    for j in range(len(morph_dist_calyx_CM_flat)):
        morph_dist_calyx_ed = scipy.spatial.distance.cdist(morph_dist_calyx_flt[i], morph_dist_calyx_flt[j])
        morph_dist_LH_ed = scipy.spatial.distance.cdist(morph_dist_LH_flt[i], morph_dist_LH_flt[j])
        morph_dist_AL_ed = scipy.spatial.distance.cdist(morph_dist_AL_flt[i], morph_dist_AL_flt[j])
        
        # NNmetric
        if len(morph_dist_calyx_flt[i]) < len(morph_dist_calyx_flt[j]):
            N_calyx = len(morph_dist_calyx_flt[i])
            dmin_calyx = np.min(morph_dist_calyx_ed, axis=1)
        elif len(morph_dist_calyx_flt[i]) > len(morph_dist_calyx_flt[j]):
            N_calyx = len(morph_dist_calyx_flt[j])
            dmin_calyx = np.min(morph_dist_calyx_ed, axis=0)
        else:
            N_calyx = len(morph_dist_calyx_flt[i])
            r1 = np.min(morph_dist_calyx_ed, axis=0)
            r2 = np.min(morph_dist_calyx_ed, axis=1)
            if np.sum(r1) < np.sum(r2):
                dmin_calyx = r1
            else:
                dmin_calyx = r2
        
        if len(morph_dist_LH_flt[i]) < len(morph_dist_LH_flt[j]):
            N_LH = len(morph_dist_LH_flt[i])
            dmin_LH = np.min(morph_dist_LH_ed, axis=1)
        elif len(morph_dist_LH_flt[i]) > len(morph_dist_LH_flt[j]):
            N_LH = len(morph_dist_LH_flt[j])
            dmin_LH = np.min(morph_dist_LH_ed, axis=0)
        else:
            N_LH = len(morph_dist_LH_flt[i])
            r1 = np.min(morph_dist_LH_ed, axis=0)
            r2 = np.min(morph_dist_LH_ed, axis=1)
            if np.sum(r1) < np.sum(r2):
                dmin_LH = r1
            else:
                dmin_LH = r2
        
        if len(morph_dist_AL_flt[i]) < len(morph_dist_AL_flt[j]):
            N_AL = len(morph_dist_AL_flt[i])
            dmin_AL = np.min(morph_dist_AL_ed, axis=1)
        elif len(morph_dist_AL_flt[i]) > len(morph_dist_AL_flt[j]):
            N_AL = len(morph_dist_AL_flt[j])
            dmin_AL = np.min(morph_dist_AL_ed, axis=0)
        else:
            N_AL = len(morph_dist_AL_flt[i])
            r1 = np.min(morph_dist_AL_ed, axis=0)
            r2 = np.min(morph_dist_AL_ed, axis=1)
            if np.sum(r1) < np.sum(r2):
                dmin_AL = r1
            else:
                dmin_AL = r2
        
        morph_dist_calyx_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_calyx)), N_calyx))
        morph_dist_LH_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_LH)), N_LH))
        morph_dist_AL_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_AL)), N_AL))
        
        # Nmetric
        # morph_dist_calyx_r_new[i][j] = np.sqrt(np.divide(np.sum(np.divide(np.square(morph_dist_calyx_ed), 
        #                                                                   np.shape(morph_dist_calyx_ed)[0])), 
        #                                                  np.shape(morph_dist_calyx_ed)[1]))
        # morph_dist_LH_r_new[i][j] = np.sqrt(np.divide(np.sum(np.divide(np.square(morph_dist_LH_ed), 
        #                                                                   np.shape(morph_dist_LH_ed)[0])), 
        #                                                  np.shape(morph_dist_LH_ed)[1]))
        # morph_dist_AL_r_new[i][j] = np.sqrt(np.divide(np.sum(np.divide(np.square(morph_dist_AL_ed), 
        #                                                                   np.shape(morph_dist_AL_ed)[0])), 
        #                                                  np.shape(morph_dist_AL_ed)[1]))

calyxdist_cluster_u_full_new = []
calyxdist_noncluster_u_full_new = []

for i in range(len(glo_list)):
    calyx_sq = morph_dist_calyx_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i]:glo_lbs[i+1]]
    calyx_sq_tri = calyx_sq[np.triu_indices_from(calyx_sq, k=1)]
    calyx_nc = morph_dist_calyx_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i+1]:]
        
    if len(calyx_sq_tri) > 0:
        calyxdist_cluster_u_full_new.append(calyx_sq_tri)
    else:
        calyxdist_cluster_u_full_new.append([])
    calyxdist_noncluster_u_full_new.append(calyx_nc.flatten())

calyxdist_cluster_u_full_flat_new = [item for sublist in calyxdist_cluster_u_full_new for item in sublist]
calyxdist_noncluster_u_full_flat_new = [item for sublist in calyxdist_noncluster_u_full_new for item in sublist]

LHdist_cluster_u_full_new = []
LHdist_noncluster_u_full_new = []

for i in range(len(glo_list)):
    LH_sq = morph_dist_LH_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i]:glo_lbs[i+1]]
    LH_sq_tri = LH_sq[np.triu_indices_from(LH_sq, k=1)]
    LH_nc = morph_dist_LH_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i+1]:]
        
    if len(LH_sq_tri) > 0:
        LHdist_cluster_u_full_new.append(LH_sq_tri)
    else:
        LHdist_cluster_u_full_new.append([])
    LHdist_noncluster_u_full_new.append(LH_nc.flatten())

LHdist_cluster_u_full_flat_new = [item for sublist in LHdist_cluster_u_full_new for item in sublist]
LHdist_noncluster_u_full_flat_new = [item for sublist in LHdist_noncluster_u_full_new for item in sublist]

ALdist_cluster_u_full_new = []
ALdist_noncluster_u_full_new = []

for i in range(len(glo_list)):
    AL_sq = morph_dist_AL_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i]:glo_lbs[i+1]]
    AL_sq_tri = AL_sq[np.triu_indices_from(AL_sq, k=1)]
    AL_nc = morph_dist_AL_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i+1]:]
        
    if len(AL_sq_tri) > 0:
        ALdist_cluster_u_full_new.append(AL_sq_tri)
    else:
        ALdist_cluster_u_full_new.append([])
    ALdist_noncluster_u_full_new.append(AL_nc.flatten())

ALdist_cluster_u_full_flat_new = [item for sublist in ALdist_cluster_u_full_new for item in sublist]
ALdist_noncluster_u_full_flat_new = [item for sublist in ALdist_noncluster_u_full_new for item in sublist]


print("Calyx cluster Mean: " + str(np.mean(calyxdist_cluster_u_full_flat_new)) + ", STD: " + str(np.std(calyxdist_cluster_u_full_flat_new)))
print("Calyx noncluster Mean: " + str(np.mean(calyxdist_noncluster_u_full_flat_new)) + ", STD: " + str(np.std(calyxdist_noncluster_u_full_flat_new)))

print("LH cluster Mean: " + str(np.mean(LHdist_cluster_u_full_flat_new)) + ", STD: " + str(np.std(LHdist_cluster_u_full_flat_new)))
print("LH noncluster Mean: " + str(np.mean(LHdist_noncluster_u_full_flat_new)) + ", STD: " + str(np.std(LHdist_noncluster_u_full_flat_new)))

print("AL cluster Mean: " + str(np.mean(ALdist_cluster_u_full_flat_new)) + ", STD: " + str(np.std(ALdist_cluster_u_full_flat_new)))
print("AL noncluster Mean: " + str(np.mean(ALdist_noncluster_u_full_flat_new)) + ", STD: " + str(np.std(ALdist_noncluster_u_full_flat_new)))

#%%

import scipy.stats

fig, ax = plt.subplots()
labels = ['Calyx', 'LH', 'AL']
x = np.arange(len(labels))
width = .3

cmeans = [np.mean(calyxdist_cluster_u_full_flat), np.mean(LHdist_cluster_u_full_flat), np.mean(ALdist_cluster_u_full_flat)]
cerr = [np.std(calyxdist_cluster_u_full_flat), np.std(LHdist_cluster_u_full_flat), np.std(ALdist_cluster_u_full_flat)]
ncmeans = [np.mean(calyxdist_noncluster_u_full_flat), np.mean(LHdist_noncluster_u_full_flat), np.mean(ALdist_noncluster_u_full_flat)]
ncerr = [np.std(calyxdist_noncluster_u_full_flat), np.std(LHdist_noncluster_u_full_flat), np.std(ALdist_noncluster_u_full_flat)]

ax.bar(x - width/2, cmeans, width, yerr=cerr, capsize=5, label='Cluster')
ax.bar(x + width/2, ncmeans, width, yerr=ncerr, capsize=5, label='Non-Cluster')
ax.set_ylabel('Distance')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_title('Average distance within and outside cluster')
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(5, 9))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.hist(ALdist_cluster_u_full_flat, bins=20, alpha=0.5, density=True)
ax1.hist(ALdist_noncluster_u_full_flat, bins=20, alpha=0.5, density=True)
# ax1.vlines(np.mean(ALdist_cluster_u_full_flat), 0, 0.12, color='tab:blue')
# ax1.vlines(np.mean(ALdist_noncluster_u_full_flat), 0, 0.12, color='tab:orange')
# ax1.vlines(np.median(ALdist_cluster_u_full_flat), 0, 0.12, ls='--', color='tab:blue')
# ax1.vlines(np.median(ALdist_noncluster_u_full_flat), 0, 0.12, ls='--', color='tab:orange')
ax1.set_ylim(0, 0.12)
ax1.set_ylabel('AL', fontsize=15)
ax1.legend(['Identical Glomerulus', 'Different Glomeruli'], fontsize=13)
ax2.hist(calyxdist_cluster_u_full_flat, bins=20, alpha=0.5, density=True)
ax2.hist(calyxdist_noncluster_u_full_flat, bins=20, alpha=0.5, density=True)
# ax2.vlines(np.mean(calyxdist_cluster_u_full_flat), 0, 0.22, color='tab:blue')
# ax2.vlines(np.mean(calyxdist_noncluster_u_full_flat), 0, 0.22, color='tab:orange')
# ax2.vlines(np.median(calyxdist_cluster_u_full_flat), 0, 0.22, ls='--', color='tab:blue')
# ax2.vlines(np.median(calyxdist_noncluster_u_full_flat), 0, 0.22, ls='--', color='tab:orange')
ax2.set_ylim(0, 0.22)
ax2.set_ylabel('MB calyx', fontsize=15)
ax3.hist(LHdist_cluster_u_full_flat, bins=20, alpha=0.5, density=True)
ax3.hist(LHdist_noncluster_u_full_flat, bins=20, alpha=0.5, density=True)
# ax3.vlines(np.mean(LHdist_cluster_u_full_flat), 0, 0.2, color='tab:blue')
# ax3.vlines(np.mean(LHdist_noncluster_u_full_flat), 0, 0.2, color='tab:orange')
# ax3.vlines(np.median(LHdist_cluster_u_full_flat), 0, 0.2, ls='--', color='tab:blue')
# ax3.vlines(np.median(LHdist_noncluster_u_full_flat), 0, 0.2, ls='--', color='tab:orange')
ax3.set_ylim(0, 0.2)
ax3.set_ylabel('LH', fontsize=15)
ax3.set_xlabel('Distance', fontsize=15)
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/skewed_dist.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(8,6))
labels = ['MB calyx', 'LH', 'AL']
x = np.arange(len(labels))
width = .3

cmeans = [np.median(calyxdist_cluster_u_full_flat), np.median(LHdist_cluster_u_full_flat), np.median(ALdist_cluster_u_full_flat)]
cerr = [scipy.stats.median_abs_deviation(calyxdist_cluster_u_full_flat, center=np.median), 
        scipy.stats.median_abs_deviation(LHdist_cluster_u_full_flat, center=np.median), 
        scipy.stats.median_abs_deviation(ALdist_cluster_u_full_flat, center=np.median)]
ncmeans = [np.median(calyxdist_noncluster_u_full_flat), np.median(LHdist_noncluster_u_full_flat), np.median(ALdist_noncluster_u_full_flat)]
ncerr = [scipy.stats.median_abs_deviation(calyxdist_noncluster_u_full_flat, center=np.median), 
         scipy.stats.median_abs_deviation(LHdist_noncluster_u_full_flat, center=np.median), 
         scipy.stats.median_abs_deviation(ALdist_noncluster_u_full_flat, center=np.median)]

ax.bar(x - width/2, cmeans, width, yerr=cerr, capsize=5, label='Identical Glomerulus')
ax.bar(x + width/2, ncmeans, width, yerr=ncerr, capsize=5, label='Different Glomeruli')
ax.set_ylabel('Distance', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=15)
ax.legend(fontsize=13)
#ax.set_title('Median distance within and outside cluster')
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/glomerulus_dist_diff_median.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots()
lab = ['Calyx C', 'Calyx NC', 'LH C', 'LH NC', 'AL C', 'AL NC']
plt.boxplot([calyxdist_cluster_u_full_flat, calyxdist_noncluster_u_full_flat, 
             LHdist_cluster_u_full_flat, LHdist_noncluster_u_full_flat, 
             ALdist_cluster_u_full_flat, ALdist_noncluster_u_full_flat], 
            notch=True, labels=lab, bootstrap=500)
plt.title('Median distance within and outside cluster')
plt.tight_layout()
plt.show()


fig = plt.figure()
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(morph_dist_calyx_r)#, vmax=np.max(morph_dist_AL_r))
ax1.set_xticks([]) 
ax1.set_yticks([]) 
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(glo_float)
ax3.set_yticks(glo_float)
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float[1:] + glo_float[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((glo_float[1:] + glo_float[:-1])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=4, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=4, rotation_mode='default')
plt.colorbar()
plt.title("Inter-cluster distance calyx", pad=40)
plt.show()

fig = plt.figure()
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(morph_dist_LH_r)#, vmax=np.max(morph_dist_AL_r))
ax1.set_xticks([]) 
ax1.set_yticks([]) 
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(glo_float)
ax3.set_yticks(glo_float)
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float[1:] + glo_float[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((glo_float[1:] + glo_float[:-1])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=4, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=4, rotation_mode='default')
plt.colorbar()
plt.title("Inter-cluster distance LH", pad=40)
plt.show()

fig = plt.figure()
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(morph_dist_AL_r)#, vmax=np.max(morph_dist_AL_r))
ax1.set_xticks([]) 
ax1.set_yticks([]) 
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(glo_float)
ax3.set_yticks(glo_float)
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float[1:] + glo_float[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((glo_float[1:] + glo_float[:-1])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=4, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=4, rotation_mode='default')
plt.colorbar()
plt.title("Inter-cluster distance AL", pad=40)
plt.show()



calyxtri = morph_dist_calyx_r[np.triu_indices_from(morph_dist_calyx_r, k=1)]
LHtri = morph_dist_LH_r[np.triu_indices_from(morph_dist_LH_r, k=1)]
ALtri = morph_dist_AL_r[np.triu_indices_from(morph_dist_AL_r, k=1)]

df = pd.DataFrame({'calyx': calyxtri})
df['LH'] = LHtri
df['AL'] = ALtri

print(df.corr())

pd.plotting.scatter_matrix(df, figsize=(6, 6))
plt.show()


df = pd.DataFrame({'calyx': calyxdist_cluster_u_full_flat})
df['LH'] = LHdist_cluster_u_full_flat
df['AL'] = ALdist_cluster_u_full_flat

print(df.corr())

pd.plotting.scatter_matrix(df, figsize=(6, 6))
plt.show()


df = pd.DataFrame({'calyx': calyxdist_noncluster_u_full_flat})
df['LH'] = LHdist_noncluster_u_full_flat
df['AL'] = ALdist_noncluster_u_full_flat

print(df.corr())

pd.plotting.scatter_matrix(df, figsize=(6, 6))
plt.show()


ALcalyx_corr = []
ALLH_corr = []
LHcalyx_corr = []

for i in range(len(morph_dist_AL_r)):
    ALcalyx_corr.append(np.corrcoef(morph_dist_calyx_r[i], morph_dist_AL_r[i])[0][1])
    ALLH_corr.append(np.corrcoef(morph_dist_LH_r[i], morph_dist_AL_r[i])[0][1])
    LHcalyx_corr.append(np.corrcoef(morph_dist_calyx_r[i], morph_dist_LH_r[i])[0][1])

ALcalyx_corr_glo = []
ALLH_corr_glo = []

for i in range(len(glo_lb)-1):
    ALcalyx_corr_glo.append(np.array(ALcalyx_corr)[np.arange(glo_lb[i],glo_lb[i+1])])
    ALLH_corr_glo.append(np.array(ALLH_corr)[np.arange(glo_lb[i],glo_lb[i+1])])

ALcalyx_corr_glo_avg = []
ALcalyx_corr_glo_std = []
ALLH_corr_glo_avg = []
ALLH_corr_glo_std = []

for i in range(len(ALcalyx_corr_glo)):
    ALcalyx_corr_glo_avg.append(np.average(ALcalyx_corr_glo[i]))
    ALcalyx_corr_glo_std.append(np.std(ALcalyx_corr_glo[i]))
    ALLH_corr_glo_avg.append(np.average(ALLH_corr_glo[i]))
    ALLH_corr_glo_std.append(np.std(ALLH_corr_glo[i]))


fig = plt.figure()
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.scatter(np.arange(len(ALcalyx_corr))+0.5, ALcalyx_corr, marker='.')
plt.scatter(np.arange(len(ALLH_corr))+0.5, ALLH_corr, marker='.')
plt.vlines(glo_float*len(ALcalyx_corr), -0.4, 0.8, ls='dashed', lw=1)
plt.xlim(0, len(ALcalyx_corr))
plt.ylim(-0.4, 0.8)
plt.ylabel('Correlation Coefficient')
plt.legend(['calyx-AL correlation', 'LH-AL correlation'])
ax1.set_xticks([]) 
ax2 = ax1.twiny()
offset1 = 0, -10
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
ax2.axis["bottom"] = new_axisline1(loc="bottom", axes=ax2, offset=offset1)
ax2.axis["bottom"].minor_ticks.set_ticksize(0)
ax2.set_xticks(glo_float)
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float[1:] + glo_float[:-1])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list))
ax2.axis["bottom"].minor_ticklabels.set(rotation=90, fontsize=4, ha='right')
ax2.axis["top"].minor_ticks.set(visible=False)
ax2.axis["top"].major_ticks.set(visible=False)
plt.show()


fig, ax = plt.subplots()
x = np.arange(len(glo_list))
width = 1.
ax.bar(x, 
       ALcalyx_corr_glo_avg, 
       width, 
       yerr=ALcalyx_corr_glo_std, 
       label='Calyx-AL', 
       alpha=0.5, 
       error_kw=dict(ecolor='tab:blue', lw=1, capsize=2, capthick=1))
ax.bar(x, 
       ALLH_corr_glo_avg,
       width, 
       yerr=ALLH_corr_glo_std,
       label='LH-AL', 
       alpha=0.5,
       error_kw=dict(ecolor='tab:orange', lw=1, capsize=2, capthick=1))
ax.set_ylabel('Correlation Coefficient')
ax.set_xticks(x)
ax.set_xticklabels(glo_list, rotation=90, fontsize=7)
ax.legend()
ax.set_title('Distance correlation between calyx/LH and AL by glomerulus')
plt.xlim(0-0.5, len(glo_list)-0.5)
plt.tight_layout()
plt.show()


validx = np.argwhere(np.array(ALLH_corr_glo_avg) > 0.6).T[0]

diffidx = np.argwhere(np.subtract(ALLH_corr_glo_avg, ALcalyx_corr_glo_avg) > 0.6).T[0]

print(np.sort(np.array(glo_list)[validx]))
print(np.sort(np.array(glo_list)[diffidx]))


#%% Correlation matrix cluster

morph_dist_calyx_CM_avg = []
morph_dist_LH_CM_avg = []
morph_dist_AL_CM_avg = []

for i in range(len(morph_dist_calyx_CM)):
    morph_dist_calyx_CM_avg.append(np.average(morph_dist_calyx_CM[i], axis=0))
    morph_dist_LH_CM_avg.append(np.average(morph_dist_LH_CM[i], axis=0))
    morph_dist_AL_CM_avg.append(np.average(morph_dist_AL_CM[i], axis=0))

morph_dist_calyx_r_avg = scipy.spatial.distance.cdist(morph_dist_calyx_CM_avg, morph_dist_calyx_CM_avg)
morph_dist_LH_r_avg = scipy.spatial.distance.cdist(morph_dist_LH_CM_avg, morph_dist_LH_CM_avg)
morph_dist_AL_r_avg = scipy.spatial.distance.cdist(morph_dist_AL_CM_avg, morph_dist_AL_CM_avg)

morph_dist_calyx_r_df = pd.DataFrame(morph_dist_calyx_r)
morph_dist_calyx_r_avg_df = pd.DataFrame(morph_dist_calyx_r_avg)

morph_dist_LH_r_df = pd.DataFrame(morph_dist_LH_r)
morph_dist_LH_r_avg_df = pd.DataFrame(morph_dist_LH_r_avg)

morph_dist_AL_r_df = pd.DataFrame(morph_dist_AL_r)
morph_dist_AL_r_avg_df = pd.DataFrame(morph_dist_AL_r_avg)



L_AL = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_AL_r_avg), method='complete', optimal_ordering=True)

fig, ax = plt.subplots(figsize=(12, 4))
R_AL = scipy.cluster.hierarchy.dendrogram(L_AL,
                                       orientation='top',
                                       labels=glo_list,
                                       distance_sort='descending',
                                       show_leaf_counts=False)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
# plt.savefig(Parameter.outputdir + '/hier_AL.pdf', dpi=300, bbox_inches='tight')
plt.show()

ind_AL = scipy.cluster.hierarchy.fcluster(L_AL, 0.5*morph_dist_AL_r_avg.max(), 'maxclust')
columns_AL = R_AL['leaves']#[morph_dist_AL_r_avg_df.columns.tolist()[i] for i in list((np.argsort(ind_AL)))]


L_calyx = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_calyx_r_avg), method='complete', optimal_ordering=True)

fig, ax = plt.subplots(figsize=(12, 4))
R_calyx = scipy.cluster.hierarchy.dendrogram(L_calyx,
                                       orientation='top',
                                       labels=glo_list,
                                       distance_sort='descending',
                                       show_leaf_counts=False)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
# plt.savefig(Parameter.outputdir + '/hier_calyx.pdf', dpi=300, bbox_inches='tight')
plt.show()

ind_calyx = scipy.cluster.hierarchy.fcluster(L_calyx, 0.5*morph_dist_calyx_r_avg.max(), 'maxclust')
columns_calyx = R_calyx['leaves']#[morph_dist_calyx_r_avg_df.columns.tolist()[i] for i in list((np.argsort(ind_calyx)))]



L_LH = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_LH_r_avg), method='complete', optimal_ordering=True)

fig, ax = plt.subplots(figsize=(12, 4))
R_LH = scipy.cluster.hierarchy.dendrogram(L_LH,
                                       orientation='top',
                                       labels=glo_list,
                                       distance_sort='descending',
                                       show_leaf_counts=False)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
# plt.savefig(Parameter.outputdir + '/hier_LH.pdf', dpi=300, bbox_inches='tight')
plt.show()

ind_LH = scipy.cluster.hierarchy.fcluster(L_LH, 0.5*morph_dist_LH_r_avg.max(), 'maxclust')
columns_LH = R_LH['leaves']#[morph_dist_LH_r_avg_df.columns.tolist()[i] for i in list((np.argsort(ind_LH)))]



glo_list_cluster = np.array(glo_list)[columns_AL]
# glo_list_cluster = ['DL3', 'DA1', 'VM7d', 'VM7v', 'VC4', 'VM5v', 'VM5d', 'DM6', 'DM2', 'DM5', 
#   'DA2', 'DC1', 'DA4l', 'VC1', 'VA6', 'DC2', 'DC4', 'DL5', 'D', 'DL1', 'DA3', 'DA4m',
#   'DL4', 'VA1v', 'VA1d', 'DC3', 'VL2p', 'VL2a', 'VA7l', 'VA3', 'VA5', 'VA7m', 'VM1',
#   'VC3l', 'VC3m', 'VM4', 'VM6', 'VL1', 'V', 'DL2d', 'DL2v', 'VM2', 'VM3', 'DP1l', 'VA4',
#   'VC2', 'VA2', 'DP1m', 'DM3', 'DM4', 'DM1']

# columns = [glo_list.index(glo_list_cluster[i]) for i in range(len(glo_list_cluster))]

glo_len_cluster = np.array(glo_len)[columns_AL]

glo_idx_cluster = []
for i in range(len(glo_list)):
    glo_idx_cluster.append(glo_lb_idx[np.argwhere(np.array(glo_list)[columns_AL][i] == np.array(glo_list))[0][0]])

glo_idx_cluster_flat = [item for sublist in glo_idx_cluster for item in sublist]

glo_lb_cluster = [sum(glo_len_cluster[0:i]) for i in range(len(glo_len_cluster)+1)]
glo_lb_cluster_s = np.subtract(glo_lb_cluster, glo_lb_cluster[0])
glo_float_cluster = np.divide(glo_lb_cluster_s, glo_lb_cluster_s[-1])

morph_dist_calyx_r_df = morph_dist_calyx_r_df.reindex(glo_idx_cluster_flat, axis=0)
morph_dist_calyx_r_df = morph_dist_calyx_r_df.reindex(glo_idx_cluster_flat, axis=1)

morph_dist_LH_r_df = morph_dist_LH_r_df.reindex(glo_idx_cluster_flat, axis=0)
morph_dist_LH_r_df = morph_dist_LH_r_df.reindex(glo_idx_cluster_flat, axis=1)

morph_dist_AL_r_df = morph_dist_AL_r_df.reindex(glo_idx_cluster_flat, axis=0)
morph_dist_AL_r_df = morph_dist_AL_r_df.reindex(glo_idx_cluster_flat, axis=1)

fig = plt.figure(figsize=(6,6))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(morph_dist_calyx_r_df, cmap='viridis_r')#, vmax=np.max(morph_dist_calyx_r))
ax1.set_xticks([]) 
ax1.set_yticks([]) 
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(glo_float_cluster)
ax3.set_yticks(glo_float_cluster)
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster[1:] + glo_float_cluster[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster[1:] + glo_float_cluster[:-1])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=5, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=5, rotation_mode='default')
plt.colorbar(im, fraction=0.045)
# plt.title("Reorganized inter-cluster distance calyx", pad=40)
# plt.savefig(Parameter.outputdir + '/distance_grid_calyx.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(6,6))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(morph_dist_LH_r_df, cmap='viridis_r')#, vmax=np.max(morph_dist_LH_r))
ax1.set_xticks([]) 
ax1.set_yticks([]) 
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(glo_float_cluster)
ax3.set_yticks(glo_float_cluster)
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster[1:] + glo_float_cluster[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster[1:] + glo_float_cluster[:-1])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=5, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=5, rotation_mode='default')
plt.colorbar(im, fraction=0.045)
# plt.title("Reorganized inter-cluster distance LH", pad=40)
# plt.savefig(Parameter.outputdir + '/distance_grid_LH.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(6,6))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(morph_dist_AL_r_df, cmap='viridis_r')#, vmax=np.max(morph_dist_AL_r))
ax1.set_xticks([]) 
ax1.set_yticks([]) 
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(glo_float_cluster)
ax3.set_yticks(glo_float_cluster)
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster[1:] + glo_float_cluster[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster[1:] + glo_float_cluster[:-1])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=5, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=5, rotation_mode='default')
plt.colorbar(im, fraction=0.045)
# plt.title("Reorganized inter-cluster distance AL", pad=40)
# plt.savefig(Parameter.outputdir + '/distance_grid_AL.pdf', dpi=300, bbox_inches='tight')
plt.show()



fig, ax = plt.subplots(figsize=(12,4))
x = np.arange(len(glo_list))
width = 1.
ax.bar(x, np.array(ALcalyx_corr_glo_avg)[columns_AL], width, 
       yerr=np.array(ALcalyx_corr_glo_std)[columns_AL], label='Calyx-AL', alpha=0.5, 
       error_kw=dict(ecolor='tab:blue', lw=1, capsize=2, capthick=1))
ax.bar(x, np.array(ALLH_corr_glo_avg)[columns_AL], width, 
       yerr=np.array(ALLH_corr_glo_std)[columns_AL], label='LH-AL', alpha=0.5, 
       error_kw=dict(ecolor='tab:orange', lw=1, capsize=2, capthick=1))
ax.set_ylabel('Correlation Coefficient', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(glo_list_cluster, rotation=90, fontsize=10)
ax.legend(fontsize=15)
# ax.set_title('Distance correlation between calyx/LH and AL by glomerulus')
plt.xlim(0-0.5, len(glo_list)-0.5)
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/correlation_glomeruli_fix_1.pdf', dpi=300, bbox_inches='tight')
plt.show()



#%% Distance difference measure with new metric

fig, ax = plt.subplots(figsize=(6,6))
labels = ['MB calyx', 'LH', 'AL']
x = np.arange(len(labels))
width = .4

cmeans = [np.median(calyxdist_cluster_u_full_flat_new), np.median(LHdist_cluster_u_full_flat_new), np.median(ALdist_cluster_u_full_flat_new)]
cerr = [scipy.stats.median_abs_deviation(calyxdist_cluster_u_full_flat_new, center=np.median), 
        scipy.stats.median_abs_deviation(LHdist_cluster_u_full_flat_new, center=np.median), 
        scipy.stats.median_abs_deviation(ALdist_cluster_u_full_flat_new, center=np.median)]
ncmeans = [np.median(calyxdist_noncluster_u_full_flat_new), np.median(LHdist_noncluster_u_full_flat_new), np.median(ALdist_noncluster_u_full_flat_new)]
ncerr = [scipy.stats.median_abs_deviation(calyxdist_noncluster_u_full_flat_new, center=np.median), 
         scipy.stats.median_abs_deviation(LHdist_noncluster_u_full_flat_new, center=np.median), 
         scipy.stats.median_abs_deviation(ALdist_noncluster_u_full_flat_new, center=np.median)]

ax.bar(x - width/2, cmeans, width, yerr=cerr, capsize=5, label='Identical Glomerulus')
ax.bar(x + width/2, ncmeans, width, yerr=ncerr, capsize=5, label='Different Glomeruli')
ax.set_ylabel('Distance', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=17)
ax.tick_params(axis="y", labelsize=15)
ax.legend(fontsize=15)
#ax.set_title('Median distance within and outside cluster')
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/glomerulus_dist_diff_median_nnmetric_3.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(6,6))
labels = ['MB calyx', 'LH', 'AL']
x = np.arange(len(labels))
width = .4

cmeans = [np.mean(calyxdist_cluster_u_full_flat_new), np.mean(LHdist_cluster_u_full_flat_new), np.mean(ALdist_cluster_u_full_flat_new)]
cerr = [np.std(calyxdist_cluster_u_full_flat_new), 
        np.std(LHdist_cluster_u_full_flat_new), 
        np.std(ALdist_cluster_u_full_flat_new)]
ncmeans = [np.mean(calyxdist_noncluster_u_full_flat_new), np.mean(LHdist_noncluster_u_full_flat_new), np.mean(ALdist_noncluster_u_full_flat_new)]
ncerr = [np.std(calyxdist_noncluster_u_full_flat_new), 
         np.std(LHdist_noncluster_u_full_flat_new), 
         np.std(ALdist_noncluster_u_full_flat_new)]

ax.bar(x - width/2, cmeans, width, yerr=cerr, capsize=5, label='Identical Glomerulus')
ax.bar(x + width/2, ncmeans, width, yerr=ncerr, capsize=5, label='Different Glomeruli')
ax.set_ylabel('Distance', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=17)
ax.tick_params(axis="y", labelsize=15)
ax.legend(fontsize=15)
#ax.set_title('Median distance within and outside cluster')
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/glomerulus_dist_diff_average_nnmetric_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(8,6))
labels = ['MB calyx', 'LH', 'AL']
x = np.arange(len(labels))
width = .3

cmeans = [np.median(np.divide(calyxdist_cluster_u_full_flat_new, calyx_vol)), 
          np.median(np.divide(LHdist_cluster_u_full_flat_new, LH_vol)),
          np.median(np.divide(ALdist_cluster_u_full_flat_new, AL_vol))]
cerr = [scipy.stats.median_abs_deviation(np.divide(calyxdist_cluster_u_full_flat_new, calyx_vol), center=np.median), 
        scipy.stats.median_abs_deviation(np.divide(LHdist_cluster_u_full_flat_new, LH_vol), center=np.median), 
        scipy.stats.median_abs_deviation(np.divide(ALdist_cluster_u_full_flat_new, AL_vol), center=np.median)]
ncmeans = [np.median(np.divide(calyxdist_noncluster_u_full_flat_new, calyx_vol)),
           np.median(np.divide(LHdist_noncluster_u_full_flat_new, LH_vol)), 
           np.median(np.divide(ALdist_noncluster_u_full_flat_new, AL_vol))]
ncerr = [scipy.stats.median_abs_deviation(np.divide(calyxdist_noncluster_u_full_flat_new, calyx_vol), center=np.median), 
         scipy.stats.median_abs_deviation(np.divide(LHdist_noncluster_u_full_flat_new, LH_vol), center=np.median), 
         scipy.stats.median_abs_deviation(np.divide(ALdist_noncluster_u_full_flat_new, AL_vol), center=np.median)]

ax.bar(x - width/2, cmeans, width, yerr=cerr, capsize=5, label='Identical Glomerulus')
ax.bar(x + width/2, ncmeans, width, yerr=ncerr, capsize=5, label='Different Glomeruli')
ax.set_ylabel('Volume Corrected Distance', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=17)
ax.tick_params(axis="y", labelsize=15)
ax.legend(fontsize=15)
#ax.set_title('Median distance within and outside cluster')
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/glomerulus_dist_diff_median_nnmetric_volcorr_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(8,6))
labels = ['MB calyx', 'LH', 'AL']
x = np.arange(len(labels))
width = .3

cmeans = [np.mean(np.divide(calyxdist_cluster_u_full_flat_new, calyx_vol)), 
          np.mean(np.divide(LHdist_cluster_u_full_flat_new, LH_vol)), 
          np.mean(np.divide(ALdist_cluster_u_full_flat_new, AL_vol))]
cerr = [np.std(np.divide(calyxdist_cluster_u_full_flat_new, calyx_vol)), 
        np.std(np.divide(LHdist_cluster_u_full_flat_new, LH_vol)), 
        np.std(np.divide(ALdist_cluster_u_full_flat_new, AL_vol))]
ncmeans = [np.mean(np.divide(calyxdist_noncluster_u_full_flat_new, calyx_vol)), 
           np.mean(np.divide(LHdist_noncluster_u_full_flat_new, LH_vol)), 
           np.mean(np.divide(ALdist_noncluster_u_full_flat_new, AL_vol))]
ncerr = [np.std(np.divide(calyxdist_noncluster_u_full_flat_new, calyx_vol)), 
         np.std(np.divide(LHdist_noncluster_u_full_flat_new, LH_vol)), 
         np.std(np.divide(ALdist_noncluster_u_full_flat_new, AL_vol))]

ax.bar(x - width/2, cmeans, width, yerr=cerr, capsize=5, label='Identical Glomerulus')
ax.bar(x + width/2, ncmeans, width, yerr=ncerr, capsize=5, label='Different Glomeruli')
ax.set_ylabel('Volume Corrected Distance', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=17)
ax.tick_params(axis="y", labelsize=15)
ax.legend(fontsize=15)
#ax.set_title('Median distance within and outside cluster')
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/glomerulus_dist_diff_average_nnmetric_volcorr_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(5, 9))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.hist(ALdist_cluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
ax1.hist(ALdist_noncluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
# ax1.vlines(np.mean(ALdist_cluster_u_full_flat), 0, 0.12, color='tab:blue')
# ax1.vlines(np.mean(ALdist_noncluster_u_full_flat), 0, 0.12, color='tab:orange')
# ax1.vlines(np.median(ALdist_cluster_u_full_flat), 0, 0.12, ls='--', color='tab:blue')
# ax1.vlines(np.median(ALdist_noncluster_u_full_flat), 0, 0.12, ls='--', color='tab:orange')
ax1.set_ylim(0, 0.4)
ax1.set_ylabel('AL', fontsize=15)
ax1.legend(['Identical Glomerulus', 'Different Glomeruli'], fontsize=13)
ax2.hist(calyxdist_cluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
ax2.hist(calyxdist_noncluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
# ax2.vlines(np.mean(calyxdist_cluster_u_full_flat), 0, 0.22, color='tab:blue')
# ax2.vlines(np.mean(calyxdist_noncluster_u_full_flat), 0, 0.22, color='tab:orange')
# ax2.vlines(np.median(calyxdist_cluster_u_full_flat), 0, 0.22, ls='--', color='tab:blue')
# ax2.vlines(np.median(calyxdist_noncluster_u_full_flat), 0, 0.22, ls='--', color='tab:orange')
ax2.set_ylim(0, 0.4)
ax2.set_ylabel('MB calyx', fontsize=15)
ax3.hist(LHdist_cluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
ax3.hist(LHdist_noncluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
# ax3.vlines(np.mean(LHdist_cluster_u_full_flat), 0, 0.2, color='tab:blue')
# ax3.vlines(np.mean(LHdist_noncluster_u_full_flat), 0, 0.2, color='tab:orange')
# ax3.vlines(np.median(LHdist_cluster_u_full_flat), 0, 0.2, ls='--', color='tab:blue')
# ax3.vlines(np.median(LHdist_noncluster_u_full_flat), 0, 0.2, ls='--', color='tab:orange')
ax3.set_ylim(0, 0.4)
ax3.set_ylabel('LH', fontsize=15)
ax3.set_xlabel('Distance', fontsize=15)
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/skewed_dist_new_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%

calyxtest_cl = []
calyxtest_ncl = []
for i in range(len(calyxdist_cluster_u_full_new)):
    calyxtest_cl.append(np.mean(calyxdist_cluster_u_full_new[i]))
for i in range(len(calyxdist_noncluster_u_full_new)):
    calyxtest_ncl.append(np.mean(calyxdist_noncluster_u_full_new[i]))
    
LHtest_cl = []
LHtest_ncl = []
for i in range(len(LHdist_cluster_u_full_new)):
    LHtest_cl.append(np.mean(LHdist_cluster_u_full_new[i]))
for i in range(len(LHdist_noncluster_u_full_new)):
    LHtest_ncl.append(np.mean(LHdist_noncluster_u_full_new[i]))

ALtest_cl = []
ALtest_ncl = []
for i in range(len(ALdist_cluster_u_full_new)):
    ALtest_cl.append(np.mean(ALdist_cluster_u_full_new[i]))
for i in range(len(ALdist_noncluster_u_full_new)):
    ALtest_ncl.append(np.mean(ALdist_noncluster_u_full_new[i]))
    
calyxtest_cl = np.nan_to_num(calyxtest_cl)
calyxtest_ncl = np.nan_to_num(calyxtest_ncl)
LHtest_cl = np.nan_to_num(LHtest_cl)
LHtest_ncl = np.nan_to_num(LHtest_ncl)
ALtest_cl = np.nan_to_num(ALtest_cl)
ALtest_ncl = np.nan_to_num(ALtest_ncl)

ALtest_idx = np.where(ALtest_cl != 0)[0]
LHtest_idx = np.where(LHtest_cl != 0)[0]
calyxtest_idx = np.where(calyxtest_cl != 0)[0]

ALtest_percent = (ALtest_ncl[ALtest_idx] - ALtest_cl[ALtest_idx])/ALtest_ncl[ALtest_idx]
LHtest_percent = (LHtest_ncl[LHtest_idx] - LHtest_cl[LHtest_idx])/LHtest_ncl[LHtest_idx]
calyxtest_percent = (calyxtest_ncl[calyxtest_idx] - calyxtest_cl[calyxtest_idx])/calyxtest_ncl[calyxtest_idx]

attavlist = ['tab:red', 'tab:green', 'k', 'tab:green', 'tab:green', 'tab:red', 
             'tab:red', 'k', 'tab:green', 'tab:red', 'k', 'k', 'k', 'k', 'tab:red', 
             'tab:green', 'tab:red', 'tab:green', 'k', 'tab:green', 'tab:green', 
             'tab:red', 'tab:red', 'tab:green', 'tab:red', 'tab:blue', 'tab:blue', 
             'k', 'tab:red', 'tab:red', 'tab:blue', 'tab:blue']

updatedxlabel = np.array(glo_list)[LHtest_idx][np.argsort(LHtest_percent)]

attavdict = {updatedxlabel[i]: attavlist[i] for i in range(len(updatedxlabel))} 

fig, ax = plt.subplots(3, 1, figsize=(12,8))
x = np.arange(len(calyxtest_idx))
width = .3

ax[0].bar(x - width/2, ALtest_cl[ALtest_idx[np.argsort(LHtest_percent)]], width, capsize=5, label='Identical Glomerulus')
ax[0].bar(x + width/2, ALtest_ncl[ALtest_idx[np.argsort(LHtest_percent)]], width, capsize=5, label='Different Glomeruli')
# ax[0].set_ylabel('Distance', fontsize=17)
ax[0].set_xticks(x)
ax[0].set_title('AL', fontsize=21)
ax[0].set_xticklabels([])
ax[0].set_yticks(np.array([0, 25, 50]))
ax[0].legend(fontsize=15)
ax[0].tick_params(axis="y", labelsize=15)
ax[0].set_xlim(x[0] - 1, x[-1] + 1)

ax[1].bar(x - width/2, calyxtest_cl[calyxtest_idx[np.argsort(LHtest_percent)]], width, capsize=5, label='Identical Glomerulus')
ax[1].bar(x + width/2, calyxtest_ncl[calyxtest_idx[np.argsort(LHtest_percent)]], width, capsize=5, label='Different Glomeruli')
ax[1].set_ylabel('Distance', fontsize=17)
ax[1].set_xticks(x)
ax[1].set_title('MB calyx', fontsize=21)
ax[1].set_xticklabels([])
ax[1].set_yticks(np.array([0, 5, 10, 15]))
ax[1].tick_params(axis="y", labelsize=15)
ax[1].set_xlim(x[0] - 1, x[-1] + 1)

ax[2].bar(x - width/2, LHtest_cl[LHtest_idx[np.argsort(LHtest_percent)]], width, capsize=5, label='Identical Glomerulus')
ax[2].bar(x + width/2, LHtest_ncl[LHtest_idx[np.argsort(LHtest_percent)]], width, capsize=5, label='Different Glomeruli')
# ax[2].set_ylabel('Distance', fontsize=17)
ax[2].set_xticks(x)
ax[2].set_title('LH', fontsize=21)
ax[2].set_xticklabels(updatedxlabel, rotation=90, fontsize=15)
ax[2].set_yticks(np.array([0, 10, 20]))
ax[2].tick_params(axis="y", labelsize=15)
ax[2].set_xlim(x[0] - 1, x[-1] + 1)
[i.set_color(attavdict[i.get_text()]) for i in ax[2].xaxis.get_ticklabels()]
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/glo_dist_diff_per_glo_all_3.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(3, 1, figsize=(12,8))
x = np.arange(len(calyxtest_idx))
width = .3

ax[0].bar(x, 
          np.divide(ALtest_cl[ALtest_idx[np.argsort(LHtest_percent)]], 
                    ALtest_ncl[ALtest_idx[np.argsort(LHtest_percent)]]),
          width, capsize=5, color='tab:orange')
# ax[0].set_ylabel('Distance', fontsize=17)
ax[0].set_xticks(x)
ax[0].set_title('AL', fontsize=21)
ax[0].set_xticklabels([])
ax[0].set_yticks(np.array([0, 0.5, 1, 1.5]))
# ax[0].legend(fontsize=15)
ax[0].tick_params(axis="y", labelsize=15)
ax[0].set_xlim(x[0] - 1, x[-1] + 1)

ax[1].bar(x, 
          np.divide(calyxtest_cl[calyxtest_idx[np.argsort(LHtest_percent)]],
                    calyxtest_ncl[calyxtest_idx[np.argsort(LHtest_percent)]]),
          width, capsize=5, color='tab:orange')
ax[1].set_ylabel('Intra/Inter Distance Ratio', fontsize=17)
ax[1].set_xticks(x)
ax[1].set_title('MB calyx', fontsize=21)
ax[1].set_xticklabels([])
ax[1].set_yticks(np.array([0, 0.5, 1, 1.5]))
ax[1].tick_params(axis="y", labelsize=15)
ax[1].set_xlim(x[0] - 1, x[-1] + 1)

ax[2].bar(x, 
          np.divide(LHtest_cl[LHtest_idx[np.argsort(LHtest_percent)]],
                    LHtest_ncl[LHtest_idx[np.argsort(LHtest_percent)]]),
          width, capsize=5, color='tab:orange')
# ax[2].set_ylabel('Distance', fontsize=17)
ax[2].set_xticks(x)
ax[2].set_title('LH', fontsize=21)
ax[2].set_xticklabels(updatedxlabel, rotation=90, fontsize=17)
ax[2].set_yticks(np.array([0, 0.5, 1, 1.5]))
ax[2].tick_params(axis="y", labelsize=15)
ax[2].set_xlim(x[0] - 1, x[-1] + 1)
[t.set_color(i) for (i,t) in zip(attavlist, ax[2].xaxis.get_ticklabels())]
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/glo_dist_diff_per_glo_all_3.pdf', dpi=300, bbox_inches='tight')
plt.show()


LH_glo_col = np.sort(np.array(glo_list)[LHtest_idx][np.where(LHtest_percent >= 0.75)[0]])
calyx_glo_col = np.sort(np.array(glo_list)[calyxtest_idx][np.where(calyxtest_percent >= 0.75)[0]])

print(LH_glo_col)
print(calyx_glo_col)

LH_glo_col_idx = []

for i in range(len(LH_glo_col)):
    LH_glo_col_idx.append(glo_list.index(LH_glo_col[i]))

print(np.array(glo_len)[LH_glo_col_idx])


#%% Correlation matrix cluster with new metric

ALcalyx_corr_new = []
ALLH_corr_new = []
LHcalyx_corr_new = []

for i in range(len(morph_dist_AL_r_new)):
    ALcalyx_corr_new.append(np.corrcoef(morph_dist_calyx_r_new[i], morph_dist_AL_r_new[i])[0][1])
    ALLH_corr_new.append(np.corrcoef(morph_dist_LH_r_new[i], morph_dist_AL_r_new[i])[0][1])
    LHcalyx_corr_new.append(np.corrcoef(morph_dist_calyx_r_new[i], morph_dist_LH_r_new[i])[0][1])

ALcalyx_corr_new_glo = []
ALLH_corr_new_glo = []
LHcalyx_corr_new_glo = []

for i in range(len(glo_lb)-1):
    ALcalyx_corr_new_glo.append(np.array(ALcalyx_corr_new)[np.arange(glo_lb[i],glo_lb[i+1])])
    ALLH_corr_new_glo.append(np.array(ALLH_corr_new)[np.arange(glo_lb[i],glo_lb[i+1])])
    LHcalyx_corr_new_glo.append(np.array(LHcalyx_corr_new)[np.arange(glo_lb[i],glo_lb[i+1])])

ALcalyx_corr_new_glo_avg = []
ALcalyx_corr_new_glo_std = []
ALLH_corr_new_glo_avg = []
ALLH_corr_new_glo_std = []
LHcalyx_corr_new_glo_avg = []
LHcalyx_corr_new_glo_std = []

for i in range(len(ALcalyx_corr_new_glo)):
    ALcalyx_corr_new_glo_avg.append(np.average(ALcalyx_corr_new_glo[i]))
    ALcalyx_corr_new_glo_std.append(np.std(ALcalyx_corr_new_glo[i]))
    ALLH_corr_new_glo_avg.append(np.average(ALLH_corr_new_glo[i]))
    ALLH_corr_new_glo_std.append(np.std(ALLH_corr_new_glo[i]))
    LHcalyx_corr_new_glo_avg.append(np.average(LHcalyx_corr_new_glo[i]))
    LHcalyx_corr_new_glo_std.append(np.std(LHcalyx_corr_new_glo[i]))

validx_new = np.argwhere(np.array(ALLH_corr_new_glo_avg) > 0.5).T[0]

diffidx_new = np.argwhere(np.subtract(ALLH_corr_new_glo_avg, ALcalyx_corr_new_glo_avg) > 0.5).T[0]

print(np.sort(np.array(glo_list)[validx_new]))
print(np.sort(np.array(glo_list)[diffidx_new]))


morph_dist_calyx_r_new_avg = np.empty((len(morph_dist_calyx_CM), len(morph_dist_calyx_CM)))
morph_dist_LH_r_new_avg = np.empty((len(morph_dist_calyx_CM), len(morph_dist_calyx_CM)))
morph_dist_AL_r_new_avg = np.empty((len(morph_dist_calyx_CM), len(morph_dist_calyx_CM)))

for i in range(len(morph_dist_calyx_CM)):
    for j in range(i, len(morph_dist_calyx_CM)):
        if i == j:
            morph_dist_calyx_r_new_avg[i][j] = 0
            morph_dist_LH_r_new_avg[i][j] = 0
            morph_dist_AL_r_new_avg[i][j] = 0
        else:
            morph_dist_calyx_r_new_avg[i][j] = np.average(morph_dist_calyx_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[j]:glo_lbs[j+1]])
            morph_dist_calyx_r_new_avg[j][i] = morph_dist_calyx_r_new_avg[i][j]
            morph_dist_LH_r_new_avg[i][j] = np.average(morph_dist_LH_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[j]:glo_lbs[j+1]])
            morph_dist_LH_r_new_avg[j][i] = morph_dist_LH_r_new_avg[i][j]
            morph_dist_AL_r_new_avg[i][j] = np.average(morph_dist_AL_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[j]:glo_lbs[j+1]])
            morph_dist_AL_r_new_avg[j][i] = morph_dist_AL_r_new_avg[i][j]

morph_dist_calyx_r_new_df = pd.DataFrame(morph_dist_calyx_r_new)
morph_dist_calyx_r_new_avg_df = pd.DataFrame(morph_dist_calyx_r_new_avg)

morph_dist_LH_r_new_df = pd.DataFrame(morph_dist_LH_r_new)
morph_dist_LH_r_new_avg_df = pd.DataFrame(morph_dist_LH_r_new_avg)

morph_dist_AL_r_new_df = pd.DataFrame(morph_dist_AL_r_new)
morph_dist_AL_r_new_avg_df = pd.DataFrame(morph_dist_AL_r_new_avg)

L_AL_new = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_AL_r_new_avg), method='complete', optimal_ordering=True)

fig, ax = plt.subplots(figsize=(15, 3))
R_AL_new = scipy.cluster.hierarchy.dendrogram(L_AL_new,
                                        orientation='top',
                                        labels=glo_list,
                                        distance_sort='descending',
                                        show_leaf_counts=False,
                                        leaf_font_size=15)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
# plt.savefig(Parameter.outputdir + '/hier_AL_fixed_nnmetric_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

ind_AL_new = scipy.cluster.hierarchy.fcluster(L_AL_new, 0.5*morph_dist_AL_r_new_avg.max(), 'maxclust')
columns_AL_new = R_AL_new['leaves']#[morph_dist_AL_r_new_avg_df.columns.tolist()[i] for i in list((np.argsort(ind_AL_new)))]


L_calyx_new = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_calyx_r_new_avg), method='complete', optimal_ordering=True)

fig, ax = plt.subplots(figsize=(15, 3))
R_calyx_new = scipy.cluster.hierarchy.dendrogram(L_calyx_new,
                                        orientation='top',
                                        labels=glo_list,
                                        distance_sort='descending',
                                        show_leaf_counts=False,
                                        leaf_font_size=10)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
# plt.savefig(Parameter.outputdir + '/hier_calyx_fixed_nnmetric_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

ind_calyx_new = scipy.cluster.hierarchy.fcluster(L_calyx_new, 0.5*morph_dist_calyx_r_new_avg.max(), 'maxclust')
columns_calyx_new = R_calyx_new['leaves']#[morph_dist_calyx_r_new_avg_df.columns.tolist()[i] for i in list((np.argsort(ind_calyx_new)))]



L_LH_new = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_LH_r_new_avg), method='complete', optimal_ordering=True)

fig, ax = plt.subplots(figsize=(15, 3))
R_LH_new = scipy.cluster.hierarchy.dendrogram(L_LH_new,
                                        orientation='top',
                                        labels=glo_list,
                                        distance_sort='descending',
                                        show_leaf_counts=False,
                                        leaf_font_size=10)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
# plt.savefig(Parameter.outputdir + '/hier_LH_fixed_nnmetric_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

ind_LH_new = scipy.cluster.hierarchy.fcluster(L_LH_new, 0.5*morph_dist_LH_r_new_avg.max(), 'maxclust')
columns_LH_new = R_LH_new['leaves']#[morph_dist_LH_r_new_avg_df.columns.tolist()[i] for i in list((np.argsort(ind_LH_new)))]



glo_list_cluster_new = np.array(glo_list)[columns_AL_new]

glo_len_cluster_new = np.array(glo_len)[columns_AL_new]

glo_idx_cluster_new = []
for i in range(len(glo_list)):
    glo_idx_cluster_new.append(glo_lb_idx[np.argwhere(np.array(glo_list)[columns_AL_new][i] == np.array(glo_list))[0][0]])

glo_idx_cluster_new_flat = [item for sublist in glo_idx_cluster_new for item in sublist]

glo_lb_cluster_new = [sum(glo_len_cluster_new[0:i]) for i in range(len(glo_len_cluster_new)+1)]
glo_lb_cluster_new_s = np.subtract(glo_lb_cluster_new, glo_lb_cluster_new[0])
glo_float_cluster_new = np.divide(glo_lb_cluster_new_s, glo_lb_cluster_new_s[-1])

morph_dist_calyx_r_new_df = morph_dist_calyx_r_new_df.reindex(glo_idx_cluster_new_flat, axis=0)
morph_dist_calyx_r_new_df = morph_dist_calyx_r_new_df.reindex(glo_idx_cluster_new_flat, axis=1)

morph_dist_LH_r_new_df = morph_dist_LH_r_new_df.reindex(glo_idx_cluster_new_flat, axis=0)
morph_dist_LH_r_new_df = morph_dist_LH_r_new_df.reindex(glo_idx_cluster_new_flat, axis=1)

morph_dist_AL_r_new_df = morph_dist_AL_r_new_df.reindex(glo_idx_cluster_new_flat, axis=0)
morph_dist_AL_r_new_df = morph_dist_AL_r_new_df.reindex(glo_idx_cluster_new_flat, axis=1)

fig = plt.figure(figsize=(6,6))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(morph_dist_calyx_r_new_df, cmap='viridis_r')#, vmax=np.max(morph_dist_calyx_r_new))
ax1.set_xticks([]) 
ax1.set_yticks([]) 
ax1.axis["left"].set_visible(False)
ax1.axis["bottom"].set_visible(False)
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(glo_float_cluster_new)
ax3.set_yticks(glo_float_cluster_new)
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster_new[1:] + glo_float_cluster_new[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster_new[1:] + glo_float_cluster_new[:-1])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster_new))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster_new))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=8, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=8, rotation_mode='default')
plt.colorbar(im, fraction=0.045)
# plt.savefig(Parameter.outputdir + '/distance_grid_calyx_fixed_nnmetric_clst_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(6,6))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(morph_dist_LH_r_new_df, cmap='viridis_r')#, vmax=np.max(morph_dist_LH_r_new))
ax1.set_xticks([])
ax1.set_yticks([])
ax1.axis["left"].set_visible(False)
ax1.axis["bottom"].set_visible(False)
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(glo_float_cluster_new)
ax3.set_yticks(glo_float_cluster_new)
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster_new[1:] + glo_float_cluster_new[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster_new[1:] + glo_float_cluster_new[:-1])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster_new))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster_new))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=8, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=8, rotation_mode='default')
plt.colorbar(im, fraction=0.045)
# plt.savefig(Parameter.outputdir + '/distance_grid_LH_fixed_nnmetric_clst_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(6,6))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(morph_dist_AL_r_new_df, cmap='viridis_r')#, vmax=np.max(morph_dist_AL_r_new))
ax1.set_xticks([])
ax1.set_yticks([])
ax1.axis["left"].set_visible(False)
ax1.axis["bottom"].set_visible(False)
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(glo_float_cluster_new)
ax3.set_yticks(glo_float_cluster_new)
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster_new[1:] + glo_float_cluster_new[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster_new[1:] + glo_float_cluster_new[:-1])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster_new))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster_new))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=8, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=8, rotation_mode='default')
plt.colorbar(im, fraction=0.045)
# plt.savefig(Parameter.outputdir + '/distance_grid_AL_fixed_nnmetric_clst_2.pdf', dpi=300, bbox_inches='tight')
plt.show()



fig, ax = plt.subplots(figsize=(8,6))
x = np.arange(len(glo_list))
width = 1.
ax.bar(x, np.array(ALcalyx_corr_new_glo_avg)[columns_AL_new], width, 
       yerr=np.array(ALcalyx_corr_new_glo_std)[columns_AL_new], label='Calyx-AL', alpha=0.5, 
       error_kw=dict(ecolor='tab:blue', lw=1, capsize=2, capthick=1))
ax.bar(x, np.array(ALLH_corr_new_glo_avg)[columns_AL_new], width, 
       yerr=np.array(ALLH_corr_new_glo_std)[columns_AL_new], label='LH-AL', alpha=0.5, 
       error_kw=dict(ecolor='tab:orange', lw=1, capsize=2, capthick=1))
ax.set_ylabel('Correlation Coefficient', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(glo_list_cluster_new, rotation=90, fontsize=10)
ax.legend(fontsize=13)
plt.xlim(0-0.5, len(glo_list)-0.5)
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/correlation_glomeruli_nnmetric_clst_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
x = np.arange(len(glo_list))
width = 1.
ax.bar(x, np.array(ALcalyx_corr_glo_avg)[columns_AL_new], width, 
       yerr=np.array(ALcalyx_corr_glo_std)[columns_AL_new], label='Calyx-AL', alpha=0.5, 
       error_kw=dict(ecolor='tab:blue', lw=1, capsize=2, capthick=1))
ax.bar(x, np.array(ALLH_corr_glo_avg)[columns_AL_new], width, 
       yerr=np.array(ALLH_corr_glo_std)[columns_AL_new], label='LH-AL', alpha=0.5, 
       error_kw=dict(ecolor='tab:orange', lw=1, capsize=2, capthick=1))
ax.set_ylabel('Correlation Coefficient', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(glo_list_cluster_new, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
ax.legend(fontsize=15)
# ax.set_title('Distance correlation between calyx/LH and AL by glomerulus')
plt.xlim(0-0.5, len(glo_list)-0.5)
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/correlation_glomeruli_fixed_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Plotting of LH glomeruli with high LH-AL correlations


hull_LH_1 = ConvexHull(np.array(morph_dist_LH_flat)[:,:2])
hull_LH_2 = ConvexHull(np.array(morph_dist_LH_flat)[:,[0,2]])
v1 = np.append(hull_LH_1.vertices, hull_LH_1.vertices[0])
v2 = np.append(hull_LH_2.vertices, hull_LH_2.vertices[0])

from scipy.stats import kde

nbins=100

morph_dist_LH_n_flat_0 = [item for sublist in morph_dist_LH[validx[0]] for item in sublist]
morph_dist_LH_n_flat_1 = [item for sublist in morph_dist_LH[validx[4]] for item in sublist]
morph_dist_LH_n_flat_2 = [item for sublist in morph_dist_LH[validx[2]] for item in sublist]
morph_dist_LH_n_flat_3 = [item for sublist in morph_dist_LH[validx[1]] for item in sublist]
morph_dist_LH_n_flat_4 = [item for sublist in morph_dist_LH[validx[3]] for item in sublist]

kdeLHdorsal_0 = kde.gaussian_kde([np.array(morph_dist_LH_n_flat_0)[:,0], np.array(morph_dist_LH_n_flat_0)[:,1]])
kdeLHant_0 = kde.gaussian_kde([np.array(morph_dist_LH_n_flat_0)[:,0], np.array(morph_dist_LH_n_flat_0)[:,2]])

kdeLHdorsal_1 = kde.gaussian_kde([np.array(morph_dist_LH_n_flat_1)[:,0], np.array(morph_dist_LH_n_flat_1)[:,1]])
kdeLHant_1 = kde.gaussian_kde([np.array(morph_dist_LH_n_flat_1)[:,0], np.array(morph_dist_LH_n_flat_1)[:,2]])

kdeLHdorsal_2 = kde.gaussian_kde([np.array(morph_dist_LH_n_flat_2)[:,0], np.array(morph_dist_LH_n_flat_2)[:,1]])
kdeLHant_2 = kde.gaussian_kde([np.array(morph_dist_LH_n_flat_2)[:,0], np.array(morph_dist_LH_n_flat_2)[:,2]])

kdeLHdorsal_3 = kde.gaussian_kde([np.array(morph_dist_LH_n_flat_3)[:,0], np.array(morph_dist_LH_n_flat_3)[:,1]])
kdeLHant_3 = kde.gaussian_kde([np.array(morph_dist_LH_n_flat_3)[:,0], np.array(morph_dist_LH_n_flat_3)[:,2]])

kdeLHdorsal_4 = kde.gaussian_kde([np.array(morph_dist_LH_n_flat_4)[:,0], np.array(morph_dist_LH_n_flat_4)[:,1]])
kdeLHant_4 = kde.gaussian_kde([np.array(morph_dist_LH_n_flat_4)[:,0], np.array(morph_dist_LH_n_flat_4)[:,2]])

xLHd, yLHd = np.mgrid[370:500:nbins*1j, 155:285:nbins*1j]
xLHa, yLHa = np.mgrid[370:500:nbins*1j, 90:210:nbins*1j]

zLHd0 = kdeLHdorsal_0(np.vstack([xLHd.flatten(), yLHd.flatten()]))
zLHa0 = kdeLHant_0(np.vstack([xLHa.flatten(), yLHa.flatten()]))

zLHd1 = kdeLHdorsal_1(np.vstack([xLHd.flatten(), yLHd.flatten()]))
zLHa1 = kdeLHant_1(np.vstack([xLHa.flatten(), yLHa.flatten()]))

zLHd2 = kdeLHdorsal_2(np.vstack([xLHd.flatten(), yLHd.flatten()]))
zLHa2 = kdeLHant_2(np.vstack([xLHa.flatten(), yLHa.flatten()]))

zLHd3 = kdeLHdorsal_3(np.vstack([xLHd.flatten(), yLHd.flatten()]))
zLHa3 = kdeLHant_3(np.vstack([xLHa.flatten(), yLHa.flatten()]))

zLHd4 = kdeLHdorsal_4(np.vstack([xLHd.flatten(), yLHd.flatten()]))
zLHa4 = kdeLHant_4(np.vstack([xLHa.flatten(), yLHa.flatten()]))


fig = plt.figure(figsize=(10, 20))
ax1 = fig.add_subplot(5,2,1)
ax2 = fig.add_subplot(5,2,2)

ax3 = fig.add_subplot(5,2,3)
ax4 = fig.add_subplot(5,2,4)

ax5 = fig.add_subplot(5,2,5)
ax6 = fig.add_subplot(5,2,6)

ax7 = fig.add_subplot(5,2,7)
ax8 = fig.add_subplot(5,2,8)

ax9 = fig.add_subplot(5,2,9)
ax10 = fig.add_subplot(5,2,10)


ax1.pcolormesh(xLHa, yLHd, zLHd0.reshape(xLHd.shape), cmap=plt.cm.jet)
ax2.pcolormesh(xLHa, yLHa, zLHa0.reshape(xLHa.shape), cmap=plt.cm.jet)
ax1.set_xlim(380, 490)
ax1.set_ylim(165, 275)
ax2.set_xlim(380, 490)
ax2.set_ylim(100, 200)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_title("Dorsal", fontsize=20)
ax1.set_ylabel(np.array(glo_list)[validx[0]], fontsize=20)
ax2.set_title("Anterior", fontsize=20)
ax1.plot(np.array(morph_dist_LH_flat)[:,0][v1], 
         np.array(morph_dist_LH_flat)[:,1][v1],
         color='white',
         lw=3)
ax2.plot(np.array(morph_dist_LH_flat)[:,0][v2], 
         np.array(morph_dist_LH_flat)[:,2][v2],
         color='white',
         lw=3)
ax1.plot([387, 397], [184, 184], 'w-', lw=2)
ax1.plot([392, 392], [178, 190], 'w-', lw=2)
ax1.plot([387, 402], [260, 260], 'k-', lw=5)
ax1.plot([387, 402], [260, 260], 'w--', dashes=(2, 2), lw=5)
ax1.text(391, 195, 'A', c='w')
ax1.text(391, 176, 'P', c='w')
ax1.text(384, 186, 'L', c='w')
ax1.text(398, 186, 'M', c='w')
ax2.plot([387, 397], [183, 183], 'w-', lw=2)
ax2.plot([392, 392], [178, 188], 'w-', lw=2)
ax2.plot([387, 402], [115, 115], 'k-', lw=5)
ax2.plot([387, 402], [115, 115], 'w--', dashes=(2, 2), lw=5)
ax2.text(391, 190, 'D', c='w')
ax2.text(391, 173, 'V', c='w')
ax2.text(384, 181, 'L', c='w')
ax2.text(398, 181, 'M', c='w')
ax1.invert_yaxis()


ax3.pcolormesh(xLHa, yLHd, zLHd1.reshape(xLHd.shape), cmap=plt.cm.jet)
ax4.pcolormesh(xLHa, yLHa, zLHa1.reshape(xLHa.shape), cmap=plt.cm.jet)
ax3.set_xlim(380, 490)
ax3.set_ylim(165, 275)
ax4.set_xlim(380, 490)
ax4.set_ylim(100, 200)
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax3.set_ylabel(np.array(glo_list)[validx[4]], fontsize=20)
ax3.plot(np.array(morph_dist_LH_flat)[:,0][v1], 
         np.array(morph_dist_LH_flat)[:,1][v1],
         color='white',
         lw=3)
ax4.plot(np.array(morph_dist_LH_flat)[:,0][v2], 
         np.array(morph_dist_LH_flat)[:,2][v2],
         color='white',
         lw=3)
ax3.invert_yaxis()

ax5.pcolormesh(xLHa, yLHd, zLHd2.reshape(xLHd.shape), cmap=plt.cm.jet)
ax6.pcolormesh(xLHa, yLHa, zLHa2.reshape(xLHa.shape), cmap=plt.cm.jet)
ax5.set_xlim(380, 490)
ax5.set_ylim(165, 275)
ax6.set_xlim(380, 490)
ax6.set_ylim(100, 200)
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
ax5.set_ylabel(np.array(glo_list)[validx[2]], fontsize=20)
ax5.plot(np.array(morph_dist_LH_flat)[:,0][v1], 
         np.array(morph_dist_LH_flat)[:,1][v1],
         color='white',
         lw=3)
ax6.plot(np.array(morph_dist_LH_flat)[:,0][v2], 
         np.array(morph_dist_LH_flat)[:,2][v2],
         color='white',
         lw=3)
ax5.invert_yaxis()

ax7.pcolormesh(xLHa, yLHd, zLHd3.reshape(xLHd.shape), cmap=plt.cm.jet)
ax8.pcolormesh(xLHa, yLHa, zLHa3.reshape(xLHa.shape), cmap=plt.cm.jet)
ax7.set_xlim(380, 490)
ax7.set_ylim(165, 275)
ax8.set_xlim(380, 490)
ax8.set_ylim(100, 200)
ax7.set_xticks([])
ax7.set_yticks([])
ax8.set_xticks([])
ax8.set_yticks([])
ax7.set_ylabel(np.array(glo_list)[validx[1]], fontsize=20)
ax7.plot(np.array(morph_dist_LH_flat)[:,0][v1], 
         np.array(morph_dist_LH_flat)[:,1][v1],
         color='white',
         lw=3)
ax8.plot(np.array(morph_dist_LH_flat)[:,0][v2], 
         np.array(morph_dist_LH_flat)[:,2][v2],
         color='white',
         lw=3)
ax7.invert_yaxis()

ax9.pcolormesh(xLHa, yLHd, zLHd4.reshape(xLHd.shape), cmap=plt.cm.jet)
ax10.pcolormesh(xLHa, yLHa, zLHa4.reshape(xLHa.shape), cmap=plt.cm.jet)
ax9.set_xlim(380, 490)
ax9.set_ylim(165, 275)
ax10.set_xlim(380, 490)
ax10.set_ylim(100, 200)
ax9.set_xticks([])
ax9.set_yticks([])
ax10.set_xticks([])
ax10.set_yticks([])
ax9.set_ylabel(np.array(glo_list)[validx[3]], fontsize=20)
ax9.plot(np.array(morph_dist_LH_flat)[:,0][v1], 
         np.array(morph_dist_LH_flat)[:,1][v1],
         color='white',
         lw=3)
ax10.plot(np.array(morph_dist_LH_flat)[:,0][v2], 
         np.array(morph_dist_LH_flat)[:,2][v2],
         color='white',
         lw=3)
ax9.invert_yaxis()

# fig.suptitle(str(glo_list[gi]), fontsize=30)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig(Parameter.outputdir + '/heatmap_glo_high_corr_LH', dpi=300, bbox_inches='tight')
plt.show()

#%% Plotting of AL glomeruli with high LH-AL correlations


hull_AL_1 = ConvexHull(np.array(morph_dist_AL_flat)[:,:2])
hull_AL_2 = ConvexHull(np.array(morph_dist_AL_flat)[:,[0,2]])
v1 = np.append(hull_AL_1.vertices, hull_AL_1.vertices[0])
v2 = np.append(hull_AL_2.vertices, hull_AL_2.vertices[0])

from scipy.stats import kde

nbins=100

morph_dist_AL_n_flat_0 = [item for sublist in morph_dist_AL[validx[0]] for item in sublist]
morph_dist_AL_n_flat_1 = [item for sublist in morph_dist_AL[validx[4]] for item in sublist]
morph_dist_AL_n_flat_2 = [item for sublist in morph_dist_AL[validx[2]] for item in sublist]
morph_dist_AL_n_flat_3 = [item for sublist in morph_dist_AL[validx[1]] for item in sublist]
morph_dist_AL_n_flat_4 = [item for sublist in morph_dist_AL[validx[3]] for item in sublist]

kdeALdorsal_0 = kde.gaussian_kde([np.array(morph_dist_AL_n_flat_0)[:,0], np.array(morph_dist_AL_n_flat_0)[:,1]])
kdeALant_0 = kde.gaussian_kde([np.array(morph_dist_AL_n_flat_0)[:,0], np.array(morph_dist_AL_n_flat_0)[:,2]])

kdeALdorsal_1 = kde.gaussian_kde([np.array(morph_dist_AL_n_flat_1)[:,0], np.array(morph_dist_AL_n_flat_1)[:,1]])
kdeALant_1 = kde.gaussian_kde([np.array(morph_dist_AL_n_flat_1)[:,0], np.array(morph_dist_AL_n_flat_1)[:,2]])

kdeALdorsal_2 = kde.gaussian_kde([np.array(morph_dist_AL_n_flat_2)[:,0], np.array(morph_dist_AL_n_flat_2)[:,1]])
kdeALant_2 = kde.gaussian_kde([np.array(morph_dist_AL_n_flat_2)[:,0], np.array(morph_dist_AL_n_flat_2)[:,2]])

kdeALdorsal_3 = kde.gaussian_kde([np.array(morph_dist_AL_n_flat_3)[:,0], np.array(morph_dist_AL_n_flat_3)[:,1]])
kdeALant_3 = kde.gaussian_kde([np.array(morph_dist_AL_n_flat_3)[:,0], np.array(morph_dist_AL_n_flat_3)[:,2]])

kdeALdorsal_4 = kde.gaussian_kde([np.array(morph_dist_AL_n_flat_4)[:,0], np.array(morph_dist_AL_n_flat_4)[:,1]])
kdeALant_4 = kde.gaussian_kde([np.array(morph_dist_AL_n_flat_4)[:,0], np.array(morph_dist_AL_n_flat_4)[:,2]])

xALd, yALd = np.mgrid[455:625:nbins*1j, 240:410:nbins*1j]
xALa, yALa = np.mgrid[455:625:nbins*1j, -30:140:nbins*1j]

zALd0 = kdeALdorsal_0(np.vstack([xALd.flatten(), yALd.flatten()]))
zALa0 = kdeALant_0(np.vstack([xALa.flatten(), yALa.flatten()]))

zALd1 = kdeALdorsal_1(np.vstack([xALd.flatten(), yALd.flatten()]))
zALa1 = kdeALant_1(np.vstack([xALa.flatten(), yALa.flatten()]))

zALd2 = kdeALdorsal_2(np.vstack([xALd.flatten(), yALd.flatten()]))
zALa2 = kdeALant_2(np.vstack([xALa.flatten(), yALa.flatten()]))

zALd3 = kdeALdorsal_3(np.vstack([xALd.flatten(), yALd.flatten()]))
zALa3 = kdeALant_3(np.vstack([xALa.flatten(), yALa.flatten()]))

zALd4 = kdeALdorsal_4(np.vstack([xALd.flatten(), yALd.flatten()]))
zALa4 = kdeALant_4(np.vstack([xALa.flatten(), yALa.flatten()]))


fig = plt.figure(figsize=(10, 20))
ax1 = fig.add_subplot(5,2,1)
ax2 = fig.add_subplot(5,2,2)

ax3 = fig.add_subplot(5,2,3)
ax4 = fig.add_subplot(5,2,4)

ax5 = fig.add_subplot(5,2,5)
ax6 = fig.add_subplot(5,2,6)

ax7 = fig.add_subplot(5,2,7)
ax8 = fig.add_subplot(5,2,8)

ax9 = fig.add_subplot(5,2,9)
ax10 = fig.add_subplot(5,2,10)


ax1.pcolormesh(xALa, yALd, zALd0.reshape(xALd.shape), cmap=plt.cm.jet)
ax2.pcolormesh(xALa, yALa, zALa0.reshape(xALa.shape), cmap=plt.cm.jet)
ax1.set_xlim(465, 615)
ax1.set_ylim(250, 400)
ax2.set_xlim(465, 615)
ax2.set_ylim(-20, 130)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_title("Dorsal", fontsize=20)
ax1.set_ylabel(np.array(glo_list)[validx[0]], fontsize=20)
ax2.set_title("Anterior", fontsize=20)
ax1.plot(np.array(morph_dist_AL_flat)[:,0][v1], 
         np.array(morph_dist_AL_flat)[:,1][v1],
         color='white',
         lw=3)
ax2.plot(np.array(morph_dist_AL_flat)[:,0][v2], 
         np.array(morph_dist_AL_flat)[:,2][v2],
         color='white',
         lw=3)
ax1.plot([471.5, 486.5], [270, 270], 'w-', lw=2)
ax1.plot([479, 479], [262.5, 277.5], 'w-', lw=2)
ax1.plot([471.5, 486.5], [380, 380], 'k-', lw=5)
ax1.plot([471.5, 486.5], [380, 380], 'w--', dashes=(2, 2), lw=5)
ax1.text(477.5, 260.5, 'P', c='w')
ax1.text(477.5, 284.5, 'A', c='w')
ax1.text(467.5, 272, 'L', c='w')
ax1.text(487.5, 272, 'M', c='w')
ax2.plot([471.5, 486.5], [110, 110], 'w-', lw=2)
ax2.plot([479, 479], [101.5, 118.5], 'w-', lw=2)
ax2.plot([471.5, 486.5], [0, 0], 'k-', lw=5)
ax2.plot([471.5, 486.5], [0, 0], 'w--', dashes=(2, 2), lw=5)
ax2.text(477.5, 120.5, 'D', c='w')
ax2.text(477.5, 95.5, 'V', c='w')
ax2.text(467.5, 108, 'L', c='w')
ax2.text(487.5, 108, 'M', c='w')
ax1.invert_yaxis()

ax3.pcolormesh(xALa, yALd, zALd1.reshape(xALd.shape), cmap=plt.cm.jet)
ax4.pcolormesh(xALa, yALa, zALa1.reshape(xALa.shape), cmap=plt.cm.jet)
ax3.set_xlim(465, 615)
ax3.set_ylim(250, 400)
ax4.set_xlim(465, 615)
ax4.set_ylim(-20, 130)
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax3.set_ylabel(np.array(glo_list)[validx[4]], fontsize=20)
ax3.plot(np.array(morph_dist_AL_flat)[:,0][v1], 
         np.array(morph_dist_AL_flat)[:,1][v1],
         color='white',
         lw=3)
ax4.plot(np.array(morph_dist_AL_flat)[:,0][v2], 
         np.array(morph_dist_AL_flat)[:,2][v2],
         color='white',
         lw=3)
ax3.invert_yaxis()

ax5.pcolormesh(xALa, yALd, zALd2.reshape(xALd.shape), cmap=plt.cm.jet)
ax6.pcolormesh(xALa, yALa, zALa2.reshape(xALa.shape), cmap=plt.cm.jet)
ax5.set_xlim(465, 615)
ax5.set_ylim(250, 400)
ax6.set_xlim(465, 615)
ax6.set_ylim(-20, 130)
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
ax5.set_ylabel(np.array(glo_list)[validx[2]], fontsize=20)
ax5.plot(np.array(morph_dist_AL_flat)[:,0][v1], 
         np.array(morph_dist_AL_flat)[:,1][v1],
         color='white',
         lw=3)
ax6.plot(np.array(morph_dist_AL_flat)[:,0][v2], 
         np.array(morph_dist_AL_flat)[:,2][v2],
         color='white',
         lw=3)
ax5.invert_yaxis()

ax7.pcolormesh(xALa, yALd, zALd3.reshape(xALd.shape), cmap=plt.cm.jet)
ax8.pcolormesh(xALa, yALa, zALa3.reshape(xALa.shape), cmap=plt.cm.jet)
ax7.set_xlim(465, 615)
ax7.set_ylim(250, 400)
ax8.set_xlim(465, 615)
ax8.set_ylim(-20, 130)
ax7.set_xticks([])
ax7.set_yticks([])
ax8.set_xticks([])
ax8.set_yticks([])
ax7.set_ylabel(np.array(glo_list)[validx[1]], fontsize=20)
ax7.plot(np.array(morph_dist_AL_flat)[:,0][v1], 
         np.array(morph_dist_AL_flat)[:,1][v1],
         color='white',
         lw=3)
ax8.plot(np.array(morph_dist_AL_flat)[:,0][v2], 
         np.array(morph_dist_AL_flat)[:,2][v2],
         color='white',
         lw=3)
ax7.invert_yaxis()

ax9.pcolormesh(xALa, yALd, zALd4.reshape(xALd.shape), cmap=plt.cm.jet)
ax10.pcolormesh(xALa, yALa, zALa4.reshape(xALa.shape), cmap=plt.cm.jet)
ax9.set_xlim(465, 615)
ax9.set_ylim(250, 400)
ax10.set_xlim(465, 615)
ax10.set_ylim(-20, 130)
ax9.set_xticks([])
ax9.set_yticks([])
ax10.set_xticks([])
ax10.set_yticks([])
ax9.set_ylabel(np.array(glo_list)[validx[3]], fontsize=20)
ax9.plot(np.array(morph_dist_AL_flat)[:,0][v1], 
         np.array(morph_dist_AL_flat)[:,1][v1],
         color='white',
         lw=3)
ax10.plot(np.array(morph_dist_AL_flat)[:,0][v2], 
         np.array(morph_dist_AL_flat)[:,2][v2],
         color='white',
         lw=3)
ax9.invert_yaxis()

# fig.suptitle(str(glo_list[gi]), fontsize=30)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig(Parameter.outputdir + '/heatmap_glo_high_corr_AL', dpi=300, bbox_inches='tight')
plt.show()



#%% Correlation plot

morph_dist_LH_r_sel_avg_list = []
morph_dist_AL_r_sel_avg_list = []

for i in [0,4,2,1,3]:
    selidx = np.array(glo_lb_idx[validx[i]])
    morph_dist_LH_r_sel_avg_list.append(np.average(morph_dist_LH_r[selidx], axis=0))
    morph_dist_AL_r_sel_avg_list.append(np.average(morph_dist_AL_r[selidx], axis=0))

fig = plt.figure(figsize=(5, 10))
ax1 = fig.add_subplot(5,1,1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel(np.array(glo_list)[validx[0]], fontsize=15)
ax2 = fig.add_subplot(5,1,2)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_ylabel(np.array(glo_list)[validx[4]], fontsize=15)
ax3 = fig.add_subplot(5,1,3)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylabel(np.array(glo_list)[validx[2]], fontsize=15)
ax4 = fig.add_subplot(5,1,4)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_ylabel(np.array(glo_list)[validx[1]], fontsize=15)
ax5 = fig.add_subplot(5,1,5)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_ylabel(np.array(glo_list)[validx[3]], fontsize=15)
ax1.scatter(morph_dist_AL_r_sel_avg_list[0], morph_dist_LH_r_sel_avg_list[0])
ax2.scatter(morph_dist_AL_r_sel_avg_list[1], morph_dist_LH_r_sel_avg_list[1])
ax3.scatter(morph_dist_AL_r_sel_avg_list[2], morph_dist_LH_r_sel_avg_list[2])
ax4.scatter(morph_dist_AL_r_sel_avg_list[3], morph_dist_LH_r_sel_avg_list[3])
ax5.scatter(morph_dist_AL_r_sel_avg_list[4], morph_dist_LH_r_sel_avg_list[4])
plt.tight_layout()
# plt.savefig(Parameter.outputdir + '/high_corr_LH_fixed_1.pdf', dpi=300, bbox_inches='tight')
plt.show()




#%% Cluster quantification heatmap visualization for calyx

from scipy.stats import kde

nscale = 1

morph_dist_calyx_flat = [item for sublist in morph_dist_calyx for item in sublist]
morph_dist_calyx_flat = [item for sublist in morph_dist_calyx_flat for item in sublist]

mdcalyx_xmax = np.max(np.array(morph_dist_calyx_flat)[:,0])+5
mdcalyx_xmin = np.min(np.array(morph_dist_calyx_flat)[:,0])-5
mdcalyx_ymax = np.max(np.array(morph_dist_calyx_flat)[:,1])+5
mdcalyx_ymin = np.min(np.array(morph_dist_calyx_flat)[:,1])-5
mdcalyx_zmax = np.max(np.array(morph_dist_calyx_flat)[:,2])+5
mdcalyx_zmin = np.min(np.array(morph_dist_calyx_flat)[:,2])-5

nbinsx = int((mdcalyx_xmax-mdcalyx_xmin)/nscale)
nbinsy = int((mdcalyx_ymax-mdcalyx_ymin)/nscale)
nbinsz = int((mdcalyx_zmax-mdcalyx_zmin)/nscale)

t13 = time.time()

for i in range(len(morph_dist_calyx)):
    morph_dist_calyx_n_flat = [item for sublist in morph_dist_calyx[i] for item in sublist]
    
    x = np.array(morph_dist_calyx_n_flat)[:,0]
    y = np.array(morph_dist_calyx_n_flat)[:,1]
    z = np.array(morph_dist_calyx_n_flat)[:,2]
    
    xyz = np.vstack([x,y,z])
    kdecalyx = kde.gaussian_kde(xyz, bw_method=0.16)
    
    xi, yi, zi = np.mgrid[mdcalyx_xmin:mdcalyx_xmax:nbinsx*1j, 
                          mdcalyx_ymin:mdcalyx_ymax:nbinsy*1j, 
                          mdcalyx_zmin:mdcalyx_zmax:nbinsz*1j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
    density = kdecalyx(coords).reshape(xi.shape)
    
    np.save('./clusterdata/calyx_xi_' + str(i), xi)
    np.save('./clusterdata/calyx_yi_' + str(i), yi)
    np.save('./clusterdata/calyx_zi_' + str(i), zi)
    np.save('./clusterdata/calyx_d_' + str(i), density)
    

t14 = time.time()

print('checkpoint mayavi: ' + str(t14-t13))
    

#%% 3D plotting

from matplotlib import cm
import numpy as np
import logging
from mayavi import mlab

cmap1 = cm.get_cmap('Set1')
cmap2 = cm.get_cmap('Set2')
cmap3 = cm.get_cmap('Set3')
cmap4 = cm.get_cmap('tab20b')
cmap5 = cm.get_cmap('tab20c')

cmap = cmap1.colors + cmap4.colors + cmap5.colors + cmap2.colors + cmap3.colors
cmap = cm.get_cmap('jet', len(glo_list))

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 125, 0), (125, 255, 0), (0, 125, 255),
          (125, 0, 255), (255, 0, 125), (125, 125, 0), (125, 0, 125),
          (0, 125, 125), (125, 60, 0), (125, 0, 60), (60, 125, 0), (60, 0, 125),
          (0, 125, 60), (0, 60, 125), (60, 0, 0), (0, 60, 0), (0, 0, 60), 
          (255, 60, 0), (255, 0, 60), (60, 255, 0), (60, 0, 255), (0, 255, 60),
          (0, 60, 255), (190, 0, 0), (0, 190, 0), (0, 0, 190), (190, 60, 0),
          (190, 0, 60), (60, 190, 0), (60, 0, 190), (0, 190, 60), (0, 60, 190),
          (190, 125, 0), (190, 0, 125), (125, 190, 0), (125, 0, 190), (0, 190, 125),
          (0, 125, 190), (255, 190, 0), (255, 0, 190), (190, 255, 0), (190, 0, 255),
          (0, 255, 190), (0, 190, 255), (0, 0, 0), (120, 120, 120), (255, 255, 255)]
colors = np.divide(colors, 255)


figure = mlab.figure('DensityPlot', size=(1000,1000))

for i in range(len(glo_list)):
    
    xi = np.load('./clusterdata/calyx_xi_' + str(i) + '.npy')
    yi = np.load('./clusterdata/calyx_yi_' + str(i) + '.npy')
    zi = np.load('./clusterdata/calyx_zi_' + str(i) + '.npy')
    density = np.load('./clusterdata/calyx_d_' + str(i) + '.npy')
    
    mlab.contour3d(xi, yi, zi, density, color=tuple(colors[i]))
        
mlab.axes()
mlab.show()



#%% Cluster quantification heatmap visualization for LH

from scipy.stats import kde

nscale = 1

cmap1 = cm.get_cmap('Set1')
cmap2 = cm.get_cmap('Set2')
cmap3 = cm.get_cmap('Set3')
cmap4 = cm.get_cmap('tab20b')
cmap5 = cm.get_cmap('tab20c')

cmap = cmap1.colors + cmap4.colors + cmap5.colors + cmap2.colors + cmap3.colors

morph_dist_LH_flat = [item for sublist in morph_dist_LH for item in sublist]
morph_dist_LH_flat = [item for sublist in morph_dist_LH_flat for item in sublist]

mdLH_xmax = np.max(np.array(morph_dist_LH_flat)[:,0])+5
mdLH_xmin = np.min(np.array(morph_dist_LH_flat)[:,0])-5
mdLH_ymax = np.max(np.array(morph_dist_LH_flat)[:,1])+5
mdLH_ymin = np.min(np.array(morph_dist_LH_flat)[:,1])-5
mdLH_zmax = np.max(np.array(morph_dist_LH_flat)[:,2])+5
mdLH_zmin = np.min(np.array(morph_dist_LH_flat)[:,2])-5

nbinsx = int((mdLH_xmax-mdLH_xmin)/nscale)
nbinsy = int((mdLH_ymax-mdLH_ymin)/nscale)
nbinsz = int((mdLH_zmax-mdLH_zmin)/nscale)

t13 = time.time()

for i in range(len(morph_dist_calyx)):
    morph_dist_LH_n_flat = [item for sublist in morph_dist_LH[i] for item in sublist]
    
    x = np.array(morph_dist_LH_n_flat)[:,0]
    y = np.array(morph_dist_LH_n_flat)[:,1]
    z = np.array(morph_dist_LH_n_flat)[:,2]
    
    xyz = np.vstack([x,y,z])
    kdeLH = kde.gaussian_kde(xyz, bw_method=0.16)
    
    xi, yi, zi = np.mgrid[mdLH_xmin:mdLH_xmax:nbinsx*1j, 
                          mdLH_ymin:mdLH_ymax:nbinsy*1j, 
                          mdLH_zmin:mdLH_zmax:nbinsz*1j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
    density = kdeLH(coords).reshape(xi.shape)
    
    np.save('./clusterdata/LH_xi_' + str(i), xi)
    np.save('./clusterdata/LH_yi_' + str(i), yi)
    np.save('./clusterdata/LH_zi_' + str(i), zi)
    np.save('./clusterdata/LH_d_' + str(i), density)
    

t14 = time.time()

print('checkpoint mayavi: ' + str(t14-t13))


#%%

import logging
from mayavi import mlab
from colorsys import hls_to_rgb

cmap1 = cm.get_cmap('Set1')
cmap2 = cm.get_cmap('Set2')
cmap3 = cm.get_cmap('Set3')
cmap4 = cm.get_cmap('tab20b')
cmap5 = cm.get_cmap('tab20c')

cmap = cmap1.colors + cmap4.colors + cmap5.colors + cmap2.colors + cmap3.colors
cmap = cm.get_cmap('gist_rainbow', len(glo_list))
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 125, 0), (125, 255, 0), (0, 125, 255),
          (125, 0, 255), (255, 0, 125), (125, 125, 0), (125, 0, 125),
          (0, 125, 125), (125, 60, 0), (125, 0, 60), (60, 125, 0), (60, 0, 125),
          (0, 125, 60), (0, 60, 125), (60, 0, 0), (0, 60, 0), (0, 0, 60), 
          (255, 60, 0), (255, 0, 60), (60, 255, 0), (60, 0, 255), (0, 255, 60),
          (0, 60, 255), (190, 0, 0), (0, 190, 0), (0, 0, 190), (190, 60, 0),
          (190, 0, 60), (60, 190, 0), (60, 0, 190), (0, 190, 60), (0, 60, 190),
          (190, 125, 0), (190, 0, 125), (125, 190, 0), (125, 0, 190), (0, 190, 125),
          (0, 125, 190), (255, 190, 0), (255, 0, 190), (190, 255, 0), (190, 0, 255),
          (0, 255, 190), (0, 190, 255), (0, 0, 0), (120, 120, 120), (255, 255, 255)]
colors = np.divide(colors, 255)

figure = mlab.figure('DensityPlot', size=(1000,1000))

for i in range(len(glo_list)):
    xi = np.load('./clusterdata/LH_xi_' + str(i) + '.npy')
    yi = np.load('./clusterdata/LH_yi_' + str(i) + '.npy')
    zi = np.load('./clusterdata/LH_zi_' + str(i) + '.npy')
    density = np.load('./clusterdata/LH_d_' + str(i) + '.npy')
        
    mlab.contour3d(xi, yi, zi, density, color=tuple(colors[i]))

mlab.axes()
mlab.show()


#%% Cluster quantification heatmap visualization for AL

from scipy.stats import kde

nscale = 1

morph_dist_AL_flat = [item for sublist in morph_dist_AL for item in sublist]
morph_dist_AL_flat = [item for sublist in morph_dist_AL_flat for item in sublist]

mdAL_xmax = np.max(np.array(morph_dist_AL_flat)[:,0])+5
mdAL_xmin = np.min(np.array(morph_dist_AL_flat)[:,0])-5
mdAL_ymax = np.max(np.array(morph_dist_AL_flat)[:,1])+5
mdAL_ymin = np.min(np.array(morph_dist_AL_flat)[:,1])-5
mdAL_zmax = np.max(np.array(morph_dist_AL_flat)[:,2])+5
mdAL_zmin = np.min(np.array(morph_dist_AL_flat)[:,2])-5

nbinsx = int((mdAL_xmax-mdAL_xmin)/nscale)
nbinsy = int((mdAL_ymax-mdAL_ymin)/nscale)
nbinsz = int((mdAL_zmax-mdAL_zmin)/nscale)

t13 = time.time()

for i in range(len(morph_dist_AL)):
    morph_dist_AL_n_flat = [item for sublist in morph_dist_AL[i] for item in sublist]
    
    x = np.array(morph_dist_AL_n_flat)[:,0]
    y = np.array(morph_dist_AL_n_flat)[:,1]
    z = np.array(morph_dist_AL_n_flat)[:,2]
    
    xyz = np.vstack([x,y,z])
    kdeAL = kde.gaussian_kde(xyz, bw_method=0.16)
    
    xi, yi, zi = np.mgrid[mdAL_xmin:mdAL_xmax:nbinsx*1j, 
                          mdAL_ymin:mdAL_ymax:nbinsy*1j, 
                          mdAL_zmin:mdAL_zmax:nbinsz*1j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
    density = kdeAL(coords).reshape(xi.shape)
    
    np.save('./clusterdata/AL_xi_' + str(i), xi)
    np.save('./clusterdata/AL_yi_' + str(i), yi)
    np.save('./clusterdata/AL_zi_' + str(i), zi)
    np.save('./clusterdata/AL_d_' + str(i), density)
    

t14 = time.time()

print('checkpoint mayavi: ' + str(t14-t13))
    

#%%

from matplotlib import cm
import numpy as np
import logging
from mayavi import mlab

cmap1 = cm.get_cmap('Set1')
cmap2 = cm.get_cmap('Set2')
cmap3 = cm.get_cmap('Set3')
cmap4 = cm.get_cmap('tab20b')
cmap5 = cm.get_cmap('tab20c')

cmap = cmap1.colors + cmap4.colors + cmap5.colors + cmap2.colors + cmap3.colors
cmap = cm.get_cmap('jet', len(glo_list))

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 125, 0), (125, 255, 0), (0, 125, 255),
          (125, 0, 255), (255, 0, 125), (125, 125, 0), (125, 0, 125),
          (0, 125, 125), (125, 60, 0), (125, 0, 60), (60, 125, 0), (60, 0, 125),
          (0, 125, 60), (0, 60, 125), (60, 0, 0), (0, 60, 0), (0, 0, 60), 
          (255, 60, 0), (255, 0, 60), (60, 255, 0), (60, 0, 255), (0, 255, 60),
          (0, 60, 255), (190, 0, 0), (0, 190, 0), (0, 0, 190), (190, 60, 0),
          (190, 0, 60), (60, 190, 0), (60, 0, 190), (0, 190, 60), (0, 60, 190),
          (190, 125, 0), (190, 0, 125), (125, 190, 0), (125, 0, 190), (0, 190, 125),
          (0, 125, 190), (255, 190, 0), (255, 0, 190), (190, 255, 0), (190, 0, 255),
          (0, 255, 190), (0, 190, 255), (0, 0, 0), (120, 120, 120), (255, 255, 255)]
colors = np.divide(colors, 255)


figure = mlab.figure('DensityPlot', size=(1000,1000))

for i in range(len(glo_list)):
    
    xi = np.load('./clusterdata/AL_xi_' + str(i) + '.npy')
    yi = np.load('./clusterdata/AL_yi_' + str(i) + '.npy')
    zi = np.load('./clusterdata/AL_zi_' + str(i) + '.npy')
    density = np.load('./clusterdata/AL_d_' + str(i) + '.npy')
    
    mlab.contour3d(xi, yi, zi, density, color=tuple(colors[i]))
        
mlab.axes()
mlab.show()


#%% Calyx volume

import logging
from mayavi import mlab
from colorsys import hls_to_rgb

tri_calyx = []
for i in range(len(hull_calyx.simplices)):
    tt = []
    for j in range(len(hull_calyx.simplices[i])):
        tt.append(np.where(hull_calyx.vertices == hull_calyx.simplices[i][j])[0][0])
    tri_calyx.append(tuple(tt))
    
figure = mlab.figure('DensityPlot', size=(1000,1000))

mlab.triangular_mesh(np.array(morph_dist_calyx_flat)[hull_calyx.vertices,0], 
                      np.array(morph_dist_calyx_flat)[hull_calyx.vertices,1],
                      np.array(morph_dist_calyx_flat)[hull_calyx.vertices,2],
                      tri_calyx)
mlab.axes()
mlab.show()

#%% LH volume

import logging
from mayavi import mlab
from colorsys import hls_to_rgb

tri_LH = []
for i in range(len(hull_LH.simplices)):
    tt = []
    for j in range(len(hull_LH.simplices[i])):
        tt.append(np.where(hull_LH.vertices == hull_LH.simplices[i][j])[0][0])
    tri_LH.append(tuple(tt))
    
figure = mlab.figure('DensityPlot', size=(1000,1000))

mlab.triangular_mesh(np.array(morph_dist_LH_flat)[hull_LH.vertices,0], 
                      np.array(morph_dist_LH_flat)[hull_LH.vertices,1],
                      np.array(morph_dist_LH_flat)[hull_LH.vertices,2],
                      tri_LH)
mlab.axes()
mlab.show()


#%% AL volume

import logging
from mayavi import mlab
from colorsys import hls_to_rgb

tri_AL = []
for i in range(len(hull_AL.simplices)):
    tt = []
    for j in range(len(hull_AL.simplices[i])):
        tt.append(np.where(hull_AL.vertices == hull_AL.simplices[i][j])[0][0])
    tri_AL.append(tuple(tt))
    
figure = mlab.figure('DensityPlot', size=(1000,1000))

mlab.triangular_mesh(np.array(morph_dist_AL_flat)[hull_AL.vertices,0], 
                      np.array(morph_dist_AL_flat)[hull_AL.vertices,1],
                      np.array(morph_dist_AL_flat)[hull_AL.vertices,2],
                      tri_AL)
mlab.axes()
mlab.show()

#%%

morph_dist_calyx_ep_flat = [item for sublist in morph_dist_calyx_ep for item in sublist]
morph_dist_calyx_ep_flat = [item for sublist in morph_dist_calyx_ep_flat for item in sublist]

morph_dist_LH_ep_flat = [item for sublist in morph_dist_LH_ep for item in sublist]
morph_dist_LH_ep_flat = [item for sublist in morph_dist_LH_ep_flat for item in sublist]

calyx_mean = np.mean(morph_dist_calyx_flat, axis=0)
LH_mean = np.mean(morph_dist_LH_flat, axis=0)

calyx_std = np.std(morph_dist_calyx_flat, axis=0)
LH_std = np.std(morph_dist_LH_flat, axis=0)

morph_dist_calyx_ep_norm = []

for i in range(len(morph_dist_calyx_ep)):
    morph_dist_calyx_ep_t1 = []
    for j in range(len(morph_dist_calyx_ep[i])):
        morph_dist_calyx_ep_t2 = []
        for k in range(len(morph_dist_calyx_ep[i][j])):
            morph_dist_calyx_ep_t2.append([np.divide(np.array(morph_dist_calyx_ep[i][j][k])[0] - mdcalyx_xmin, mdcalyx_xmax),
                                           np.divide(np.array(morph_dist_calyx_ep[i][j][k])[1] - mdcalyx_ymin, mdcalyx_ymax),
                                           np.divide(np.array(morph_dist_calyx_ep[i][j][k])[2] - mdcalyx_zmin, mdcalyx_zmax)])
        morph_dist_calyx_ep_t1.append(np.array(morph_dist_calyx_ep_t2))
    morph_dist_calyx_ep_norm.append(morph_dist_calyx_ep_t1)

morph_dist_LH_ep_norm = []

for i in range(len(morph_dist_LH_ep)):
    morph_dist_LH_ep_t1 = []
    for j in range(len(morph_dist_LH_ep[i])):
        morph_dist_LH_ep_t2 = []
        for k in range(len(morph_dist_LH_ep[i][j])):
            morph_dist_LH_ep_t2.append([np.divide(np.array(morph_dist_LH_ep[i][j][k])[0] - mdLH_xmin, mdLH_xmax),
                                           np.divide(np.array(morph_dist_LH_ep[i][j][k])[1] - mdLH_ymin, mdLH_ymax),
                                           np.divide(np.array(morph_dist_LH_ep[i][j][k])[2] - mdLH_zmin, mdLH_zmax)])
        morph_dist_LH_ep_t1.append(np.array(morph_dist_LH_ep_t2))
    morph_dist_LH_ep_norm.append(morph_dist_LH_ep_t1)

morph_dist_calyx_ep_std = []
morph_dist_calyx_ep_mean = []
morph_dist_LH_ep_std = []
morph_dist_LH_ep_mean = []

for i in range(len(morph_dist_calyx_ep)):
    for j in range(len(morph_dist_calyx_ep[i])):
    # morph_dist_calyx_ep_g_flat = [item for sublist in morph_dist_calyx_ep[i] for item in sublist]
        morph_dist_calyx_ep_mean.append(np.mean(morph_dist_calyx_ep_norm[i][j], axis=0))
        morph_dist_calyx_ep_std.append(np.std(morph_dist_calyx_ep_norm[i][j], axis=0))

for i in range(len(morph_dist_LH_ep)):
    for j in range(len(morph_dist_LH_ep[i])):
        # morph_dist_LH_ep_g_flat = [item for sublist in morph_dist_LH_ep[i] for item in sublist]
        morph_dist_LH_ep_mean.append(np.mean(morph_dist_LH_ep_norm[i][j], axis=0))
        morph_dist_LH_ep_std.append(np.std(morph_dist_LH_ep_norm[i][j], axis=0))


#%%
            
# print(np.mean(np.divide(morph_dist_calyx_ep_std, calyx_std), axis=0))
# print(np.mean(np.divide(morph_dist_LH_ep_std, LH_std), axis=0))

fig = plt.figure(figsize=(24, 16))
ax = plt.axes(projection='3d')
ax.scatter3D(np.array(morph_dist_calyx_ep_mean)[:,0], 
             np.array(morph_dist_calyx_ep_mean)[:,1], 
             np.array(morph_dist_calyx_ep_mean)[:,2],
             color='tab:blue')
ax.scatter3D(np.array(morph_dist_LH_ep_mean)[:,0],
             np.array(morph_dist_LH_ep_mean)[:,1], 
             np.array(morph_dist_LH_ep_mean)[:,2], 
             color='tab:orange')
plt.show()

# plt.hist(np.array(morph_dist_calyx_ep_mean)[:,0], bins=10, density=True, alpha=0.5)
# plt.hist(np.array(morph_dist_calyx_ep_mean)[:,1], bins=10, density=True, alpha=0.5)
# plt.hist(np.array(morph_dist_calyx_ep_mean)[:,2], bins=10, density=True, alpha=0.5)
# plt.show()

plt.hist(np.array(morph_dist_calyx_ep_mean)[:,0], bins=10, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_mean)[:,0], bins=10, alpha=0.5, color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_mean)[:,1], bins=10, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_mean)[:,1], bins=10, alpha=0.5, color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_mean)[:,2], bins=10, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_mean)[:,2], bins=10, alpha=0.5, color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_std)[:,0], bins=10, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_std)[:,0], bins=10, alpha=0.5, color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_std)[:,1], bins=10, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_std)[:,1], bins=10, alpha=0.5, color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_std)[:,2], bins=10, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_std)[:,2], bins=10, alpha=0.5, color='tab:orange')
plt.show()

#%%

ornlist = ['TB', 'SB', 'LB', 'TB', 'unknown', 'LB', 'AC', 'PB', 'AC', 'AC', 'LB', 
           'TB', 'unknown', 'unknown', 'unknown', 'SB', 'PB', 'AC', 'SB', 'PB', 
           'SB', 'AC', 'unknown', 'SB', 'T1', 'TB', 'AC', 'LB', 'T2', 'T3', 'unknown', 
           'T3', 'T3', 'SB', 'unknown', 'TB', 'LB', 'PB', 'unknown', 'PB', 'AC', 
           'T3', 'SB', 'LB', 'unknown', 'SB', 'T2', 'AI', 'PB', 'unknown',
           'AC', 'unknown', 'T3']

senslist = []

for i in range(len(ornlist)):
    if ornlist[i] == 'LB' or ornlist[i] == 'TB' or ornlist[i] == 'SB':
        senslist.append('ab')
    elif ornlist[i] == 'T1' or ornlist[i] == 'T2' or ornlist[i] == 'T3':
        senslist.append('at')
    elif ornlist[i] == 'AC':
        senslist.append('ac')
    elif ornlist[i] == 'PB':
        senslist.append('pb')
    else:
        senslist.append('unknown')
        

#%% ORN based calyx

import logging
from mayavi import mlab
from colorsys import hls_to_rgb

cmap1 = cm.get_cmap('Set1')
cmap2 = cm.get_cmap('Set2')
cmap3 = cm.get_cmap('Set3')
cmap4 = cm.get_cmap('tab20b')
cmap5 = cm.get_cmap('tab20c')

cmap = cmap1.colors + cmap4.colors + cmap5.colors + cmap2.colors + cmap3.colors
cmap = cm.get_cmap('gist_rainbow', len(glo_list))
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 125, 0), (125, 255, 0), (0, 125, 255),
          (125, 0, 255), (255, 0, 125), (125, 125, 0), (125, 0, 125),
          (0, 125, 125), (125, 60, 0), (125, 0, 60), (60, 125, 0), (60, 0, 125),
          (0, 125, 60), (0, 60, 125), (60, 0, 0), (0, 60, 0), (0, 0, 60), 
          (255, 60, 0), (255, 0, 60), (60, 255, 0), (60, 0, 255), (0, 255, 60),
          (0, 60, 255), (190, 0, 0), (0, 190, 0), (0, 0, 190), (190, 60, 0),
          (190, 0, 60), (60, 190, 0), (60, 0, 190), (0, 190, 60), (0, 60, 190),
          (190, 125, 0), (190, 0, 125), (125, 190, 0), (125, 0, 190), (0, 190, 125),
          (0, 125, 190), (255, 190, 0), (255, 0, 190), (190, 255, 0), (190, 0, 255),
          (0, 255, 190), (0, 190, 255), (0, 0, 0), (120, 120, 120), (255, 255, 255)]
colors = np.divide(colors, 255)

figure = mlab.figure('DensityPlot')

for k in range(len(np.unique(ornlist))):
    kidx = np.where(np.array(ornlist) == np.unique(ornlist)[k])[0]
    for i in kidx:
        xi = np.load('./clusterdata/calyx_xi_' + str(i) + '.npy')
        yi = np.load('./clusterdata/calyx_yi_' + str(i) + '.npy')
        zi = np.load('./clusterdata/calyx_zi_' + str(i) + '.npy')
        density = np.load('./clusterdata/calyx_d_' + str(i) + '.npy')
        
        dmin = density.min()
        dmax = density.max()
        mlab.contour3d(xi, yi, zi, density, color=tuple(colors[k]))

mlab.axes()
mlab.show()


#%% ORN based LH

import logging
from mayavi import mlab
from colorsys import hls_to_rgb

cmap1 = cm.get_cmap('Set1')
cmap2 = cm.get_cmap('Set2')
cmap3 = cm.get_cmap('Set3')
cmap4 = cm.get_cmap('tab20b')
cmap5 = cm.get_cmap('tab20c')

cmap = cmap1.colors + cmap4.colors + cmap5.colors + cmap2.colors + cmap3.colors
cmap = cm.get_cmap('gist_rainbow', len(glo_list))
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 125, 0), (125, 255, 0), (0, 125, 255),
          (125, 0, 255), (255, 0, 125), (125, 125, 0), (125, 0, 125),
          (0, 125, 125), (125, 60, 0), (125, 0, 60), (60, 125, 0), (60, 0, 125),
          (0, 125, 60), (0, 60, 125), (60, 0, 0), (0, 60, 0), (0, 0, 60), 
          (255, 60, 0), (255, 0, 60), (60, 255, 0), (60, 0, 255), (0, 255, 60),
          (0, 60, 255), (190, 0, 0), (0, 190, 0), (0, 0, 190), (190, 60, 0),
          (190, 0, 60), (60, 190, 0), (60, 0, 190), (0, 190, 60), (0, 60, 190),
          (190, 125, 0), (190, 0, 125), (125, 190, 0), (125, 0, 190), (0, 190, 125),
          (0, 125, 190), (255, 190, 0), (255, 0, 190), (190, 255, 0), (190, 0, 255),
          (0, 255, 190), (0, 190, 255), (0, 0, 0), (120, 120, 120), (255, 255, 255)]
colors = np.divide(colors, 255)

figure = mlab.figure('DensityPlot')

for k in range(len(np.unique(ornlist))):
    kidx = np.where(np.array(ornlist) == np.unique(ornlist)[k])[0]
    for i in kidx:
        xi = np.load('./clusterdata/LH_xi_' + str(i) + '.npy')
        yi = np.load('./clusterdata/LH_yi_' + str(i) + '.npy')
        zi = np.load('./clusterdata/LH_zi_' + str(i) + '.npy')
        density = np.load('./clusterdata/LH_d_' + str(i) + '.npy')
        
        dmin = density.min()
        dmax = density.max()
        mlab.contour3d(xi, yi, zi, density, color=tuple(colors[k]))

mlab.axes()
mlab.show()


#%% Sensilary based calyx

import logging
from mayavi import mlab
from colorsys import hls_to_rgb

cmap1 = cm.get_cmap('Set1')
cmap2 = cm.get_cmap('Set2')
cmap3 = cm.get_cmap('Set3')
cmap4 = cm.get_cmap('tab20b')
cmap5 = cm.get_cmap('tab20c')

cmap = cmap1.colors + cmap4.colors + cmap5.colors + cmap2.colors + cmap3.colors
cmap = cm.get_cmap('gist_rainbow', len(glo_list))
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 125, 0), (125, 255, 0), (0, 125, 255),
          (125, 0, 255), (255, 0, 125), (125, 125, 0), (125, 0, 125),
          (0, 125, 125), (125, 60, 0), (125, 0, 60), (60, 125, 0), (60, 0, 125),
          (0, 125, 60), (0, 60, 125), (60, 0, 0), (0, 60, 0), (0, 0, 60), 
          (255, 60, 0), (255, 0, 60), (60, 255, 0), (60, 0, 255), (0, 255, 60),
          (0, 60, 255), (190, 0, 0), (0, 190, 0), (0, 0, 190), (190, 60, 0),
          (190, 0, 60), (60, 190, 0), (60, 0, 190), (0, 190, 60), (0, 60, 190),
          (190, 125, 0), (190, 0, 125), (125, 190, 0), (125, 0, 190), (0, 190, 125),
          (0, 125, 190), (255, 190, 0), (255, 0, 190), (190, 255, 0), (190, 0, 255),
          (0, 255, 190), (0, 190, 255), (0, 0, 0), (120, 120, 120), (255, 255, 255)]
colors = np.divide(colors, 255)

figure = mlab.figure('DensityPlot')

for k in range(len(np.unique(senslist))):
    kidx = np.where(np.array(senslist) == np.unique(senslist)[k])[0]
    for i in kidx:
        xi = np.load('./clusterdata/calyx_xi_' + str(i) + '.npy')
        yi = np.load('./clusterdata/calyx_yi_' + str(i) + '.npy')
        zi = np.load('./clusterdata/calyx_zi_' + str(i) + '.npy')
        density = np.load('./clusterdata/calyx_d_' + str(i) + '.npy')
        
        dmin = density.min()
        dmax = density.max()
        mlab.contour3d(xi, yi, zi, density, color=tuple(colors[k]))

mlab.axes()
mlab.show()

#%% Sensilary based LH

import logging
from mayavi import mlab
from colorsys import hls_to_rgb

cmap1 = cm.get_cmap('Set1')
cmap2 = cm.get_cmap('Set2')
cmap3 = cm.get_cmap('Set3')
cmap4 = cm.get_cmap('tab20b')
cmap5 = cm.get_cmap('tab20c')

cmap = cmap1.colors + cmap4.colors + cmap5.colors + cmap2.colors + cmap3.colors
cmap = cm.get_cmap('gist_rainbow', len(glo_list))
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 125, 0), (125, 255, 0), (0, 125, 255),
          (125, 0, 255), (255, 0, 125), (125, 125, 0), (125, 0, 125),
          (0, 125, 125), (125, 60, 0), (125, 0, 60), (60, 125, 0), (60, 0, 125),
          (0, 125, 60), (0, 60, 125), (60, 0, 0), (0, 60, 0), (0, 0, 60), 
          (255, 60, 0), (255, 0, 60), (60, 255, 0), (60, 0, 255), (0, 255, 60),
          (0, 60, 255), (190, 0, 0), (0, 190, 0), (0, 0, 190), (190, 60, 0),
          (190, 0, 60), (60, 190, 0), (60, 0, 190), (0, 190, 60), (0, 60, 190),
          (190, 125, 0), (190, 0, 125), (125, 190, 0), (125, 0, 190), (0, 190, 125),
          (0, 125, 190), (255, 190, 0), (255, 0, 190), (190, 255, 0), (190, 0, 255),
          (0, 255, 190), (0, 190, 255), (0, 0, 0), (120, 120, 120), (255, 255, 255)]
colors = np.divide(colors, 255)

figure = mlab.figure('DensityPlot')

for k in range(len(np.unique(senslist))):
    kidx = np.where(np.array(senslist) == np.unique(senslist)[k])[0]
    for i in kidx:
        xi = np.load('./clusterdata/LH_xi_' + str(i) + '.npy')
        yi = np.load('./clusterdata/LH_yi_' + str(i) + '.npy')
        zi = np.load('./clusterdata/LH_zi_' + str(i) + '.npy')
        density = np.load('./clusterdata/LH_d_' + str(i) + '.npy')
        
        dmin = density.min()
        dmax = density.max()
        mlab.contour3d(xi, yi, zi, density, color=tuple(colors[k]))

mlab.axes()
mlab.show()

#%% orn and sensilary based

morph_dist_calyx_ep_std = []
morph_dist_calyx_ep_mean = []
morph_dist_LH_ep_std = []
morph_dist_LH_ep_mean = []

for k in range(len(np.unique(ornlist))):
    morph_dist_calyx_ep_t = []
    kidx = np.where(np.array(ornlist) == np.unique(ornlist)[k])[0]
    for i in kidx:
        morph_dist_calyx_ep_t.append(morph_dist_calyx_ep_norm[i])
            
    morph_dist_calyx_ep_t = [item for sublist in morph_dist_calyx_ep_t for item in sublist]
    morph_dist_calyx_ep_t = [item for sublist in morph_dist_calyx_ep_t for item in sublist]
    morph_dist_calyx_ep_mean.append(np.mean(morph_dist_calyx_ep_t, axis=0))
    morph_dist_calyx_ep_std.append(np.std(morph_dist_calyx_ep_t, axis=0))


for k in range(len(np.unique(ornlist))):
    morph_dist_LH_ep_t = []
    kidx = np.where(np.array(ornlist) == np.unique(ornlist)[k])[0]
    for i in kidx:
        morph_dist_LH_ep_t.append(morph_dist_LH_ep_norm[i])
            
    morph_dist_LH_ep_t = [item for sublist in morph_dist_LH_ep_t for item in sublist]
    morph_dist_LH_ep_t = [item for sublist in morph_dist_LH_ep_t for item in sublist]
    morph_dist_LH_ep_mean.append(np.mean(morph_dist_LH_ep_t, axis=0))
    morph_dist_LH_ep_std.append(np.std(morph_dist_LH_ep_t, axis=0))

fig = plt.figure(figsize=(24, 16))
ax = plt.axes(projection='3d')
ax.scatter3D(np.array(morph_dist_calyx_ep_mean)[:,0], 
             np.array(morph_dist_calyx_ep_mean)[:,1], 
             np.array(morph_dist_calyx_ep_mean)[:,2], 
             color='tab:blue')
ax.scatter3D(np.array(morph_dist_LH_ep_mean)[:,0],
             np.array(morph_dist_LH_ep_mean)[:,1], 
             np.array(morph_dist_LH_ep_mean)[:,2],
             color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_mean)[:,0], bins=10, density=True, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_mean)[:,0], bins=10, density=True, alpha=0.5, color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_mean)[:,1], bins=10, density=True, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_mean)[:,1], bins=10, density=True, alpha=0.5, color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_mean)[:,2], bins=10, density=True, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_mean)[:,2], bins=10, density=True, alpha=0.5, color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_std)[:,0], bins=10, density=True, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_std)[:,0], bins=10, density=True, alpha=0.5, color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_std)[:,1], bins=10, density=True, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_std)[:,1], bins=10, density=True, alpha=0.5, color='tab:orange')
plt.show()

plt.hist(np.array(morph_dist_calyx_ep_std)[:,2], bins=10, density=True, alpha=0.5, color='tab:blue')
plt.hist(np.array(morph_dist_LH_ep_std)[:,2], bins=10, density=True, alpha=0.5, color='tab:orange')
plt.show()



#%% 2D heatmap of spatial distribution of each neuron in calyx, LH, and AL

from scipy.stats import kde

nbins=100
gi=validx[4]

morph_dist_calyx_n_flat = [item for sublist in morph_dist_calyx[gi] for item in sublist]

kdecalyxdorsal = kde.gaussian_kde([np.array(morph_dist_calyx_n_flat)[:,0], np.array(morph_dist_calyx_n_flat)[:,1]])
kdecalyxant = kde.gaussian_kde([np.array(morph_dist_calyx_n_flat)[:,0], np.array(morph_dist_calyx_n_flat)[:,2]])
xcalyxd, ycalyxd = np.mgrid[450:580:nbins*1j, 170:280:nbins*1j]
xcalyxa, ycalyxa = np.mgrid[450:580:nbins*1j, 120:230:nbins*1j]
zcalyxd = kdecalyxdorsal(np.vstack([xcalyxd.flatten(), ycalyxd.flatten()]))
zcalyxa = kdecalyxant(np.vstack([xcalyxa.flatten(), ycalyxa.flatten()]))

morph_dist_LH_n_flat = [item for sublist in morph_dist_LH[gi] for item in sublist]

kdeLHdorsal = kde.gaussian_kde([np.array(morph_dist_LH_n_flat)[:,0], np.array(morph_dist_LH_n_flat)[:,1]])
kdeLHant = kde.gaussian_kde([np.array(morph_dist_LH_n_flat)[:,0], np.array(morph_dist_LH_n_flat)[:,2]])
xLHd, yLHd = np.mgrid[370:520:nbins*1j, 160:280:nbins*1j]
xLHa, yLHa = np.mgrid[370:520:nbins*1j, 100:210:nbins*1j]
zLHd = kdeLHdorsal(np.vstack([xLHd.flatten(), yLHd.flatten()]))
zLHa = kdeLHant(np.vstack([xLHa.flatten(), yLHa.flatten()]))

morph_dist_AL_n_flat = [item for sublist in morph_dist_AL[gi] for item in sublist]

kdeALdorsal = kde.gaussian_kde([np.array(morph_dist_AL_n_flat)[:,0], np.array(morph_dist_AL_n_flat)[:,1]])
kdeALant = kde.gaussian_kde([np.array(morph_dist_AL_n_flat)[:,0], np.array(morph_dist_AL_n_flat)[:,2]])
xALd, yALd = np.mgrid[470:650:nbins*1j, 220:450:nbins*1j]
xALa, yALa = np.mgrid[470:650:nbins*1j, 0:200:nbins*1j]
zALd = kdeALdorsal(np.vstack([xALd.flatten(), yALd.flatten()]))
zALa = kdeALant(np.vstack([xALa.flatten(), yALa.flatten()]))


fig = plt.figure(figsize=(16, 18))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax1.pcolormesh(xcalyxd, ycalyxd, zcalyxd.reshape(xcalyxd.shape), cmap=plt.cm.jet)
ax2.pcolormesh(xcalyxa, ycalyxa, zcalyxa.reshape(xcalyxa.shape), cmap=plt.cm.jet)
ax1.set_xlim(450, 580)
ax1.set_ylim(170, 280)
ax2.set_xlim(450, 580)
ax2.set_ylim(120, 230)
ax1.set_title("Dorsal", fontsize=20)
ax1.set_ylabel("Calyx", fontsize=20)
ax2.set_title("Anterior", fontsize=20)

ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax3.pcolormesh(xLHd, yLHd, zLHd.reshape(xLHd.shape), cmap=plt.cm.jet)
ax4.pcolormesh(xLHa, yLHa, zLHa.reshape(xLHa.shape), cmap=plt.cm.jet)
ax3.set_xlim(370, 520)
ax3.set_ylim(160, 280)
ax4.set_xlim(370, 520)
ax4.set_ylim(100, 210)
ax3.set_ylabel("LH", fontsize=20)

ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)
ax5.pcolormesh(xALd, yALd, zALd.reshape(xALd.shape), cmap=plt.cm.jet)
ax6.pcolormesh(xALa, yALa, zALa.reshape(xALa.shape), cmap=plt.cm.jet)
ax5.set_xlim(470, 650)
ax5.set_ylim(220, 450)
ax6.set_xlim(470, 650)
ax6.set_ylim(0, 200)
ax5.set_ylabel("AL", fontsize=20)

fig.suptitle(str(glo_list[gi]), fontsize=30)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

t12 = time.time()

print('Run Time: ' + str(t12-t0))


#%% form factor per neuron

un_calyx = np.unique(MorphData.calyxdist_trk)
un_LH = np.unique(MorphData.LHdist_trk)
un_AL = np.unique(MorphData.ALdist_trk)

q_range = np.logspace(-2,3,100)

Pq_calyx = np.empty((len(q_range), len(un_calyx)))
Pq_LH = np.empty((len(q_range), len(un_LH)))
Pq_AL = np.empty((len(q_range), len(un_AL)))

t13 = time.time()

for q in range(len(q_range)):
    for i in range(len(un_calyx)):
        idx = np.where(MorphData.calyxdist_trk == un_calyx[i])[0]
        tarval = np.array(MorphData.calyxdist)[idx]
        calyxdist_per_n_flat_t = [item for sublist in tarval for item in sublist]
        calyxdist_per_n_flat_t = np.unique(calyxdist_per_n_flat_t, axis=0)
        qrvec = q_range[q]*scipy.spatial.distance.cdist(calyxdist_per_n_flat_t, calyxdist_per_n_flat_t)
        qrvec = qrvec[np.triu_indices_from(qrvec, k=1)]
        Pq_calyx[q][i] = np.divide(np.divide(2*np.sum(np.sin(qrvec)/qrvec), len(calyxdist_per_n_flat_t)), len(calyxdist_per_n_flat_t))

np.save(r'./Pq_calyx.npy', Pq_calyx)

for q in range(len(q_range)):
    for i in range(len(un_LH)):
        idx = np.where(MorphData.LHdist_trk == un_LH[i])[0]
        tarval = np.array(MorphData.LHdist)[idx]
        LHdist_per_n_flat_t = [item for sublist in tarval for item in sublist]
        LHdist_per_n_flat_t = np.unique(LHdist_per_n_flat_t, axis=0)
        qrvec = q_range[q]*scipy.spatial.distance.cdist(LHdist_per_n_flat_t, LHdist_per_n_flat_t)
        qrvec = qrvec[np.triu_indices_from(qrvec, k=1)]
        Pq_LH[q][i] = np.divide(np.divide(2*np.sum(np.sin(qrvec)/qrvec), len(LHdist_per_n_flat_t)), len(LHdist_per_n_flat_t))

np.save(r'./Pq_LH.npy', Pq_LH)

for q in range(len(q_range)):
    for i in range(len(un_AL)):
        idx = np.where(MorphData.ALdist_trk == un_AL[i])[0]
        tarval = np.array(MorphData.ALdist)[idx]
        ALdist_per_n_flat_t = [item for sublist in tarval for item in sublist]
        ALdist_per_n_flat_t = np.unique(ALdist_per_n_flat_t, axis=0)
        qrvec = q_range[q]*scipy.spatial.distance.cdist(ALdist_per_n_flat_t, ALdist_per_n_flat_t)
        qrvec = qrvec[np.triu_indices_from(qrvec, k=1)]
        Pq_AL[q][i] = np.divide(np.divide(2*np.sum(np.sin(qrvec)/qrvec), len(ALdist_per_n_flat_t)), len(ALdist_per_n_flat_t))

np.save(r'./Pq_AL.npy', Pq_AL)

print(time.time() - t13)

plt.plot(q_range, np.average(Pq_calyx, axis=1))
plt.xscale('log')
# plt.yscale('log')
plt.show()

plt.plot(q_range, np.average(Pq_LH, axis=1))
plt.xscale('log')
# plt.yscale('log')
plt.show()

plt.plot(q_range, np.average(Pq_AL, axis=1))
plt.xscale('log')
# plt.yscale('log')
plt.show()


#%% form factor per glomerulus

q_range = np.logspace(-2,3,100)

Pq_calyx_glo = np.empty((len(q_range), len(glo_idx)))
Pq_LH_glo = np.empty((len(q_range), len(glo_idx)))
Pq_AL_glo = np.empty((len(q_range), len(glo_idx)))

t13 = time.time()

for q in range(len(q_range)):
    for i in range(len(glo_idx)):
        morph_dist_calyx_flat = np.array([item for sublist in morph_dist_calyx[i] for item in sublist])
        morph_dist_calyx_flat = np.unique(morph_dist_calyx_flat, axis=0)
        qrvec = q_range[q]*scipy.spatial.distance.cdist(morph_dist_calyx_flat, morph_dist_calyx_flat)
        qrvec = qrvec[np.triu_indices_from(qrvec, k=1)]
        Pq_calyx_glo[q][i] = np.divide(np.divide(2*np.sum(np.sin(qrvec)/qrvec), len(morph_dist_calyx_flat)), len(morph_dist_calyx_flat))

np.save(r'./Pq_calyx_glo.npy', Pq_calyx_glo)

for q in range(len(q_range)):
    for i in range(len(glo_idx)):
        morph_dist_LH_flat = np.array([item for sublist in morph_dist_LH[i] for item in sublist])
        morph_dist_LH_flat = np.unique(morph_dist_LH_flat, axis=0)
        qrvec = q_range[q]*scipy.spatial.distance.cdist(morph_dist_LH_flat, morph_dist_LH_flat)
        qrvec = qrvec[np.triu_indices_from(qrvec, k=1)]
        Pq_LH_glo[q][i] = np.divide(np.divide(2*np.sum(np.sin(qrvec)/qrvec), len(morph_dist_LH_flat)), len(morph_dist_LH_flat))

np.save(r'./Pq_LH_glo.npy', Pq_LH_glo)

for q in range(len(q_range)):
    for i in range(len(glo_idx)):
        morph_dist_AL_flat = np.array([item for sublist in morph_dist_AL[i] for item in sublist])
        morph_dist_AL_flat = np.unique(morph_dist_AL_flat, axis=0)
        qrvec = q_range[q]*scipy.spatial.distance.cdist(morph_dist_AL_flat, morph_dist_AL_flat)
        qrvec = qrvec[np.triu_indices_from(qrvec, k=1)]
        Pq_AL_glo[q][i] = np.divide(np.divide(2*np.sum(np.sin(qrvec)/qrvec), len(morph_dist_AL_flat)), len(morph_dist_AL_flat))

np.save(r'./Pq_AL_glo.npy', Pq_AL_glo)

print(time.time() - t13)

plt.plot(q_range, np.average(Pq_calyx_glo, axis=1))
plt.xscale('log')
# plt.yscale('log')
plt.show()

plt.plot(q_range, np.average(Pq_LH_glo, axis=1))
plt.xscale('log')
# plt.yscale('log')
plt.show()

plt.plot(q_range, np.average(Pq_AL_glo, axis=1))
plt.xscale('log')
# plt.yscale('log')
plt.show()


#%% form factor per neuropil plotting

q_range = np.logspace(-2,3,100)

calyx_results = np.load(r'./calyx_results_debye.npy')
LH_results = np.load(r'./LH_results_debye.npy')
AL_results = np.load(r'./AL_results_debye.npy')

Pq_calyx = np.divide(np.sum(np.divide(np.array(calyx_results).reshape(100, len(calyx_dist_flat)), 
                                      len(calyx_dist_flat)), axis=1), len(calyx_dist_flat))
Pq_LH = np.divide(np.sum(np.divide(np.array(LH_results).reshape(100, len(LH_dist_flat)),
                                   len(LH_dist_flat)), axis=1), len(LH_dist_flat))
Pq_AL = np.divide(np.sum(np.divide(np.array(AL_results).reshape(100, len(AL_dist_flat)), 
                                   len(AL_dist_flat)), axis=1), len(AL_dist_flat))

d_Pq_calyx = np.gradient(np.log10(Pq_calyx[:60]), np.log10(q_range[:60]))
d_Pq_LH = np.gradient(np.log10(Pq_LH[:60]), np.log10(q_range[:60]))
d_Pq_AL = np.gradient(np.log10(Pq_AL[:60]), np.log10(q_range[:60]))

calyx_length_temp = np.array([item for sublist in LengthData.length_calyx for item in sublist])
LH_length_temp = np.array([item for sublist in LengthData.length_LH for item in sublist])
AL_length_temp = np.array([item for sublist in LengthData.length_AL for item in sublist])

rgy_calyx_full = utils.radiusOfGyration(np.array([calyx_dist_flat]))
rgy_LH_full = utils.radiusOfGyration(np.array([LH_dist_flat]))
rgy_AL_full = utils.radiusOfGyration(np.array([AL_dist_flat]))

q_range_fit_calyx = np.where(q_range > 1/np.mean(calyx_length_temp))[0]
q_range_fit_calyx = q_range_fit_calyx[q_range_fit_calyx <= 60]

poptD_Pq_calyx, pcovD_Pq_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                          np.log10(q_range[q_range_fit_calyx]), 
                                                          np.log10(Pq_calyx[q_range_fit_calyx]), 
                                                          p0=[-0.1, 0.1], 
                                                          maxfev=10000)
perrD_Pq_calyx = np.sqrt(np.diag(pcovD_Pq_calyx))

fitYD_Pq_calyx = objFuncPpow(q_range[q_range_fit_calyx], poptD_Pq_calyx[0], poptD_Pq_calyx[1])

q_range_fit_LH = np.where(q_range > 1/np.mean(LH_length_temp))[0]
q_range_fit_LH = q_range_fit_LH[q_range_fit_LH <= 60]

poptD_Pq_LH, pcovD_Pq_LH = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(q_range[q_range_fit_LH]), 
                                                    np.log10(Pq_LH[q_range_fit_LH]), 
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_Pq_LH = np.sqrt(np.diag(pcovD_Pq_LH))

fitYD_Pq_LH = objFuncPpow(q_range[q_range_fit_LH], poptD_Pq_LH[0], poptD_Pq_LH[1])

q_range_fit_AL = np.where(q_range > 1/np.mean(AL_length_temp))[0]
q_range_fit_AL = q_range_fit_AL[q_range_fit_AL <= 60]

poptD_Pq_AL, pcovD_Pq_AL = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(q_range[q_range_fit_AL]), 
                                                    np.log10(Pq_AL[q_range_fit_AL]), 
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_Pq_AL = np.sqrt(np.diag(pcovD_Pq_AL))

fitYD_Pq_AL = objFuncPpow(q_range[q_range_fit_AL], poptD_Pq_AL[0], poptD_Pq_AL[1])


calyx_q_idx = np.where(q_range < 2*np.pi/np.percentile(calyx_length_temp, 2))[0][-1]
LH_q_idx = np.where(q_range < 2*np.pi/np.percentile(LH_length_temp, 2))[0][-1]
AL_q_idx = np.where(q_range < 2*np.pi/np.percentile(AL_length_temp, 2))[0][-1]


fig = plt.figure(figsize=(8,6))
plt.plot(q_range[:AL_q_idx], Pq_AL[:AL_q_idx], marker='.', color='tab:blue')
plt.plot(q_range[:calyx_q_idx], Pq_calyx[:calyx_q_idx], marker='.', color='tab:orange')
plt.plot(q_range[:LH_q_idx], Pq_LH[:LH_q_idx], marker='.', color='tab:green')

plt.vlines(2*np.pi/np.mean(AL_length_temp), 1e-6, 10, color='tab:blue')
plt.vlines(2*np.pi/np.mean(calyx_length_temp), 1e-6, 10, color='tab:orange')
plt.vlines(2*np.pi/np.mean(LH_length_temp), 1e-6, 10, color='tab:green')

# plt.vlines(2*np.pi/np.median(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dotted')
# plt.vlines(2*np.pi/np.median(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dotted')
# plt.vlines(2*np.pi/np.median(LH_length_temp), 1e-6, 10, color='tab:green', ls='dotted')

plt.vlines(1/rgy_AL_full[0], 1e-6, 10, color='tab:blue', ls='--')
plt.vlines(1/rgy_calyx_full[0], 1e-6, 10, color='tab:orange', ls='--')
plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')

line1 = 1/100000*np.power(q_range, -16/7)
line2 = 1/10000000*np.power(q_range, -4/1)
line3 = 1/50000*np.power(q_range, -2/1)
line4 = 1/50000*np.power(q_range, -1)

plt.plot(q_range[28:36], line1[28:36], lw=2, color='k')
plt.plot(q_range[10:17], line2[10:17], lw=2, color='k')
plt.plot(q_range[19:27], line3[19:27], lw=2, color='k')
plt.plot(q_range[38:48], line4[38:48], lw=2, color='k')

plt.text(0.15, 4e-5, r'$\lambda = \dfrac{7}{16}$', fontsize=13)
plt.text(0.02, 9e-3, r'$\lambda = \dfrac{1}{4}$', fontsize=13)
plt.text(0.07, 4e-4, r'$\lambda = \dfrac{1}{2}$', fontsize=13)
plt.text(.8, 0.6e-5, r'$\lambda = 1$', fontsize=13)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
plt.ylim(1e-6, 10)
plt.legend(['AL', 'MB calyx', 'LH'], fontsize=13)
# plt.savefig(Parameter.outputdir + '/Pq_neuropil_3.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% form factor per neuropil moving window (FIGURE FORM FACTOR)

mw_Pq_calyx = []
mw_Pq_calyx_err = []
mwx_calyx = []
shiftN = 15

for i in range(len(q_range[:calyx_q_idx]) - shiftN):
    mwx_calyx.append(np.average(q_range[:calyx_q_idx][i:i+shiftN]))
    
    poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(q_range[:calyx_q_idx][i:i+shiftN]), 
                                                np.log10(Pq_calyx[:calyx_q_idx][i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    mw_Pq_calyx.append(poptmxc[0])
    mw_Pq_calyx_err.append(np.sqrt(np.diag(pcovmxc))[0])

mw_Pq_LH = []
mw_Pq_LH_err = []
mwx_LH = []

for i in range(len(q_range[:LH_q_idx]) - shiftN):
    mwx_LH.append(np.average(q_range[:LH_q_idx][i:i+shiftN]))
    
    poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(q_range[:LH_q_idx][i:i+shiftN]), 
                                                np.log10(Pq_LH[:LH_q_idx][i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    mw_Pq_LH.append(poptmxc[0])
    mw_Pq_LH_err.append(np.sqrt(np.diag(pcovmxc))[0])

mw_Pq_AL = []
mw_Pq_AL_err = []
mwx_AL = []

for i in range(len(q_range[:AL_q_idx]) - shiftN):
    mwx_AL.append(np.average(q_range[:AL_q_idx][i:i+shiftN]))
    
    poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(q_range[:AL_q_idx][i:i+shiftN]), 
                                                np.log10(Pq_AL[:AL_q_idx][i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    mw_Pq_AL.append(poptmxc[0])
    mw_Pq_AL_err.append(np.sqrt(np.diag(pcovmxc))[0])

    

fig = plt.figure(figsize=(6,4.5))
plt.plot(mwx_AL, -1/np.array(mw_Pq_AL), lw=2)
plt.plot(mwx_calyx, -1/np.array(mw_Pq_calyx), lw=2)
plt.plot(mwx_LH, -1/np.array(mw_Pq_LH), lw=2)
plt.fill_between(mwx_AL, 
                 -1/(np.array(mw_Pq_AL)-np.array(mw_Pq_AL_err)), 
                 -1/(np.array(mw_Pq_AL)+np.array(mw_Pq_AL_err)), 
                 alpha=0.3)
plt.fill_between(mwx_calyx, 
                 -1/(np.array(mw_Pq_calyx)-np.array(mw_Pq_calyx_err)),
                 -1/(np.array(mw_Pq_calyx)+np.array(mw_Pq_calyx_err)), 
                 alpha=0.3)
plt.fill_between(mwx_LH,
                 -1/(np.array(mw_Pq_LH)-np.array(mw_Pq_LH_err)),
                 -1/(np.array(mw_Pq_LH)+np.array(mw_Pq_LH_err)), 
                 alpha=0.3)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
plt.text(10.3, 1-0.03,'Linear', fontsize=14)
plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)

plt.vlines(2*np.pi/np.mean(AL_length_temp), 1e-6, 10, color='tab:blue')
plt.vlines(2*np.pi/np.mean(calyx_length_temp), 1e-6, 10, color='tab:orange')
plt.vlines(2*np.pi/np.mean(LH_length_temp), 1e-6, 10, color='tab:green')

# plt.vlines(2*np.pi/np.median(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dotted')
# plt.vlines(2*np.pi/np.median(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dotted')
# plt.vlines(2*np.pi/np.median(LH_length_temp), 1e-6, 10, color='tab:green', ls='dotted')

plt.vlines(1/rgy_AL_full[0], 1e-6, 10, color='tab:blue', ls='--')
plt.vlines(1/rgy_calyx_full[0], 1e-6, 10, color='tab:orange', ls='--')
plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')

plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.5)
plt.xlim(0.01, 10)

plt.legend(["AL", "MB calyx", "LH"], fontsize=14)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
plt.xticks(fontsize=14)
plt.ylabel(r"$\nu$", fontsize=17)
plt.yticks(fontsize=14)
# plt.savefig(Parameter.outputdir + '/Pq_all_mv_5.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Example calyx neuron skeletal plot for characteristic scales shown in form factor

nidx = 6
bidx = 0
scaleVal = [1, 2, 5, 10]
calyxcent = MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(BranchData.calyx_branchP[nidx][bidx])]#[510, 219, 171]#calyxCM

cmap = cm.get_cmap('viridis', len(MorphData.calyxdist))

for s in scaleVal:
    fig = plt.figure(figsize=(8, 8))
    for i in range(len(MorphData.calyxdist)):
        listOfPoints = MorphData.calyxdist[i]
        for f in range(len(listOfPoints)-1):
            if (((calyxcent[0] - s/2 <= listOfPoints[f][0] <= calyxcent[0] + s/2) or
                (calyxcent[0] - s/2 <= listOfPoints[f+1][0] <= calyxcent[0] + s/2)) and
                ((calyxcent[1] - s/2 <= listOfPoints[f][1] <= calyxcent[1] + s/2) or
                (calyxcent[1] - s/2 <= listOfPoints[f+1][1] <= calyxcent[1] + s/2)) and
                ((calyxcent[2] - s/2 <= listOfPoints[f][2] <= calyxcent[2] + s/2) or
                (calyxcent[2] - s/2 <= listOfPoints[f+1][2] <= calyxcent[2] + s/2))):
                morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
                plt.plot(morph_line[:,0], morph_line[:,1], color=cmap(i))
    
    plt.xlim(calyxcent[0] - s/2, calyxcent[0] + s/2)
    plt.ylim(calyxcent[1] - s/2, calyxcent[1] + s/2)
    plt.xticks([])
    plt.yticks([])
    
    # plt.savefig(os.path.join(Parameter.outputdir, 'Pq_all_char_calyx_' + str(s) + '.png'), dpi=300, bbox_inches='tight')
    plt.show()

#%% Example LH neuron skeletal plot for characteristic scales shown in form factor

nidx = 6
bidx = 3
scaleVal = [1, 2, 5, 10]
LHcent = MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(BranchData.LH_branchP[nidx][bidx])]#[427, 223 , 152]#LHCM

cmap = cm.get_cmap('viridis', len(MorphData.LHdist))

for s in scaleVal:
    fig = plt.figure(figsize=(8, 8))
    for i in range(len(MorphData.LHdist)):
        listOfPoints = MorphData.LHdist[i]
        for f in range(len(listOfPoints)-1):
            if (((LHcent[0] - s/2 <= listOfPoints[f][0] <= LHcent[0] + s/2) or
                (LHcent[0] - s/2 <= listOfPoints[f+1][0] <= LHcent[0] + s/2)) and
                ((LHcent[1] - s/2 <= listOfPoints[f][1] <= LHcent[1] + s/2) or
                (LHcent[1] - s/2 <= listOfPoints[f+1][1] <= LHcent[1] + s/2)) and
                ((LHcent[2] - s/2 <= listOfPoints[f][2] <= LHcent[2] + s/2) or
                (LHcent[2] - s/2 <= listOfPoints[f+1][2] <= LHcent[2] + s/2))):
                morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
                plt.plot(morph_line[:,0], morph_line[:,1], color=cmap(i))
    
    plt.xlim(LHcent[0] - s/2, LHcent[0] + s/2)
    plt.ylim(LHcent[1] - s/2, LHcent[1] + s/2)
    plt.xticks([])
    plt.yticks([])
    
    # plt.savefig(os.path.join(Parameter.outputdir, 'Pq_all_char_LH_' + str(s) + '.png'), dpi=300, bbox_inches='tight')
    plt.show()

#%% Example AL neuron skeletal plot for characteristic scales shown in form factor

nidx = 6
bidx = 3
scaleVal = [0.5, 2, 5, 20]
ALcent = MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(BranchData.AL_branchP[nidx][bidx])]#[545 , 310,  44]#ALCM

cmap = cm.get_cmap('viridis', len(MorphData.ALdist))

for s in scaleVal:
    fig = plt.figure(figsize=(8, 8))
    for i in range(len(MorphData.ALdist)):
        listOfPoints = MorphData.ALdist[i]
        for f in range(len(listOfPoints)-1):
            if (((ALcent[0] - s/2 <= listOfPoints[f][0] <= ALcent[0] + s/2) or
                (ALcent[0] - s/2 <= listOfPoints[f+1][0] <= ALcent[0] + s/2)) and
                ((ALcent[1] - s/2 <= listOfPoints[f][1] <= ALcent[1] + s/2) or
                (ALcent[1] - s/2 <= listOfPoints[f+1][1] <= ALcent[1] + s/2)) and
                ((ALcent[2] - s/2 <= listOfPoints[f][2] <= ALcent[2] + s/2) or
                (ALcent[2] - s/2 <= listOfPoints[f+1][2] <= ALcent[2] + s/2))):
                morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
                plt.plot(morph_line[:,0], morph_line[:,1], color=cmap(i))
    
    plt.xlim(ALcent[0] - s/2, ALcent[0] + s/2)
    plt.ylim(ALcent[1] - s/2, ALcent[1] + s/2)
    plt.xticks([])
    plt.yticks([])
    
    # plt.savefig(os.path.join(Parameter.outputdir, 'Pq_all_char_AL_' + str(s) + '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
#%% form factor per glomerulus plotting

q_range = np.logspace(-2,3,100)

Pq_calyx_glo = np.load(r'./Pq_calyx_glo.npy')
Pq_LH_glo = np.load(r'./Pq_LH_glo.npy')
Pq_AL_glo = np.load(r'./Pq_AL_glo.npy')

fig = plt.figure(figsize=(8,6))
plt.plot(np.tile(q_range[:60], (len(glo_idx),1)).T, Pq_calyx_glo[:60])

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_calyx_glo_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.plot(np.tile(q_range[:60], (len(glo_idx),1)).T, Pq_LH_glo[:60])

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_LH_glo_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.plot(np.tile(q_range[:60], (len(glo_idx),1)).T, Pq_AL_glo[:60])

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_AL_glo_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%% form factor per neuron plotting

q_range = np.logspace(-2,3,100)

Pq_calyx_pn = np.load(r'./Pq_calyx.npy')
Pq_LH_pn = np.load(r'./Pq_LH.npy')
Pq_AL_pn = np.load(r'./Pq_AL.npy')

Pq_calyx_pn = np.delete(Pq_calyx_pn, [40, 41], 1)
Pq_AL_pn = np.delete(Pq_AL_pn, 73, 1)

fig = plt.figure(figsize=(8,6))
plt.plot(np.tile(q_range[:60], (np.shape(Pq_calyx_pn)[1],1)).T, Pq_calyx_pn[:60])

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_calyx_pn_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.plot(np.tile(q_range[:60], (np.shape(Pq_LH_pn)[1],1)).T, Pq_LH_pn[:60])

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_LH_pn_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.plot(np.tile(q_range[:60], (np.shape(Pq_AL_pn)[1],1)).T, Pq_AL_pn[:60])

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_AL_pn_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

# calyx_q_idx = len(q_range)
# LH_q_idx = len(q_range)
# AL_q_idx = len(q_range)


fig = plt.figure(figsize=(8,6))
plt.plot(q_range[:calyx_q_idx], np.average(Pq_calyx_pn, axis=1)[:calyx_q_idx], color='tab:orange')
plt.plot(q_range[:calyx_q_idx], Pq_calyx[:calyx_q_idx], color='tab:orange')
plt.fill_between(q_range[:calyx_q_idx], 
                 np.average(Pq_calyx_pn, axis=1)[:calyx_q_idx]+np.std(Pq_calyx_pn, axis=1)[:calyx_q_idx],
                 np.average(Pq_calyx_pn, axis=1)[:calyx_q_idx]-np.std(Pq_calyx_pn, axis=1)[:calyx_q_idx],
                 alpha=0.3, 
                 color='tab:orange')

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_calyx_pn_avg_1.pdf', dpi=300, bbox_inches='tight')
plt.show()



fig = plt.figure(figsize=(8,6))
plt.plot(q_range[:LH_q_idx], np.average(Pq_LH_pn, axis=1)[:LH_q_idx], color='tab:green')
plt.plot(q_range[:LH_q_idx], Pq_LH[:LH_q_idx], color='tab:green')
plt.fill_between(q_range[:LH_q_idx], 
                 np.average(Pq_LH_pn, axis=1)[:LH_q_idx]+np.std(Pq_LH_pn, axis=1)[:LH_q_idx],
                 np.average(Pq_LH_pn, axis=1)[:LH_q_idx]-np.std(Pq_LH_pn, axis=1)[:LH_q_idx],
                 alpha=0.3,
                 color='tab:green')

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_LH_pn_avg_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(8,6))
plt.plot(q_range[:AL_q_idx], np.average(Pq_AL_pn, axis=1)[:AL_q_idx], color='tab:blue')
plt.plot(q_range[:AL_q_idx], Pq_AL[:AL_q_idx], color='tab:blue')
plt.fill_between(q_range[:AL_q_idx], 
                 np.average(Pq_AL_pn, axis=1)[:AL_q_idx]+np.std(Pq_AL_pn, axis=1)[:AL_q_idx],
                 np.average(Pq_AL_pn, axis=1)[:AL_q_idx]-np.std(Pq_AL_pn, axis=1)[:AL_q_idx],
                 alpha=0.3, 
                 color='tab:blue')

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_AL_pn_avg_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(8,6))
for i in range(len(Pq_AL_pn)):
    plt.plot(q_range[:AL_q_idx], Pq_AL_pn[:AL_q_idx,i], marker='.', color='tab:blue', alpha=0.5)

plt.vlines(2*np.pi/np.mean(AL_length_temp), 1e-6, 10, color='tab:blue')

# plt.vlines(2*np.pi/np.median(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dotted')

plt.vlines(1/rgy_AL_full[0], 1e-6, 10, color='tab:blue', ls='--')

line1 = 1/7500*np.power(q_range, -16/7)
# line2 = 1/1000000*np.power(q_range, -4/1)
line3 = 1/5000*np.power(q_range, -2/1)
line4 = 1/2500*np.power(q_range, -1)

# plt.plot(q_range[10:17], line2[10:17], lw=2, color='k')
plt.plot(q_range[19:27], line1[19:27], lw=2, color='k')
plt.plot(q_range[28:36], line3[28:36], lw=2, color='k')
plt.plot(q_range[38:48], line4[38:48], lw=2, color='k')

# plt.text(0.025, 7e-3, r'$\lambda = \dfrac{1}{4}$', fontsize=13)
plt.text(0.05, 0.8e-2, r'$\lambda = \dfrac{7}{16}$', fontsize=13)
plt.text(0.16, 0.8e-3, r'$\lambda = \dfrac{1}{2}$', fontsize=13)
plt.text(0.7, 1.5e-4, r'$\lambda = 1$', fontsize=13)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
plt.ylim(1e-4, 10)
plt.xlim(0.8e-2, 1e2)
# plt.savefig(Parameter.outputdir + '/Pq_per_neuron_AL_full_1.png', dpi=600, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(8,6))
for i in range(len(Pq_AL_pn)):
    plt.plot(q_range[:LH_q_idx], Pq_LH_pn[:LH_q_idx,i], marker='.', color='tab:green', alpha=0.5)

plt.vlines(2*np.pi/np.mean(LH_length_temp), 1e-6, 10, color='tab:green')

# plt.vlines(2*np.pi/np.median(LH_length_temp), 1e-6, 10, color='tab:green', ls='dotted')

plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')

line1 = 1/7500*np.power(q_range, -16/7)
# line2 = 1/1000000*np.power(q_range, -4/1)
line3 = 1/5000*np.power(q_range, -2/1)
line4 = 1/2500*np.power(q_range, -1)

# plt.plot(q_range[10:17], line2[10:17], lw=2, color='k')
plt.plot(q_range[19:27], line1[19:27], lw=2, color='k')
plt.plot(q_range[28:36], line3[28:36], lw=2, color='k')
plt.plot(q_range[38:48], line4[38:48], lw=2, color='k')

# plt.text(0.025, 7e-3, r'$\lambda = \dfrac{1}{4}$', fontsize=13)
plt.text(0.05, 0.8e-2, r'$\lambda = \dfrac{7}{16}$', fontsize=13)
plt.text(0.16, 0.8e-3, r'$\lambda = \dfrac{1}{2}$', fontsize=13)
plt.text(0.7, 1.5e-4, r'$\lambda = 1$', fontsize=13)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
plt.ylim(1e-4, 10)
plt.xlim(0.8e-2, 1e2)
# plt.savefig(Parameter.outputdir + '/Pq_per_neuron_LH_full_1.png', dpi=600, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(8,6))
for i in range(len(Pq_AL_pn)):
    plt.plot(q_range[:calyx_q_idx], Pq_calyx_pn[:calyx_q_idx,i], marker='.', color='tab:orange', alpha=0.5)

plt.vlines(2*np.pi/np.mean(calyx_length_temp), 1e-6, 10, color='tab:orange')

# plt.vlines(2*np.pi/np.median(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dotted')

plt.vlines(1/rgy_calyx_full[0], 1e-6, 10, color='tab:orange', ls='--')

line1 = 1/7500*np.power(q_range, -16/7)
# line2 = 1/1000000*np.power(q_range, -4/1)
line3 = 1/5000*np.power(q_range, -2/1)
line4 = 1/2500*np.power(q_range, -1)

# plt.plot(q_range[10:17], line2[10:17], lw=2, color='k')
plt.plot(q_range[19:27], line1[19:27], lw=2, color='k')
plt.plot(q_range[28:36], line3[28:36], lw=2, color='k')
plt.plot(q_range[38:48], line4[38:48], lw=2, color='k')

# plt.text(0.025, 7e-3, r'$\lambda = \dfrac{1}{4}$', fontsize=13)
plt.text(0.05, 0.8e-2, r'$\lambda = \dfrac{7}{16}$', fontsize=13)
plt.text(0.16, 0.8e-3, r'$\lambda = \dfrac{1}{2}$', fontsize=13)
plt.text(0.7, 1.5e-4, r'$\lambda = 1$', fontsize=13)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel("S(q)", fontsize=15)
plt.ylim(1e-4, 10)
plt.xlim(0.8e-2, 1e2)
# plt.savefig(Parameter.outputdir + '/Pq_per_neuron_calyx_full_1.png', dpi=600, bbox_inches='tight')
plt.show()




#%% Form factor per neuron moving window (FIGURE FORM FACTOR)

mw_Pq_calyx_pn = []
mw_Pq_calyx_pn_err = []
mwx_calyx_pn = []
shiftN = 15

for j in range(len(Pq_calyx_pn[0])):
    mw_Pq_calyx_pn_temp = []
    mw_Pq_calyx_pn_err_temp = []
    mwx_calyx_pn_temp = []
    
    Pq_calyx_posidx = np.where(Pq_calyx_pn[:,j] > 0)[0]
    
    calyx_q_idx_new = Pq_calyx_posidx[Pq_calyx_posidx < calyx_q_idx]
    
    for i in range(len(calyx_q_idx_new) - shiftN):
        mwx_calyx_pn_temp.append(np.average(q_range[calyx_q_idx_new][i:i+shiftN]))
        
        poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(q_range[calyx_q_idx_new][i:i+shiftN]), 
                                                    np.log10(Pq_calyx_pn[calyx_q_idx_new,j][i:i+shiftN]), 
                                                    p0=[1., 0.], 
                                                    maxfev=100000)
        mw_Pq_calyx_pn_temp.append(poptmxc[0])
        mw_Pq_calyx_pn_err_temp.append(np.sqrt(np.diag(pcovmxc))[0])
    
    mw_Pq_calyx_pn.append(mw_Pq_calyx_pn_temp)
    mw_Pq_calyx_pn_err.append(mw_Pq_calyx_pn_err_temp)
    mwx_calyx_pn.append(mwx_calyx_pn_temp)


mw_Pq_LH_pn = []
mw_Pq_LH_pn_err = []
mwx_LH_pn = []

for j in range(len(Pq_LH_pn[0])):
    mw_Pq_LH_pn_temp = []
    mw_Pq_LH_pn_err_temp = []
    mwx_LH_pn_temp = []
    
    Pq_LH_posidx = np.where(Pq_LH_pn[:,j] > 0)[0]
    
    LH_q_idx_new = Pq_LH_posidx[Pq_LH_posidx < LH_q_idx]
    
    for i in range(len(LH_q_idx_new) - shiftN):
        mwx_LH_pn_temp.append(np.average(q_range[LH_q_idx_new][i:i+shiftN]))
        
        poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(q_range[LH_q_idx_new][i:i+shiftN]), 
                                                    np.log10(Pq_LH_pn[LH_q_idx_new,j][i:i+shiftN]), 
                                                    p0=[1., 0.], 
                                                    maxfev=100000)
        mw_Pq_LH_pn_temp.append(poptmxc[0])
        mw_Pq_LH_pn_err_temp.append(np.sqrt(np.diag(pcovmxc))[0])
    
    mw_Pq_LH_pn.append(mw_Pq_LH_pn_temp)
    mw_Pq_LH_pn_err.append(mw_Pq_LH_pn_err_temp)
    mwx_LH_pn.append(mwx_LH_pn_temp)


mw_Pq_AL_pn = []
mw_Pq_AL_pn_err = []
mwx_AL_pn = []

for j in range(len(Pq_AL_pn[0])):
    mw_Pq_AL_pn_temp = []
    mw_Pq_AL_pn_err_temp = []
    mwx_AL_pn_temp = []
    
    Pq_AL_posidx = np.where(Pq_AL_pn[:,j] > 0)[0]
    
    AL_q_idx_new = Pq_AL_posidx[Pq_AL_posidx < AL_q_idx]
    
    for i in range(len(AL_q_idx_new) - shiftN):
        mwx_AL_pn_temp.append(np.average(q_range[AL_q_idx_new][i:i+shiftN]))
        
        poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(q_range[AL_q_idx_new][i:i+shiftN]), 
                                                    np.log10(Pq_AL_pn[AL_q_idx_new,j][i:i+shiftN]), 
                                                    p0=[1., 0.], 
                                                    maxfev=100000)
        mw_Pq_AL_pn_temp.append(poptmxc[0])
        mw_Pq_AL_pn_err_temp.append(np.sqrt(np.diag(pcovmxc))[0])
    
    mw_Pq_AL_pn.append(mw_Pq_AL_pn_temp)
    mw_Pq_AL_pn_err.append(mw_Pq_AL_pn_err_temp)
    mwx_AL_pn.append(mwx_AL_pn_temp)


fig = plt.figure(figsize=(6,4.5))

for i in range(len(mw_Pq_calyx_pn)):
    plt.plot(mwx_calyx_pn[i], -1/np.array(mw_Pq_calyx_pn[i]), color='tab:orange', lw=2, alpha=0.5)

plt.plot(mwx_calyx_pn[0], -1/tolerant_mean(mw_Pq_calyx_pn).data, color='k', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
plt.text(10.3, 1-0.03,'Linear', fontsize=14)
plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)

plt.vlines(2*np.pi/np.mean(calyx_length_temp), 1e-6, 10, color='tab:orange')

# plt.vlines(2*np.pi/np.median(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dotted')

plt.vlines(1/rgy_calyx_full[0], 1e-6, 10, color='tab:orange', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
plt.xticks(fontsize=14)
plt.ylabel(r"$\nu$", fontsize=17)
plt.yticks(fontsize=14)
# plt.savefig(Parameter.outputdir + '/Pq_all_pn_calyx_mv_5.pdf', dpi=300, bbox_inches='tight')
plt.show()
  

fig = plt.figure(figsize=(6,4.5))

for i in range(len(mw_Pq_LH_pn)):
    plt.plot(mwx_LH_pn[i], -1/np.array(mw_Pq_LH_pn[i]), color='tab:green', lw=2, alpha=0.5)

plt.plot(mwx_LH_pn[1], -1/tolerant_mean(mw_Pq_LH_pn).data, color='k', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
plt.text(10.3, 1-0.03,'Linear', fontsize=14)
plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)

plt.vlines(2*np.pi/np.mean(LH_length_temp), 1e-6, 10, color='tab:green')

# plt.vlines(2*np.pi/np.median(LH_length_temp), 1e-6, 10, color='tab:green', ls='dotted')

plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
plt.xticks(fontsize=14)
plt.ylabel(r"$\nu$", fontsize=17)
plt.yticks(fontsize=14)
# plt.savefig(Parameter.outputdir + '/Pq_all_pn_LH_mv_5.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(6,4.5))

for i in range(len(mw_Pq_AL_pn)):
    plt.plot(mwx_AL_pn[i], -1/np.array(mw_Pq_AL_pn[i]), color='tab:blue', lw=2, alpha=0.5)

plt.plot(mwx_AL_pn[0], -1/tolerant_mean(mw_Pq_AL_pn).data, color='k', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
plt.text(10.3, 1-0.03,'Linear', fontsize=14)
plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)

plt.vlines(2*np.pi/np.mean(AL_length_temp), 1e-6, 10, color='tab:blue')

# plt.vlines(2*np.pi/np.median(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dotted')

plt.vlines(1/rgy_AL_full[0], 1e-6, 10, color='tab:blue', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
plt.xticks(fontsize=14)
plt.ylabel(r"$\nu$", fontsize=17)
plt.yticks(fontsize=14)
# plt.savefig(Parameter.outputdir + '/Pq_all_pn_AL_mv_5.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% 16 moving average (FIGURE FORM FACTOR 16)186573
# 35 - 24726
# 85 - 41308
# 106 - 54072
# 107 - 55085
# 110 - 56623
# 130 - 57402
# 147 - 62434
# 159 - 775731

nid = 159

calyxdist_trk_temp = copy.deepcopy(np.unique(MorphData.calyxdist_trk))
calyxdist_trk_temp = np.delete(calyxdist_trk_temp, [40, 41])

ALdist_trk_temp = copy.deepcopy(np.unique(MorphData.ALdist_trk))
ALdist_trk_temp = np.delete(ALdist_trk_temp, 73)

nid_AL = np.where(ALdist_trk_temp == nid)[0][0]
nid_calyx = np.where(calyxdist_trk_temp == nid)[0][0]
nid_LH = np.where(np.unique(MorphData.LHdist_trk) == nid)[0][0]

fig = plt.figure(figsize=(6,4.5))

plt.plot(mwx_calyx_pn[nid_calyx], -1/np.array(mw_Pq_calyx_pn[nid_calyx]), color='tab:orange', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
plt.text(10.3, 1-0.03,'Linear', fontsize=14)
plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)

plt.vlines(2*np.pi/np.mean(calyx_length_temp), 1e-6, 10, color='tab:orange')

# plt.vlines(2*np.pi/np.median(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dotted')

plt.vlines(1/rgy_calyx_full[0], 1e-6, 10, color='tab:orange', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
plt.xticks(fontsize=14)
plt.ylabel(r"$-1/\lambda$", fontsize=17)
plt.yticks(fontsize=14)
# plt.savefig(Parameter.outputdir + '/Pq_' + str(MorphData.neuron_id[nid]) + '_pn_calyx_mv_2.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(6,4.5))

plt.plot(mwx_LH_pn[nid_LH], -1/np.array(mw_Pq_LH_pn[nid_LH]), color='tab:green', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
plt.text(10.3, 1-0.03,'Linear', fontsize=14)
plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)

plt.vlines(2*np.pi/np.mean(LH_length_temp), 1e-6, 10, color='tab:green')

# plt.vlines(2*np.pi/np.median(LH_length_temp), 1e-6, 10, color='tab:green', ls='dotted')

plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
plt.xticks(fontsize=14)
plt.ylabel(r"$-1/\lambda$", fontsize=17)
plt.yticks(fontsize=14)
# plt.savefig(Parameter.outputdir + '/Pq_' + str(MorphData.neuron_id[nid]) + '_pn_LH_mv_2.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(6,4.5))

plt.plot(mwx_AL_pn[nid_AL], -1/np.array(mw_Pq_AL_pn[nid_AL]), color='tab:blue', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
plt.text(10.3, 1-0.03,'Linear', fontsize=14)
plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)

plt.vlines(2*np.pi/np.mean(AL_length_temp), 1e-6, 10, color='tab:blue')

# plt.vlines(2*np.pi/np.median(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dotted')

plt.vlines(1/rgy_AL_full[0], 1e-6, 10, color='tab:blue', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
plt.xticks(fontsize=14)
plt.ylabel(r"$-1/\lambda$", fontsize=17)
plt.yticks(fontsize=14)
# plt.savefig(Parameter.outputdir + '/Pq_' + str(MorphData.neuron_id[nid]) + '_pn_AL_mv_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%% 30891 moving average

nid = 60

fig = plt.figure(figsize=(8,6))

plt.plot(mwx_calyx_pn[nid], -1/np.array(mw_Pq_calyx_pn[nid]), color='tab:orange', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.01, 'Ideal')
plt.text(10.3, 7/16-0.01, '$\Theta$ Solvent')
plt.text(10.3, 1/2-0.01, 'Random')
plt.text(10.3, 1-0.01,'Linear')
plt.text(10.3, 3/5-0.01,'SAW')

plt.vlines(2*np.pi/np.mean(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dashdot')

plt.vlines(2*np.pi/np.median(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dotted')

plt.vlines(1/rgy_calyx_full[0], 1e-6, 10, color='tab:orange', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel(r"$-1/\lambda$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_' + str(MorphData.neuron_id[nid]) + '_pn_calyx_mv_1.pdf', dpi=300, bbox_inches='tight')
plt.show()
  

fig = plt.figure(figsize=(8,6))

plt.plot(mwx_LH_pn[nid], -1/np.array(mw_Pq_LH_pn[nid]), color='tab:green', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.01, 'Ideal')
plt.text(10.3, 7/16-0.01, '$\Theta$ Solvent')
plt.text(10.3, 1/2-0.01, 'Random')
plt.text(10.3, 1-0.01,'Linear')
plt.text(10.3, 3/5-0.01,'SAW')

plt.vlines(2*np.pi/np.mean(LH_length_temp), 1e-6, 10, color='tab:green', ls='dashdot')

plt.vlines(2*np.pi/np.median(LH_length_temp), 1e-6, 10, color='tab:green', ls='dotted')

plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel(r"$-1/\lambda$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_' + str(MorphData.neuron_id[nid]) + '_pn_LH_mv_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(8,6))

plt.plot(mwx_AL_pn[nid], -1/np.array(mw_Pq_AL_pn[nid]), color='tab:blue', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.01, 'Ideal')
plt.text(10.3, 7/16-0.01, '$\Theta$ Solvent')
plt.text(10.3, 1/2-0.01, 'Random')
plt.text(10.3, 1-0.01,'Linear')
plt.text(10.3, 3/5-0.01,'SAW')

plt.vlines(2*np.pi/np.mean(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dashdot')

plt.vlines(2*np.pi/np.median(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dotted')

plt.vlines(1/rgy_AL_full[0], 1e-6, 10, color='tab:blue', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel(r"$-1/\lambda$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_' + str(MorphData.neuron_id[nid]) + '_pn_AL_mv_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% 39254 moving average

nid = 77

fig = plt.figure(figsize=(8,6))

plt.plot(mwx_calyx_pn[nid], -1/np.array(mw_Pq_calyx_pn[nid]), color='tab:orange', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.01, 'Ideal')
plt.text(10.3, 7/16-0.01, '$\Theta$ Solvent')
plt.text(10.3, 1/2-0.01, 'Random')
plt.text(10.3, 1-0.01,'Linear')
plt.text(10.3, 3/5-0.01,'SAW')

plt.vlines(2*np.pi/np.mean(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dashdot')

plt.vlines(2*np.pi/np.median(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dotted')

plt.vlines(1/rgy_calyx_full[0], 1e-6, 10, color='tab:orange', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel(r"$-1/\lambda$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_' + str(MorphData.neuron_id[nid]) + '_pn_calyx_mv_1.pdf', dpi=300, bbox_inches='tight')
plt.show()
  

fig = plt.figure(figsize=(8,6))

plt.plot(mwx_LH_pn[nid], -1/np.array(mw_Pq_LH_pn[nid]), color='tab:green', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.01, 'Ideal')
plt.text(10.3, 7/16-0.01, '$\Theta$ Solvent')
plt.text(10.3, 1/2-0.01, 'Random')
plt.text(10.3, 1-0.01,'Linear')
plt.text(10.3, 3/5-0.01,'SAW')

plt.vlines(2*np.pi/np.mean(LH_length_temp), 1e-6, 10, color='tab:green', ls='dashdot')

plt.vlines(2*np.pi/np.median(LH_length_temp), 1e-6, 10, color='tab:green', ls='dotted')

plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel(r"$-1/\lambda$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_' + str(MorphData.neuron_id[nid]) + '_pn_LH_mv_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(8,6))

plt.plot(mwx_AL_pn[nid], -1/np.array(mw_Pq_AL_pn[nid]), color='tab:blue', lw=2)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.01, 'Ideal')
plt.text(10.3, 7/16-0.01, '$\Theta$ Solvent')
plt.text(10.3, 1/2-0.01, 'Random')
plt.text(10.3, 1-0.01,'Linear')
plt.text(10.3, 3/5-0.01,'SAW')

plt.vlines(2*np.pi/np.mean(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dashdot')

plt.vlines(2*np.pi/np.median(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dotted')

plt.vlines(1/rgy_AL_full[0], 1e-6, 10, color='tab:blue', ls='--')


plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel(r"$-1/\lambda$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_' + str(MorphData.neuron_id[nid]) + '_pn_AL_mv_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% LH form factor of glomerulus with large inter and intra distance difference

for i in range(len(LH_glo_col_idx)):
    fig = plt.figure(figsize=(6,4.5))

    for j in range(len(glo_idx[LH_glo_col_idx[i]])):
        nid = np.where(np.unique(MorphData.LHdist_trk) == glo_idx[LH_glo_col_idx[i]][j])[0][0]
        plt.plot(mwx_LH_pn[nid], -1/np.array(mw_Pq_LH_pn[nid]), color='tab:green', lw=2)
    
    plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
    plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
    plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
    plt.hlines(1, 0.01, 100, ls='dashed', color='k')
    plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
    plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
    plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
    plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
    plt.text(10.3, 1-0.03,'Linear', fontsize=14)
    plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)
    
    plt.vlines(2*np.pi/np.mean(LH_length_temp), 1e-6, 10, color='tab:green')
    plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')
    
    plt.xscale('log')
    plt.ylim(0.1, 1.7)
    plt.xlim(0.01, 10)
    plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
    plt.xticks(fontsize=14)
    plt.ylabel(r"$\nu$", fontsize=17)
    plt.yticks(fontsize=14)
    plt.title(glo_list[LH_glo_col_idx[i]])
    # plt.savefig(Parameter.outputdir + '/Pq_lIIDd_' + str(glo_list[LH_glo_col_idx[i]]) + '_pn_LH_mv_2.pdf', dpi=300, bbox_inches='tight')
    plt.show()


#%% LH form factor of glomerulus with large inter and intra distance difference - type

pher = ['DL3', 'VA1d', 'DA1', 'DC3']
# attr = ['VM2', 'VM7d', 'VM7v']
# aver = ['DA2', 'DM6', 'VA5', 'VA7m', 'VM3']

attr = ['VM2', 'VM7d', 'VM7v', 'DL3', 'VA1d', 'DA1']
aver = ['DA2', 'DM6', 'VA5', 'VA7m', 'VM3', 'DC3']

pherlist = []

for i in range(len(pher)):
    for j in range(len(glo_idx[glo_list.index(pher[i])])):
        nid = np.where(np.unique(MorphData.LHdist_trk) == glo_idx[glo_list.index(pher[i])][j])[0][0]
        pherlist.append(-1/np.array(mw_Pq_LH_pn[nid]))

attrlist = []

for i in range(len(attr)):
    for j in range(len(glo_idx[glo_list.index(attr[i])])):
        nid = np.where(np.unique(MorphData.LHdist_trk) == glo_idx[glo_list.index(attr[i])][j])[0][0]
        attrlist.append(-1/np.array(mw_Pq_LH_pn[nid]))


averlist = []

for i in range(len(aver)):
    for j in range(len(glo_idx[glo_list.index(aver[i])])):
        nid = np.where(np.unique(MorphData.LHdist_trk) == glo_idx[glo_list.index(aver[i])][j])[0][0]
        averlist.append(-1/np.array(mw_Pq_LH_pn[nid]))

fig = plt.figure(figsize=(6,4.5))
plt.plot(mwx_LH_pn[1], tolerant_mean(pherlist).data, color='tab:blue', lw=2)
plt.plot(mwx_LH_pn[1], tolerant_mean(attrlist).data, color='tab:green', lw=2)
plt.plot(mwx_LH_pn[1], tolerant_mean(averlist).data, color='tab:red', lw=2)
plt.fill_between(mwx_LH_pn[1], 
                  tolerant_mean(pherlist).data-tolerant_std_error(pherlist), 
                  tolerant_mean(pherlist).data+tolerant_std_error(pherlist),
                  alpha=0.3,
                  color='tab:blue')
plt.fill_between(mwx_LH_pn[1], 
                  tolerant_mean(attrlist).data-tolerant_std_error(attrlist), 
                  tolerant_mean(attrlist).data+tolerant_std_error(attrlist),
                  alpha=0.3,
                  color='tab:green')
plt.fill_between(mwx_LH_pn[1], 
                  tolerant_mean(averlist).data-tolerant_std_error(averlist), 
                  tolerant_mean(averlist).data+tolerant_std_error(averlist),
                  alpha=0.3,
                  color='tab:red')
# plt.legend(['Pheromones', 'Attractive', 'Aversive'], fontsize=14)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
plt.text(10.3, 1-0.03,'Linear', fontsize=14)
plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)

plt.vlines(2*np.pi/np.mean(LH_length_temp), 1e-6, 10, color='k')
plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='k', ls='--')

plt.xscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
plt.xticks(fontsize=14)
plt.ylabel(r"$\nu$", fontsize=17)
plt.yticks(fontsize=14)
# plt.savefig(Parameter.outputdir + '/Pq_lIIDd_avg_LH_4.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% AL form factor of glomerulus with large inter and intra distance difference - type

pher = ['DL3', 'VA1d', 'DA1', 'DC3']
# attr = ['VM2', 'VM7d', 'VM7v']
# aver = ['DA2', 'DM6', 'VA5', 'VA7m', 'VM3']

attr = ['VM2', 'VM7d', 'VM7v', 'DL3', 'VA1d', 'DA1']
aver = ['DA2', 'DM6', 'VA5', 'VA7m', 'VM3', 'DC3']

ALdist_trk_temp = copy.deepcopy(np.unique(MorphData.ALdist_trk))
ALdist_trk_temp = np.delete(ALdist_trk_temp, 73)

pherlist = []

for i in range(len(pher)):
    for j in range(len(glo_idx[glo_list.index(pher[i])])):
        nid = np.where(ALdist_trk_temp == glo_idx[glo_list.index(pher[i])][j])[0][0]
        pherlist.append(-1/np.array(mw_Pq_AL_pn[nid]))

attrlist = []

for i in range(len(attr)):
    for j in range(len(glo_idx[glo_list.index(attr[i])])):
        nid = np.where(ALdist_trk_temp == glo_idx[glo_list.index(attr[i])][j])[0][0]
        attrlist.append(-1/np.array(mw_Pq_AL_pn[nid]))


averlist = []

for i in range(len(aver)):
    for j in range(len(glo_idx[glo_list.index(aver[i])])):
        nid = np.where(ALdist_trk_temp == glo_idx[glo_list.index(aver[i])][j])[0][0]
        averlist.append(-1/np.array(mw_Pq_AL_pn[nid]))

fig = plt.figure(figsize=(6,4.5))
plt.plot(mwx_AL_pn[0], tolerant_mean(pherlist).data, color='tab:blue', lw=2)
plt.plot(mwx_AL_pn[0], tolerant_mean(attrlist).data, color='tab:green', lw=2)
plt.plot(mwx_AL_pn[0], tolerant_mean(averlist).data, color='tab:red', lw=2)
plt.fill_between(mwx_AL_pn[0], 
                  tolerant_mean(pherlist).data-tolerant_std_error(pherlist), 
                  tolerant_mean(pherlist).data+tolerant_std_error(pherlist),
                  alpha=0.3,
                  color='tab:blue')
plt.fill_between(mwx_AL_pn[0], 
                  tolerant_mean(attrlist).data-tolerant_std_error(attrlist), 
                  tolerant_mean(attrlist).data+tolerant_std_error(attrlist),
                  alpha=0.3,
                  color='tab:green')
plt.fill_between(mwx_AL_pn[0], 
                  tolerant_mean(averlist).data-tolerant_std_error(averlist), 
                  tolerant_mean(averlist).data+tolerant_std_error(averlist),
                  alpha=0.3,
                  color='tab:red')
plt.legend(['Pheromones', 'Attractive', 'Aversive'], fontsize=14)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
# plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
# plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
# plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
# plt.text(10.3, 1-0.03,'Linear', fontsize=14)
# plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)

plt.vlines(2*np.pi/np.mean(AL_length_temp), 1e-6, 10, color='k')
plt.vlines(1/rgy_AL_full[0], 1e-6, 10, color='k', ls='--')

plt.xscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
plt.xticks(fontsize=14)
plt.ylabel(r"$\nu$", fontsize=17)
plt.yticks(fontsize=14)
# plt.savefig(Parameter.outputdir + '/Pq_lIIDd_avg_AL_4.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% calyx form factor of glomerulus with large inter and intra distance difference - type

pher = ['DL3', 'VA1d', 'DA1', 'DC3']
# attr = ['VM2', 'VM7d', 'VM7v']
# aver = ['DA2', 'DM6', 'VA5', 'VA7m', 'VM3']

attr = ['VM2', 'VM7d', 'VM7v', 'DL3', 'VA1d', 'DA1']
aver = ['DA2', 'DM6', 'VA5', 'VA7m', 'VM3', 'DC3']

calyxdist_trk_temp = copy.deepcopy(np.unique(MorphData.calyxdist_trk))
calyxdist_trk_temp = np.delete(calyxdist_trk_temp, [40, 41])

pherlist = []

for i in range(len(pher)):
    for j in range(len(glo_idx[glo_list.index(pher[i])])):
        nid = np.where(calyxdist_trk_temp == glo_idx[glo_list.index(pher[i])][j])[0][0]
        pherlist.append(-1/np.array(mw_Pq_calyx_pn[nid]))

attrlist = []

for i in range(len(attr)):
    for j in range(len(glo_idx[glo_list.index(attr[i])])):
        nid = np.where(calyxdist_trk_temp == glo_idx[glo_list.index(attr[i])][j])[0][0]
        attrlist.append(-1/np.array(mw_Pq_calyx_pn[nid]))


averlist = []

for i in range(len(aver)):
    for j in range(len(glo_idx[glo_list.index(aver[i])])):
        nid = np.where(calyxdist_trk_temp == glo_idx[glo_list.index(aver[i])][j])[0][0]
        averlist.append(-1/np.array(mw_Pq_calyx_pn[nid]))

fig = plt.figure(figsize=(6,4.5))
plt.plot(mwx_calyx_pn[1], tolerant_mean(pherlist).data, color='tab:blue', lw=2)
plt.plot(mwx_calyx_pn[1], tolerant_mean(attrlist).data, color='tab:green', lw=2)
plt.plot(mwx_calyx_pn[1], tolerant_mean(averlist).data, color='tab:red', lw=2)
plt.fill_between(mwx_calyx_pn[1], 
                  tolerant_mean(pherlist).data-tolerant_std_error(pherlist), 
                  tolerant_mean(pherlist).data+tolerant_std_error(pherlist),
                  alpha=0.3,
                  color='tab:blue')
plt.fill_between(mwx_calyx_pn[1], 
                  tolerant_mean(attrlist).data-tolerant_std_error(attrlist), 
                  tolerant_mean(attrlist).data+tolerant_std_error(attrlist),
                  alpha=0.3,
                  color='tab:green')
plt.fill_between(mwx_calyx_pn[1], 
                  tolerant_mean(averlist).data-tolerant_std_error(averlist), 
                  tolerant_mean(averlist).data+tolerant_std_error(averlist),
                  alpha=0.3,
                  color='tab:red')
# plt.legend(['Pheromones', 'Attractive', 'Aversive'], fontsize=14)

plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
plt.hlines(1, 0.01, 100, ls='dashed', color='k')
plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
# plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
# plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
# plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
# plt.text(10.3, 1-0.03,'Linear', fontsize=14)
# plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)

plt.vlines(2*np.pi/np.mean(calyx_length_temp), 1e-6, 10, color='k')
plt.vlines(1/rgy_calyx_full[0], 1e-6, 10, color='k', ls='--')

plt.xscale('log')
plt.ylim(0.1, 1.7)
plt.xlim(0.01, 10)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
plt.xticks(fontsize=14)
plt.ylabel(r"$\nu$", fontsize=17)
plt.yticks(fontsize=14)
# plt.savefig(Parameter.outputdir + '/Pq_lIIDd_avg_calyx_4.pdf', dpi=300, bbox_inches='tight')
plt.show()




#%% LH form factor of glomerulus with large PN number

for i in range(len(np.argsort(glo_len)[-6:])):
    fig = plt.figure(figsize=(6,4.5))

    for j in range(len(glo_idx[np.argsort(glo_len)[-6:][i]])):
        nid = np.where(np.unique(MorphData.LHdist_trk) == glo_idx[np.argsort(glo_len)[-6:][i]][j])[0][0]
        plt.plot(mwx_LH_pn[nid], -1/np.array(mw_Pq_LH_pn[nid]), color='tab:green', lw=2)
    
    plt.hlines(1/4, 0.01, 100, ls='dashed', color='k')
    plt.hlines(7/16, 0.01, 100, ls='dashed', color='k')
    plt.hlines(1/2, 0.01, 100, ls='dashed', color='k')
    plt.hlines(1, 0.01, 100, ls='dashed', color='k')
    plt.hlines(3/5, 0.01, 100, ls='dashed', color='k')
    plt.text(10.3, 1/4-0.03, 'Ideal', fontsize=14)
    plt.text(10.3, 7/16-0.04, '$\Theta$ Solvent', fontsize=14)
    plt.text(10.3, 1/2-0.02, 'Random', fontsize=14)
    plt.text(10.3, 1-0.03,'Linear', fontsize=14)
    plt.text(10.3, 3/5-0.03,'SAW', fontsize=14)
    
    plt.vlines(2*np.pi/np.mean(LH_length_temp), 1e-6, 10, color='tab:green')
    plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')
    
    plt.xscale('log')
    plt.ylim(0.1, 1.7)
    plt.xlim(0.01, 10)
    plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=17)
    plt.xticks(fontsize=14)
    plt.ylabel(r"$-1/\lambda$", fontsize=17)
    plt.yticks(fontsize=14)
    # plt.savefig(Parameter.outputdir + '/Pq_lPNN_' + str(MorphData.neuron_id[nid]) + '_pn_LH_mv_2.pdf', dpi=300, bbox_inches='tight')
    plt.show()


#%% Neuron plot clump non-clump LH

clump_noclump = ['DL3', 'DM5']

idx_all_aver = []
for i in range(len(clump_noclump)):
    idx_all_aver.append(glo_idx[glo_list.index(clump_noclump[i])])

# for k in range(len(aver)):
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.set_box_aspect((1,1,1))
# cmap = cm.get_cmap('jet', len(clump_noclump))
clist = ['tab:blue', 'tab:red']
for i in range(len(MorphData.LHdist)):
    glo_n = MorphData.LHdist_trk[i]
    isglo = [i for i, idx in enumerate(idx_all_aver) if glo_n in idx]
    listOfPoints = MorphData.LHdist[i]
    if len(isglo) > 0:
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=clist[isglo[0]], lw=1.)
    else:
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color='gray', lw=0.25, alpha=0.25)
ax.grid(True)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlim(420, 480)
ax.set_ylim(280, 220)
ax.set_zlim(140, 200)
# plt.savefig(os.path.join(Parameter.outputdir, 'neurons_LH_clump_noclump_2.png'), dpi=600, bbox_inches='tight')
plt.show()


#%% Neuron plot clump non-clump calyx

clump_noclump = ['DL3', 'DM5']

idx_all_aver = []
for i in range(len(clump_noclump)):
    idx_all_aver.append(glo_idx[glo_list.index(clump_noclump[i])])

# for k in range(len(aver)):
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.set_box_aspect((1,1,1))
# cmap = cm.get_cmap('jet', len(clump_noclump))
clist = ['tab:blue', 'tab:red']
for i in range(len(MorphData.calyxdist)):
    glo_n = MorphData.calyxdist_trk[i]
    isglo = [i for i, idx in enumerate(idx_all_aver) if glo_n in idx]
    listOfPoints = MorphData.calyxdist[i]
    if len(isglo) > 0:
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=clist[isglo[0]], lw=1.)
    else:
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color='gray', lw=0.25, alpha=0.25)
ax.grid(True)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlim(500, 560)
ax.set_ylim(280, 220)
ax.set_zlim(160, 220)
# plt.savefig(os.path.join(Parameter.outputdir, 'neurons_calyx_clump_noclump_1.png'), dpi=600, bbox_inches='tight')
plt.show()



#%% Neuron plot aver non aver

aver_naver = ['DA2', 'VA1d']

idx_aver_naver = []
for i in range(len(aver_naver)):
    idx_aver_naver.append(np.array(fp)[glo_idx[glo_list.index(aver_naver[i])]])


#%% Example calyx neuron skeletal plot for characteristic scales shown in form factor

nidx = 8
bidx = 0
scaleVal = [1, 2, 5, 10]
calyxcent = MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(BranchData.calyx_branchP[nidx][bidx])]

cmap = cm.get_cmap('viridis', len(MorphData.calyxdist))

for s in scaleVal:
    fig = plt.figure(figsize=(6, 6))
    tararr = np.array(MorphData.morph_dist[nidx])
    for p in range(len(MorphData.morph_parent[nidx])):
        if MorphData.morph_parent[nidx][p] < 0:
            pass
        else:
            if ((calyxcent[0] - s/2 <= MorphData.morph_dist[nidx][p][0] <= calyxcent[0] + s/2) and
                (calyxcent[1] - s/2 <= MorphData.morph_dist[nidx][p][1] <= calyxcent[1] + s/2) and
                (calyxcent[2] - s/2 <= MorphData.morph_dist[nidx][p][2] <= calyxcent[2] + s/2)):
                morph_line = np.vstack((MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(MorphData.morph_parent[nidx][p])], 
                                        MorphData.morph_dist[nidx][p]))
                plt.plot(morph_line[:,0], morph_line[:,1], color=cmap(nidx))
    
    plt.xlim(calyxcent[0] - s/2, calyxcent[0] + s/2)
    plt.ylim(calyxcent[1] - s/2, calyxcent[1] + s/2)
    plt.xticks([])
    plt.yticks([])
    
    # plt.savefig(os.path.join(Parameter.outputdir, 'Pq_per_neuron_calyx_' + str(s) + '.png'), dpi=300, bbox_inches='tight')
    plt.show()


#%% Example LH neuron skeletal plot for characteristic scales shown in form factor

nidx = 8
bidx = 2
scaleVal = [1, 2, 5, 10]
LHcent = MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(BranchData.LH_branchP[nidx][bidx])]

cmap = cm.get_cmap('viridis', len(MorphData.LHdist))

for s in scaleVal:
    fig = plt.figure(figsize=(6, 6))
    tararr = np.array(MorphData.morph_dist[nidx])
    for p in range(len(MorphData.morph_parent[nidx])):
        if MorphData.morph_parent[nidx][p] < 0:
            pass
        else:
            if ((LHcent[0] - s/2 <= MorphData.morph_dist[nidx][p][0] <= LHcent[0] + s/2) and
                (LHcent[1] - s/2 <= MorphData.morph_dist[nidx][p][1] <= LHcent[1] + s/2) and
                (LHcent[2] - s/2 <= MorphData.morph_dist[nidx][p][2] <= LHcent[2] + s/2)):
                morph_line = np.vstack((MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(MorphData.morph_parent[nidx][p])], 
                                        MorphData.morph_dist[nidx][p]))
                plt.plot(morph_line[:,0], morph_line[:,1], color=cmap(nidx))
    
    plt.xlim(LHcent[0] - s/2, LHcent[0] + s/2)
    plt.ylim(LHcent[1] - s/2, LHcent[1] + s/2)
    plt.xticks([])
    plt.yticks([])
    
    # plt.savefig(os.path.join(Parameter.outputdir, 'Pq_per_neuron_LH_' + str(s) + '.png'), dpi=300, bbox_inches='tight')
    plt.show()


#%% Example AL neuron skeletal plot for characteristic scales shown in form factor

nidx = 8
bidx = 15
scaleVal = [1, 2, 5, 10]
ALcent = MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(BranchData.AL_branchP[nidx][bidx])]

cmap = cm.get_cmap('viridis', len(MorphData.ALdist))

for s in scaleVal:
    fig = plt.figure(figsize=(6, 6))
    tararr = np.array(MorphData.morph_dist[nidx])
    for p in range(len(MorphData.morph_parent[nidx])):
        if MorphData.morph_parent[nidx][p] < 0:
            pass
        else:
            if ((ALcent[0] - s/2 <= MorphData.morph_dist[nidx][p][0] <= ALcent[0] + s/2) and
                (ALcent[1] - s/2 <= MorphData.morph_dist[nidx][p][1] <= ALcent[1] + s/2) and
                (ALcent[2] - s/2 <= MorphData.morph_dist[nidx][p][2] <= ALcent[2] + s/2)):
                morph_line = np.vstack((MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(MorphData.morph_parent[nidx][p])], 
                                        MorphData.morph_dist[nidx][p]))
                plt.plot(morph_line[:,0], morph_line[:,1], color=cmap(nidx))
    
    plt.xlim(ALcent[0] - s/2, ALcent[0] + s/2)
    plt.ylim(ALcent[1] - s/2, ALcent[1] + s/2)
    plt.xticks([])
    plt.yticks([])
    
    # plt.savefig(os.path.join(Parameter.outputdir, 'Pq_per_neuron_AL_' + str(s) + '.png'), dpi=300, bbox_inches='tight')
    plt.show()

            

#%% Form factor per neuron moving window fitted averaged

mw_Pq_calyx_pn = []
mw_Pq_calyx_pn_err = []
mwx_calyx_pn = []
shiftN = 15

for j in range(np.shape(Pq_calyx_pn)[1]):
    mw_Pq_calyx_pn_temp = []
    mw_Pq_calyx_pn_err_temp = []
    mwx_calyx_pn_temp = []
    for i in range(len(q_range[:calyx_q_idx]) - shiftN):
        mwx_calyx_pn_temp.append(np.average(q_range[:calyx_q_idx][i:i+shiftN]))
        
        poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(q_range[:calyx_q_idx][i:i+shiftN]), 
                                                    np.log10(Pq_calyx_pn[:calyx_q_idx,j][i:i+shiftN]), 
                                                    p0=[1., 0.], 
                                                    maxfev=100000)
        mw_Pq_calyx_pn_temp.append(poptmxc[0])
        mw_Pq_calyx_pn_err_temp.append(np.sqrt(np.diag(pcovmxc))[0])
        
    mwx_calyx_pn.append(mwx_calyx_pn_temp)
    mw_Pq_calyx_pn.append(mw_Pq_calyx_pn_temp)
    mw_Pq_calyx_pn_err.append(mw_Pq_calyx_pn_err_temp)

mw_Pq_LH_pn = []
mw_Pq_LH_pn_err = []
mwx_LH_pn = []

for j in range(np.shape(Pq_LH_pn)[1]):
    mw_Pq_LH_pn_temp = []
    mw_Pq_LH_pn_err_temp = []
    mwx_LH_pn_temp = []
    for i in range(len(q_range[:LH_q_idx]) - shiftN):
        mwx_LH_pn_temp.append(np.average(q_range[:LH_q_idx][i:i+shiftN]))
        
        poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(q_range[:LH_q_idx][i:i+shiftN]), 
                                                    np.log10(Pq_LH_pn[:LH_q_idx,j][i:i+shiftN]), 
                                                    p0=[1., 0.], 
                                                    maxfev=100000)
        mw_Pq_LH_pn_temp.append(poptmxc[0])
        mw_Pq_LH_pn_err_temp.append(np.sqrt(np.diag(pcovmxc))[0])
        
    mwx_LH_pn.append(mwx_LH_pn_temp)
    mw_Pq_LH_pn.append(mw_Pq_LH_pn_temp)
    mw_Pq_LH_pn_err.append(mw_Pq_LH_pn_err_temp)

mw_Pq_AL_pn = []
mw_Pq_AL_pn_err = []
mwx_AL_pn = []

for j in range(np.shape(Pq_AL_pn)[1]):
    mw_Pq_AL_pn_temp = []
    mw_Pq_AL_pn_err_temp = []
    mwx_AL_pn_temp = []
    for i in range(len(q_range[:AL_q_idx]) - shiftN):
        mwx_AL_pn_temp.append(np.average(q_range[:AL_q_idx][i:i+shiftN]))
        
        poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(q_range[:AL_q_idx][i:i+shiftN]), 
                                                    np.log10(Pq_AL_pn[:AL_q_idx,j][i:i+shiftN]), 
                                                    p0=[1., 0.], 
                                                    maxfev=100000)
        mw_Pq_AL_pn_temp.append(poptmxc[0])
        mw_Pq_AL_pn_err_temp.append(np.sqrt(np.diag(pcovmxc))[0])
        
    mwx_AL_pn.append(mwx_AL_pn_temp)
    mw_Pq_AL_pn.append(mw_Pq_AL_pn_temp)
    mw_Pq_AL_pn_err.append(mw_Pq_AL_pn_err_temp)

    

fig = plt.figure(figsize=(8,6))
plt.plot(np.average(mwx_AL_pn, axis=0), -1/np.average(mw_Pq_AL_pn, axis=0), lw=2)
plt.plot(np.average(mwx_calyx_pn, axis=0), -1/np.average(mw_Pq_calyx_pn, axis=0), lw=2)
plt.plot(np.average(mwx_LH_pn, axis=0), -1/np.average(mw_Pq_LH_pn, axis=0), lw=2)
plt.fill_between(np.average(mwx_AL_pn, axis=0), 
                  -1/(np.average(mw_Pq_AL_pn, axis=0)-np.std(mw_Pq_AL_pn, axis=0)), 
                  -1/(np.average(mw_Pq_AL_pn, axis=0)+np.std(mw_Pq_AL_pn, axis=0)), 
                  alpha=0.3)
plt.fill_between(np.average(mwx_calyx_pn, axis=0), 
                  -1/(np.average(mw_Pq_calyx_pn, axis=0)-np.std(mw_Pq_calyx_pn, axis=0)), 
                  -1/(np.average(mw_Pq_calyx_pn, axis=0)+np.std(mw_Pq_calyx_pn, axis=0)), 
                  alpha=0.3)
plt.fill_between(np.average(mwx_LH_pn, axis=0), 
                  -1/(np.average(mw_Pq_LH_pn, axis=0)-np.std(mw_Pq_LH_pn, axis=0)), 
                  -1/(np.average(mw_Pq_LH_pn, axis=0)+np.std(mw_Pq_LH_pn, axis=0)), 
                  alpha=0.3)

plt.hlines(1/4, 0.01, 100, ls='dashed')
plt.hlines(7/16, 0.01, 100, ls='dashed')
plt.hlines(1/2, 0.01, 100, ls='dashed')
plt.hlines(1, 0.01, 100, ls='dashed')
plt.text(10.3, 1/4-0.01, 'Ideal')
plt.text(10.3, 7/16-0.01, '$\Theta$ Solvent')
plt.text(10.3, 1/2-0.01, 'Random')
plt.text(10.3, 1-0.01,' Rigid')

plt.vlines(1/np.mean(AL_length_temp), 1e-6, 10, color='tab:blue')
plt.vlines(1/np.mean(calyx_length_temp), 1e-6, 10, color='tab:orange')
plt.vlines(1/np.mean(LH_length_temp), 1e-6, 10, color='tab:green')

plt.vlines(1/np.median(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dotted')
plt.vlines(1/np.median(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dotted')
plt.vlines(1/np.median(LH_length_temp), 1e-6, 10, color='tab:green', ls='dotted')

plt.vlines(1/rgy_AL_full[0], 1e-6, 10, color='tab:blue', ls='--')
plt.vlines(1/rgy_calyx_full[0], 1e-6, 10, color='tab:orange', ls='--')
plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')

plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.5)
plt.xlim(0.01, 10)
plt.legend(["AL", "MB calyx", "LH"], fontsize=13)
plt.xlabel("q ($\mu\mathrm{m}^{-1}$)", fontsize=15)
plt.ylabel(r"$-1/\lambda$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Pq_all_pn_mv_1.pdf', dpi=300, bbox_inches='tight')
plt.show()



#%% Rgy centered at BP

def cons_check(val):
    val = sorted(set(val))
    gaps = [[s, e] for s, e in zip(val, val[1:]) if s+1 < e]
    edges = iter(val[:1] + sum(gaps, []) + val[-1:])
    return list(zip(edges, edges))

radiussize = np.logspace(-1, 2, 100)[::3]

un_calyx = np.unique(MorphData.calyxdist_trk)
un_LH = np.unique(MorphData.LHdist_trk)
un_AL = np.unique(MorphData.ALdist_trk)

rGy_calyx_bp = []
contour_calyx_bp = []
count_calyx_bp = []

for ib in range(len(un_calyx)):
    rGy_calyx_bp_temp1 = []
    count_calyx_bp_temp1 = []
    contour_calyx_bp_temp1 = []
    
    idx = np.where(MorphData.calyxdist_trk == un_calyx[ib])[0]
    tarval = list(np.array(MorphData.calyxdist, dtype=object)[idx])
    calyxdist_bp_flat = np.array([item for sublist in tarval for item in sublist])
    
    branchPidx = BranchData.calyx_branchP[un_calyx[ib]]
    branchPTrkidx = BranchData.calyx_branchTrk[un_calyx[ib]]
    branchPTrkidx_flat = np.array([item for sublist in branchPTrkidx for item in sublist])
    
    for ibp in branchPidx:
        rGy_calyx_bp_temp2 = []
        count_calyx_bp_temp2 = []
        contour_calyx_bp_temp2 = []
        
        calyx_CM_temp = calyxdist_bp_flat[np.where(branchPTrkidx_flat == ibp)[0][0]]
        
        for b in range(len(radiussize)):
            inbound_calyx = np.where(np.sqrt(np.square(calyxdist_bp_flat[:,0] - calyx_CM_temp[0]) +
                                             np.square(calyxdist_bp_flat[:,1] - calyx_CM_temp[1]) +
                                             np.square(calyxdist_bp_flat[:,2] - calyx_CM_temp[2])) <= radiussize[b])[0]
            (rGy_temp, cML_temp) = utils.radiusOfGyration(np.array([calyxdist_bp_flat[inbound_calyx]]))
            
            dist_calyx = 0
            lenc = 0
            if len(inbound_calyx) > 1:
                valist = cons_check(inbound_calyx)
                for ibx in range(len(valist)):
                    val = calyxdist_bp_flat[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                    x = val[:,0]
                    y = val[:,1]
                    z = val[:,2]
                
                    xd = [j-i for i, j in zip(x[:-1], x[1:])]
                    yd = [j-i for i, j in zip(y[:-1], y[1:])]
                    zd = [j-i for i, j in zip(z[:-1], z[1:])]
                    dist_calyx += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                    if len(val) > 1:
                        lenc += len(val)
            rGy_calyx_bp_temp2.append(rGy_temp[0])
            contour_calyx_bp_temp2.append(dist_calyx)
            count_calyx_bp_temp2.append(lenc)
        
        rGy_calyx_bp_temp1.append(rGy_calyx_bp_temp2)
        contour_calyx_bp_temp1.append(contour_calyx_bp_temp2)
        count_calyx_bp_temp1.append(count_calyx_bp_temp2)
    
    rGy_calyx_bp.append(rGy_calyx_bp_temp1)
    contour_calyx_bp.append(contour_calyx_bp_temp1)
    count_calyx_bp.append(count_calyx_bp_temp1)

rGy_LH_bp = []
contour_LH_bp = []
count_LH_bp = []

for ib in range(len(un_LH)):
    rGy_LH_bp_temp1 = []
    count_LH_bp_temp1 = []
    contour_LH_bp_temp1 = []
    
    idx = np.where(MorphData.LHdist_trk == un_LH[ib])[0]
    tarval = list(np.array(MorphData.LHdist, dtype=object)[idx])
    LHdist_bp_flat = np.array([item for sublist in tarval for item in sublist])
    
    branchPidx = BranchData.LH_branchP[un_LH[ib]]
    branchPTrkidx = BranchData.LH_branchTrk[un_LH[ib]]
    branchPTrkidx_flat = np.array([item for sublist in branchPTrkidx for item in sublist])
    
    for ibp in branchPidx:
        rGy_LH_bp_temp2 = []
        count_LH_bp_temp2 = []
        contour_LH_bp_temp2 = []
        
        LH_CM_temp = LHdist_bp_flat[np.where(branchPTrkidx_flat == ibp)[0][0]]
        
        for b in range(len(radiussize)):
            inbound_LH = np.where(np.sqrt(np.square(LHdist_bp_flat[:,0] - LH_CM_temp[0]) +
                                          np.square(LHdist_bp_flat[:,1] - LH_CM_temp[1]) +
                                          np.square(LHdist_bp_flat[:,2] - LH_CM_temp[2])) <= radiussize[b])[0]
            (rGy_temp, cML_temp) = utils.radiusOfGyration(np.array([LHdist_bp_flat[inbound_LH]]))
            
            dist_LH = 0
            lenc = 0
            if len(inbound_LH) > 1:
                valist = cons_check(inbound_LH)
                for ibx in range(len(valist)):
                    val = LHdist_bp_flat[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                    x = val[:,0]
                    y = val[:,1]
                    z = val[:,2]
                
                    xd = [j-i for i, j in zip(x[:-1], x[1:])]
                    yd = [j-i for i, j in zip(y[:-1], y[1:])]
                    zd = [j-i for i, j in zip(z[:-1], z[1:])]
                    dist_LH += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                    if len(val) > 1:
                        lenc += len(val)
            rGy_LH_bp_temp2.append(rGy_temp[0])
            contour_LH_bp_temp2.append(dist_LH)
            count_LH_bp_temp2.append(lenc)
        
        rGy_LH_bp_temp1.append(rGy_LH_bp_temp2)
        contour_LH_bp_temp1.append(contour_LH_bp_temp2)
        count_LH_bp_temp1.append(count_LH_bp_temp2)
    
    rGy_LH_bp.append(rGy_LH_bp_temp1)
    contour_LH_bp.append(contour_LH_bp_temp1)
    count_LH_bp.append(count_LH_bp_temp1)

rGy_AL_bp = []
contour_AL_bp = []
count_AL_bp = []

for ib in range(len(un_AL)):
    rGy_AL_bp_temp1 = []
    count_AL_bp_temp1 = []
    contour_AL_bp_temp1 = []
    
    idx = np.where(MorphData.ALdist_trk == un_AL[ib])[0]
    tarval = list(np.array(MorphData.ALdist, dtype=object)[idx])
    ALdist_bp_flat = np.array([item for sublist in tarval for item in sublist])
    
    branchPidx = BranchData.AL_branchP[un_AL[ib]]
    branchPTrkidx = BranchData.AL_branchTrk[un_AL[ib]]
    branchPTrkidx_flat = np.array([item for sublist in branchPTrkidx for item in sublist])
    
    for ibp in branchPidx:
        rGy_AL_bp_temp2 = []
        count_AL_bp_temp2 = []
        contour_AL_bp_temp2 = []
        
        AL_CM_temp = ALdist_bp_flat[np.where(branchPTrkidx_flat == ibp)[0][0]]
        
        for b in range(len(radiussize)):
            inbound_AL = np.where(np.sqrt(np.square(ALdist_bp_flat[:,0] - AL_CM_temp[0]) +
                                          np.square(ALdist_bp_flat[:,1] - AL_CM_temp[1]) +
                                          np.square(ALdist_bp_flat[:,2] - AL_CM_temp[2])) <= radiussize[b])[0]
            (rGy_temp, cML_temp) = utils.radiusOfGyration(np.array([ALdist_bp_flat[inbound_AL]]))
            
            dist_AL = 0
            lenc = 0
            if len(inbound_AL) > 1:
                valist = cons_check(inbound_AL)
                for ibx in range(len(valist)):
                    val = ALdist_bp_flat[np.arange(valist[ibx][0], valist[ibx][1]+1)]
                    x = val[:,0]
                    y = val[:,1]
                    z = val[:,2]
                
                    xd = [j-i for i, j in zip(x[:-1], x[1:])]
                    yd = [j-i for i, j in zip(y[:-1], y[1:])]
                    zd = [j-i for i, j in zip(z[:-1], z[1:])]
                    dist_AL += np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                    if len(val) > 1:
                        lenc += len(val)
            rGy_AL_bp_temp2.append(rGy_temp[0])
            contour_AL_bp_temp2.append(dist_AL)
            count_AL_bp_temp2.append(lenc)
        
        rGy_AL_bp_temp1.append(rGy_AL_bp_temp2)
        contour_AL_bp_temp1.append(contour_AL_bp_temp2)
        count_AL_bp_temp1.append(count_AL_bp_temp2)
    
    rGy_AL_bp.append(rGy_AL_bp_temp1)
    contour_AL_bp.append(contour_AL_bp_temp1)
    count_AL_bp.append(count_AL_bp_temp1)

rGy_calyx_bp_avg = []
contour_calyx_bp_avg = []
count_calyx_bp_avg = []

for i in range(len(un_calyx)):
    if len(rGy_calyx_bp[i]) > 0:
        rGy_calyx_bp_avg.append(np.average(rGy_calyx_bp[i], axis=0))
        contour_calyx_bp_avg.append(np.average(contour_calyx_bp[i], axis=0))
        count_calyx_bp_avg.append(np.average(count_calyx_bp[i], axis=0))

rGy_LH_bp_avg = []
contour_LH_bp_avg = []
count_LH_bp_avg = []

for i in range(len(un_LH)):
    if len(rGy_LH_bp[i]) > 0:
        rGy_LH_bp_avg.append(np.average(rGy_LH_bp[i], axis=0))
        contour_LH_bp_avg.append(np.average(contour_LH_bp[i], axis=0))
        count_LH_bp_avg.append(np.average(count_LH_bp[i], axis=0))

rGy_AL_bp_avg = []
contour_AL_bp_avg = []
count_AL_bp_avg = []

for i in range(len(un_AL)):
    if len(rGy_AL_bp[i]) > 0:
        rGy_AL_bp_avg.append(np.average(rGy_AL_bp[i], axis=0))
        contour_AL_bp_avg.append(np.average(contour_AL_bp[i], axis=0))
        count_AL_bp_avg.append(np.average(count_AL_bp[i], axis=0))


rGy_calyx_bp_avg_avg = np.average(rGy_calyx_bp_avg, axis=0)
rGy_LH_bp_avg_avg = np.average(rGy_LH_bp_avg, axis=0)
rGy_AL_bp_avg_avg = np.average(rGy_AL_bp_avg, axis=0)

contour_calyx_bp_avg_avg = np.average(contour_calyx_bp_avg, axis=0)
contour_LH_bp_avg_avg = np.average(contour_LH_bp_avg, axis=0)
contour_AL_bp_avg_avg = np.average(contour_AL_bp_avg, axis=0)

count_calyx_bp_avg_avg = np.average(count_calyx_bp_avg, axis=0)
count_LH_bp_avg_avg = np.average(count_LH_bp_avg, axis=0)
count_AL_bp_avg_avg = np.average(count_AL_bp_avg, axis=0)

#%% Rgy centered at BP plotting

radiussize_q = radiussize

fig = plt.figure(figsize=(8,6))
for i in range(len(rGy_calyx_bp_avg)):
    plt.scatter(radiussize_q[np.nonzero(contour_calyx_bp_avg[i])], rGy_calyx_bp_avg[i][np.nonzero(contour_calyx_bp_avg[i])], marker='.')
plt.xscale('log')
plt.yscale('log')
plt.show()

fig = plt.figure(figsize=(8,6))
for i in range(len(rGy_LH_bp_avg)):
    plt.scatter(radiussize_q[np.nonzero(contour_LH_bp_avg[i])], rGy_LH_bp_avg[i][np.nonzero(contour_LH_bp_avg[i])], marker='.')
plt.xscale('log')
plt.yscale('log')
plt.show()

fig = plt.figure(figsize=(8,6))
for i in range(len(rGy_AL_bp_avg)):
    plt.scatter(radiussize_q[np.nonzero(contour_AL_bp_avg[i])], rGy_AL_bp_avg[i][np.nonzero(contour_AL_bp_avg[i])], marker='.')
plt.xscale('log')
plt.yscale('log')
plt.show()



#%% Rgy centered at BP average

fig = plt.figure(figsize=(8,6))
plt.plot(np.average(count_AL_bp_avg, axis=0), np.average(rGy_AL_bp_avg, axis=0), marker='.', color='tab:blue')
plt.fill_between(np.average(count_AL_bp_avg, axis=0), 
                 np.average(rGy_AL_bp_avg, axis=0)+np.std(rGy_AL_bp_avg, axis=0),
                 np.average(rGy_AL_bp_avg, axis=0)-np.std(rGy_AL_bp_avg, axis=0),
                 alpha=0.3,
                 color='tab:blue')
plt.plot(np.average(count_calyx_bp_avg, axis=0), np.average(rGy_calyx_bp_avg, axis=0), marker='.', color='tab:orange')
plt.fill_between(np.average(count_calyx_bp_avg, axis=0), 
                 np.average(rGy_calyx_bp_avg, axis=0)+np.std(rGy_calyx_bp_avg, axis=0),
                 np.average(rGy_calyx_bp_avg, axis=0)-np.std(rGy_calyx_bp_avg, axis=0),
                 alpha=0.3,
                 color='tab:orange')
plt.plot(np.average(count_LH_bp_avg, axis=0), np.average(rGy_LH_bp_avg, axis=0), marker='.', color='tab:green')
plt.fill_between(np.average(count_LH_bp_avg, axis=0), 
                 np.average(rGy_LH_bp_avg, axis=0)+np.std(rGy_LH_bp_avg, axis=0),
                 np.average(rGy_LH_bp_avg, axis=0)-np.std(rGy_LH_bp_avg, axis=0),
                 alpha=0.3,
                 color='tab:green')
plt.xscale('log')
plt.yscale('log')
plt.legend(["AL", "MB calyx", "LH"], fontsize=13)
plt.xlabel("L", fontsize=15)
plt.ylabel(r"$R_{g}$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Rgy_BP_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%

mw_rGy_calyx_bp_avg = []
mw_rGy_calyx_bp_avg_err = []
mwx_calyx = []
shiftN = 5

for i in range(len(contour_calyx_bp_avg_avg) - shiftN):
    mwx_calyx.append(np.average(contour_calyx_bp_avg_avg[i:i+shiftN]))
    
    poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(contour_calyx_bp_avg_avg[i:i+shiftN]), 
                                                np.log10(rGy_calyx_bp_avg_avg[i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    mw_rGy_calyx_bp_avg.append(poptmxc[0])
    mw_rGy_calyx_bp_avg_err.append(np.sqrt(np.diag(pcovmxc))[0])

mw_rGy_LH_bp_avg = []
mw_rGy_LH_bp_avg_err = []
mwx_LH = []

for i in range(len(contour_LH_bp_avg_avg) - shiftN):
    mwx_LH.append(np.average(contour_LH_bp_avg_avg[i:i+shiftN]))
    
    poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(contour_LH_bp_avg_avg[i:i+shiftN]), 
                                                np.log10(rGy_LH_bp_avg_avg[i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    mw_rGy_LH_bp_avg.append(poptmxc[0])
    mw_rGy_LH_bp_avg_err.append(np.sqrt(np.diag(pcovmxc))[0])

mw_rGy_AL_bp_avg = []
mw_rGy_AL_bp_avg_err = []
mwx_AL = []

for i in range(len(contour_AL_bp_avg_avg) - shiftN):
    mwx_AL.append(np.average(contour_AL_bp_avg_avg[i:i+shiftN]))
    
    poptmxc, pcovmxc = scipy.optimize.curve_fit(objFuncGL, 
                                                np.log10(contour_AL_bp_avg_avg[i:i+shiftN]), 
                                                np.log10(rGy_AL_bp_avg_avg[i:i+shiftN]), 
                                                p0=[1., 0.], 
                                                maxfev=100000)
    mw_rGy_AL_bp_avg.append(poptmxc[0])
    mw_rGy_AL_bp_avg_err.append(np.sqrt(np.diag(pcovmxc))[0])

    

fig = plt.figure(figsize=(8,6))
plt.plot(1/np.array(mwx_AL), np.array(mw_rGy_AL_bp_avg), lw=2)
plt.plot(1/np.array(mwx_calyx), np.array(mw_rGy_calyx_bp_avg), lw=2)
plt.plot(1/np.array(mwx_LH), np.array(mw_rGy_LH_bp_avg), lw=2)
plt.fill_between(1/np.array(mwx_AL), 
                 (np.array(mw_rGy_AL_bp_avg)-np.array(mw_rGy_AL_bp_avg_err)), 
                 (np.array(mw_rGy_AL_bp_avg)+np.array(mw_rGy_AL_bp_avg_err)), 
                 alpha=0.3)
plt.fill_between(1/np.array(mwx_calyx), 
                 (np.array(mw_rGy_calyx_bp_avg)-np.array(mw_rGy_calyx_bp_avg_err)),
                 (np.array(mw_rGy_calyx_bp_avg)+np.array(mw_rGy_calyx_bp_avg_err)), 
                 alpha=0.3)
plt.fill_between(1/np.array(mwx_LH),
                 (np.array(mw_rGy_LH_bp_avg)-np.array(mw_rGy_LH_bp_avg_err)),
                 (np.array(mw_rGy_LH_bp_avg)+np.array(mw_rGy_LH_bp_avg_err)), 
                 alpha=0.3)

plt.hlines(1/4, 0.001, 30, ls='dashed')
plt.hlines(7/16, 0.001, 30, ls='dashed')
plt.hlines(1/2, 0.001, 30, ls='dashed')
plt.hlines(1, 0.001, 30, ls='dashed')
plt.text(30.5, 1/4-0.01, 'Ideal')
plt.text(30.5, 7/16-0.01, '$\Theta$ Solvent')
plt.text(30.5, 1/2-0.01, 'Random')
plt.text(30.5, 1-0.01,' Rigid')

plt.vlines(1/np.mean(AL_length_temp), 1e-6, 10, color='tab:blue')
plt.vlines(1/np.mean(calyx_length_temp), 1e-6, 10, color='tab:orange')
plt.vlines(1/np.mean(LH_length_temp), 1e-6, 10, color='tab:green')

plt.vlines(1/np.median(AL_length_temp), 1e-6, 10, color='tab:blue', ls='dotted')
plt.vlines(1/np.median(calyx_length_temp), 1e-6, 10, color='tab:orange', ls='dotted')
plt.vlines(1/np.median(LH_length_temp), 1e-6, 10, color='tab:green', ls='dotted')

plt.vlines(1/rgy_AL_full[0], 1e-6, 10, color='tab:blue', ls='--')
plt.vlines(1/rgy_calyx_full[0], 1e-6, 10, color='tab:orange', ls='--')
plt.vlines(1/rgy_LH_full[0], 1e-6, 10, color='tab:green', ls='--')

plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.1, 1.5)
plt.xlim(0.001, 30)

plt.legend(["AL", "MB calyx", "LH"], fontsize=13)
plt.xlabel("$1/L$", fontsize=15)
plt.ylabel(r"$\nu$", fontsize=15)
# plt.savefig(Parameter.outputdir + '/Rg_all_mv_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%% Segmentation process diagram

nidx_list = [6, 8, 11, 12, 13, 18, 19, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 
             33, 34, 35, 36, 37, 38, 39, 53, 54, 57, 63, 64, 66, 67, 68, 69, 
             74, 75, 77, 78, 79, 80, 81]

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.set_xlim(440, 610)
ax.set_ylim(375, 205)
ax.set_zlim(45, 215)
cmap = cm.get_cmap('viridis', len(nidx_list))

for i, nidx in enumerate(nidx_list):
    
    for p in range(len(MorphData.morph_parent[nidx])):
        if MorphData.morph_parent[nidx][p] < 0:
            pass
        else:
            morph_line = np.vstack((MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(MorphData.morph_parent[nidx][p])], 
                                    MorphData.morph_dist[nidx][p]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.75)

ax.grid(True)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# plt.savefig(Parameter.outputdir + '/spd_neuron_all.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


#%% Rotated segmentation process diagram


r_d_x = -10
r_rad_x = np.radians(r_d_x)
r_x = np.array([0, 1, 0])
r_vec_x = r_rad_x * r_x
rotx = Rotation.from_rotvec(r_vec_x)

r_d_y = -25
r_rad_y = np.radians(r_d_y)
r_y = np.array([0, 1, 0])
r_vec_y = r_rad_y * r_y
roty = Rotation.from_rotvec(r_vec_y)

r_d_z = -40
r_rad_z = np.radians(r_d_z)
r_z = np.array([0, 1, 0])
r_vec_z = r_rad_z * r_z
rotz = Rotation.from_rotvec(r_vec_z)

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.set_xlim(-55, 115)
ax.set_ylim(330, 160)
ax.set_zlim(420, 590)
cmap = cm.get_cmap('viridis', len(nidx_list))

for i, nidx in enumerate(nidx_list):
    
    for p in range(len(MorphData.morph_parent[nidx])):
        if MorphData.morph_parent[nidx][p] < 0:
            pass
        else:
            morph_line = np.vstack((MorphData.morph_dist[nidx][MorphData.morph_id[nidx].index(MorphData.morph_parent[nidx][p])], 
                                    MorphData.morph_dist[nidx][p]))
            morph_line = rotx.apply(morph_line)
            morph_line = roty.apply(morph_line)
            morph_line = rotz.apply(morph_line)
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.75)

ax.grid(True)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# plt.savefig(Parameter.outputdir + '/spd_neuron_all_rotated.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

#%% Segmentation process diagram 2

from scipy.signal import argrelextrema

r_d_x = -10
r_rad_x = np.radians(r_d_x)
r_x = np.array([0, 1, 0])
r_vec_x = r_rad_x * r_x
rotx = Rotation.from_rotvec(r_vec_x)

r_d_y = -25
r_rad_y = np.radians(r_d_y)
r_y = np.array([0, 1, 0])
r_vec_y = r_rad_y * r_y
roty = Rotation.from_rotvec(r_vec_y)

r_d_z = -40
r_rad_z = np.radians(r_d_z)
r_z = np.array([0, 1, 0])
r_vec_z = r_rad_z * r_z
rotz = Rotation.from_rotvec(r_vec_z)

sel_morph_dist_flat = np.array([item for sublist in np.array(MorphData.morph_dist,dtype=object)[nidx_list] for item in sublist])

morph_dist_flat_rot = rotx.apply(sel_morph_dist_flat)
calyxdist_flat_rot = rotx.apply(calyx_dist_flat)
LHdist_flat_rot = rotx.apply(LH_dist_flat)
ALdist_flat_rot = rotx.apply(AL_dist_flat)

morph_dist_flat_rot = roty.apply(morph_dist_flat_rot)
# calyxdist_flat_rot = roty.apply(calyxdist_flat)
# LHdist_flat_rot = roty.apply(LHdist_flat)
# ALdist_flat_rot = roty.apply(ALdist_flat)

morph_dist_flat_rot = rotz.apply(morph_dist_flat_rot)

x = np.histogram(morph_dist_flat_rot[:,0], bins=int((np.max(morph_dist_flat_rot[:,0]) - np.min(morph_dist_flat_rot[:,0]))/1), density=True)
y = np.histogram(morph_dist_flat_rot[:,1], bins=int((np.max(morph_dist_flat_rot[:,1]) - np.min(morph_dist_flat_rot[:,1]))/1), density=True)
z = np.histogram(morph_dist_flat_rot[:,2], bins=int((np.max(morph_dist_flat_rot[:,2]) - np.min(morph_dist_flat_rot[:,2]))/1), density=True)

xex = argrelextrema(x[0], np.less)[0]
yex = argrelextrema(y[0], np.less)[0]
zex = argrelextrema(z[0], np.less)[0]


xval = np.linspace(min(morph_dist_flat_rot[:,0])-0.1, max(morph_dist_flat_rot[:,0])+0.1, 300)
kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=3).fit(morph_dist_flat_rot[:,0].reshape((len(morph_dist_flat_rot[:,0]),1)))
log_dens = kde.score_samples(xval.reshape((len(xval),1)))

fig = plt.figure(figsize=(8,6))
plt.plot(xval, np.exp(log_dens), lw=3)
# plt.xlabel('x Coordinates', fontsize=15)
# plt.ylabel('Count', fontsize=15)
# plt.legend(['All', 'AL', 'MB calyx', 'LH'], fontsize=13)
# plt.scatter(x[1][xex], x[0][xex], color='tab:red')
plt.xticks([])
plt.yticks([])
# plt.savefig(Parameter.outputdir + '/spd_x_segment_hist_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

yval = np.linspace(min(morph_dist_flat_rot[:,1])-0.1, max(morph_dist_flat_rot[:,1])+0.1, 300)
kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=3).fit(morph_dist_flat_rot[:,1].reshape((len(morph_dist_flat_rot[:,1]),1)))
log_dens = kde.score_samples(yval.reshape((len(yval),1)))

fig = plt.figure(figsize=(8,6))
plt.plot(yval, np.exp(log_dens), lw=3)
# plt.xlabel('y Coordinates', fontsize=15)
# plt.ylabel('Count', fontsize=15)
# plt.legend(['All', 'AL', 'MB calyx', 'LH'], fontsize=13)
# plt.scatter(y[1][yex[[9,26,46]]], y[0][yex[[9,26,46]]], color='tab:red')
plt.xticks([])
plt.yticks([])
# plt.savefig(Parameter.outputdir + '/spd_y_segment_hist_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

zval = np.linspace(min(morph_dist_flat_rot[:,2])-0.1, max(morph_dist_flat_rot[:,2])+0.1, 300)
kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=3).fit(morph_dist_flat_rot[:,2].reshape((len(morph_dist_flat_rot[:,2]),1)))
log_dens = kde.score_samples(zval.reshape((len(zval),1)))

fig = plt.figure(figsize=(8,6))
plt.plot(zval, np.exp(log_dens), lw=3)
# plt.xlabel('z Coordinates', fontsize=15)
# plt.ylabel('Count', fontsize=15)
# plt.legend(['All', 'AL', 'MB calyx', 'LH'], fontsize=13)
# plt.scatter(z[1][zex[[7,12,22,24,28]]], z[0][zex[[7,12,22,24,28]]], color='tab:red')
# plt.scatter(z[1][zex[[14]]], z[0][zex[[14]]], color='tab:red')
plt.xticks([])
plt.yticks([])
# plt.savefig(Parameter.outputdir + '/spd_z_segment_hist_2.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Segmentation process diagram 3

from scipy.signal import argrelextrema

sel_morph_dist_flat = np.array([item for sublist in np.array(MorphData.morph_dist,dtype=object)[nidx_list] for item in sublist])

cmap = cm.get_cmap('plasma', 19)

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')

for i, deg in enumerate(np.arange(-90, 91, 10)):
    r_d_x = -deg
    r_rad_x = np.radians(r_d_x)
    r_x = np.array([0, 0, 1])
    r_vec_x = r_rad_x * r_x
    rotx = Rotation.from_rotvec(r_vec_x)
    
    morph_dist_flat_rot = rotx.apply(sel_morph_dist_flat)

    x = np.histogram(morph_dist_flat_rot[:,0], bins=int((np.max(morph_dist_flat_rot[:,0]) - np.min(morph_dist_flat_rot[:,0]))/1), density=True)

    xval = np.linspace(min(morph_dist_flat_rot[:,0])-0.1, max(morph_dist_flat_rot[:,0])+0.1, 300)
    kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=3).fit(morph_dist_flat_rot[:,0].reshape((len(morph_dist_flat_rot[:,0]),1)))
    log_dens = kde.score_samples(xval.reshape((len(xval),1)))

    ax.plot3D(np.arange(len(xval)), np.repeat(-deg,len(xval)), np.exp(log_dens), color=cmap(i), lw=3)
    # ax.add_collection3d(plt.fill_between(np.arange(len(xval)), np.exp(log_dens), color=cmap(i), alpha=1), zs=-deg, zdir='y')

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.grid(True)
# plt.savefig(Parameter.outputdir + '/spd_x_multiple_hist_2.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')

for i, deg in enumerate(np.arange(-90, 91, 10)):
    r_d_y = -deg
    r_rad_y = np.radians(r_d_y)
    r_y = np.array([1, 0, 0])
    r_vec_y = r_rad_y * r_y
    roty = Rotation.from_rotvec(r_vec_y)
    
    morph_dist_flat_rot = roty.apply(sel_morph_dist_flat)

    y = np.histogram(morph_dist_flat_rot[:,1], bins=int((np.max(morph_dist_flat_rot[:,1]) - np.min(morph_dist_flat_rot[:,1]))/1), density=True)

    yval = np.linspace(min(morph_dist_flat_rot[:,1])-0.1, max(morph_dist_flat_rot[:,1])+0.1, 300)
    kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=3).fit(morph_dist_flat_rot[:,1].reshape((len(morph_dist_flat_rot[:,1]),1)))
    log_dens = kde.score_samples(yval.reshape((len(yval),1)))

    ax.plot3D(np.repeat(-deg,len(yval)), np.arange(len(yval)), np.exp(log_dens), color=cmap(i), lw=3)
    # ax.add_collection3d(plt.fill_between(np.arange(len(yval)), np.exp(log_dens), color=cmap(i), alpha=1), zs=-deg, zdir='y')

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.grid(True)
# plt.savefig(Parameter.outputdir + '/spd_y_multiple_hist_2.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')

for i, deg in enumerate(np.arange(-90, 91, 10)):
    r_d_z = -deg
    r_rad_z = np.radians(r_d_z)
    r_z = np.array([0, 1, 0])
    r_vec_z = r_rad_z * r_z
    rotz = Rotation.from_rotvec(r_vec_z)
    
    morph_dist_flat_rot = rotz.apply(sel_morph_dist_flat)

    z = np.histogram(morph_dist_flat_rot[:,2], bins=int((np.max(morph_dist_flat_rot[:,2]) - np.min(morph_dist_flat_rot[:,2]))/1), density=True)

    zval = np.linspace(min(morph_dist_flat_rot[:,2])-0.1, max(morph_dist_flat_rot[:,2])+0.1, 300)
    kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=3).fit(morph_dist_flat_rot[:,2].reshape((len(morph_dist_flat_rot[:,2]),1)))
    log_dens = kde.score_samples(zval.reshape((len(zval),1)))

    ax.plot3D(np.exp(log_dens), np.repeat(-deg,len(zval)), np.arange(len(zval)), color=cmap(i), lw=3)
    # ax.add_collection3d(plt.fill_between(np.arange(len(zval)), np.exp(log_dens), color=cmap(i), alpha=1), zs=-deg, zdir='y')

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.grid(True)
# plt.savefig(Parameter.outputdir + '/spd_z_multiple_hist_2.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()




#%% Segmentation process diagram 4

from scipy.signal import argrelextrema

sel_morph_dist_flat = np.array([item for sublist in np.array(MorphData.morph_dist,dtype=object)[nidx_list] for item in sublist])

xminlist = []
yminlist = []
zminlist = []

for i, degi in enumerate(np.arange(-90, 91, 10)):
    xminlist_i = []
    yminlist_i = []
    zminlist_i = []
    
    r_d_x = -degi
    r_rad_x = np.radians(r_d_x)
    r_x = np.array([0, 1, 0])
    r_vec_x = r_rad_x * r_x
    rotx = Rotation.from_rotvec(r_vec_x)
    
    for j, degj in enumerate(np.arange(-90, 91, 10)):
        xminlist_j = []
        yminlist_j = []
        zminlist_j = []
        
        r_d_y = -degj
        r_rad_y = np.radians(r_d_y)
        r_y = np.array([0, 0, 1])
        r_vec_y = r_rad_y * r_y
        roty = Rotation.from_rotvec(r_vec_y)
        
        for k, degk in enumerate(np.arange(-90, 91, 10)):
            r_d_z = -degk
            r_rad_z = np.radians(r_d_z)
            r_z = np.array([1, 0, 0])
            r_vec_z = r_rad_z * r_z
            rotz = Rotation.from_rotvec(r_vec_z)
            
            morph_dist_flat_rot = rotx.apply(sel_morph_dist_flat)
            morph_dist_flat_rot = roty.apply(morph_dist_flat_rot)
            morph_dist_flat_rot = rotz.apply(morph_dist_flat_rot)

            x = np.histogram(morph_dist_flat_rot[:,0], bins=int((np.max(morph_dist_flat_rot[:,0]) - np.min(morph_dist_flat_rot[:,0]))/1))
            y = np.histogram(morph_dist_flat_rot[:,1], bins=int((np.max(morph_dist_flat_rot[:,1]) - np.min(morph_dist_flat_rot[:,1]))/1))
            z = np.histogram(morph_dist_flat_rot[:,2], bins=int((np.max(morph_dist_flat_rot[:,2]) - np.min(morph_dist_flat_rot[:,2]))/1))
    
            xex = argrelextrema(x[0], np.less)[0]
            yex = argrelextrema(y[0], np.less)[0]
            zex = argrelextrema(z[0], np.less)[0]
            
            xminlist_j.append(x[0][np.argmin(x[0][xex][5:-5])])
            yminlist_j.append(y[0][np.argmin(y[0][yex][5:-5])])
            zminlist_j.append(z[0][np.argmin(z[0][zex][5:-5])])
        
        xminlist_i.append(xminlist_j)
        yminlist_i.append(yminlist_j)
        zminlist_i.append(zminlist_j)
    
    xminlist.append(xminlist_i)
    yminlist.append(yminlist_i)
    zminlist.append(zminlist_i)

# for i in range(len(xminlist)):
#     plt.imshow(np.array(xminlist)[:,i,:], cmap='plasma', interpolation='nearest')
#     plt.show()

# plt.imshow(np.array(zminlist), cmap='plasma', interpolation='nearest')
# plt.show()

#%% Segmentation process diagram

nidx_list = [6, 8, 11, 12, 13, 18, 19, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 
             33, 34, 35, 36, 37, 38, 39, 53, 54, 57, 63, 64, 66, 67, 68, 69, 
             74, 75, 77, 78, 79, 80, 81]

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.set_xlim(440, 610)
ax.set_ylim(375, 205)
ax.set_zlim(45, 215)
cmap = cm.get_cmap('viridis', len(nidx_list))

for i, nidx in enumerate(nidx_list):
    for p in range(len(MorphData.calyxdist_per_n[nidx])):
        morph_line = np.array(MorphData.calyxdist_per_n[nidx][p])
        ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.75)
    for p in range(len(MorphData.LHdist_per_n[nidx])):
        morph_line = np.array(MorphData.LHdist_per_n[nidx][p])
        ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.75)
    for p in range(len(MorphData.ALdist_per_n[nidx])):
        morph_line = np.array(MorphData.ALdist_per_n[nidx][p])
        ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.75)
    

ax.grid(True)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# plt.savefig(Parameter.outputdir + '/spd_neuron_all_seg.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


