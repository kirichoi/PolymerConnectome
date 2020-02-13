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
from matplotlib import cm
import matplotlib.patches as mpatches
import seaborn
import pandas as pd
import scipy.optimize
from collections import Counter
import multiprocessing as mp
import networkx as nx
import copy
import time
import utils

class Parameter:

    PATH = r'./TEMCA2/Skels connectome'
    
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
    
    outputdir = './output_TEMCA/RN_' + str(RN)

fp = [f for f in os.listdir(Parameter.PATH) if os.path.isfile(os.path.join(Parameter.PATH, f))]
fp = [os.path.join(Parameter.PATH, f) for f in fp]

fp.pop(17)
#fp = fp[:5]

class MorphData():
    
    def __init__(self):
        self.morph_id = []
        self.morph_parent = []
        self.morph_dist = []
        self.neuron_id = []
        self.endP = []
        self.somaP = []
        self.indMorph_dist = []
    
    def plotNeuronFromPoints(self, listOfPoints, showPoint=False):
        """
        plot 3-D neuron morphology plot using a list of coordinates.
        
        :param listOfPoints: List of 3-D coordinates
        :param showPoint: Flag to visualize points
        """
        
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
    #    ax.set_xlim(-300, 300)
    #    ax.set_ylim(-150, 150)
    #    ax.set_zlim(-300, 300)
        cmap = cm.get_cmap('viridis', len(listOfPoints))
        for f in range(len(listOfPoints)-1):
    #        tararr = np.array(morph_dist[f])
    #        somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(f))
            if showPoint:
                ax.scatter3D(listOfPoints[f][0], listOfPoints[f][1], listOfPoints[f][2], color=cmap(f), marker='x')
    #        ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(f))
        plt.show()
        
    
    def plotAllNeuron(self, showPoint=False):
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        ax.set_xlim(-300, 300)
        ax.set_ylim(-150, 150)
        ax.set_zlim(-300, 300)
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
        

    def plotNeuronFromListPoints(self, multListOfPoints, scale=False, showPoint=False):
        """
        plot 3-D neuron morphology plot using a list of coordinates.
        
        :param listOfPoints: List of 3-D coordinates
        :param showPoint: Flag to visualize points
        """
        
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        if scale:
            ax.set_xlim(-300, 300)
            ax.set_ylim(-150, 150)
            ax.set_zlim(-300, 300)
        cmap = cm.get_cmap('viridis', len(multListOfPoints))
        for i in range(len(multListOfPoints)):
            listOfPoints = self.indMorph_dist[multListOfPoints[i][0]][multListOfPoints[i][1]:multListOfPoints[i][2]]
            for f in range(len(listOfPoints)-1):
        #        tararr = np.array(morph_dist[f])
        #        somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
                morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i))
                if showPoint:
                    ax.scatter3D(listOfPoints[f][0], listOfPoints[f][1], listOfPoints[f][2], color=cmap(i), marker='x')
        #        ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(f))
        plt.show()
        
        
    def plotNeuron(self, idx, scale=False, cmass=False, showPoint=False, lw=1, label=True):
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        if scale:
            ax.set_xlim(-300, 300)
            ax.set_ylim(-150, 150)
            ax.set_zlim(-300, 300)
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
        if label:
            plt.title(np.array(self.neuron_id)[idx], fontsize=15)
        plt.show()
        
            
    def plotProjection(self, idx, project='z', scale=False, customBound=None, lw=1, label=True):
        if project != 'x' and project != 'y' and project != 'z':
            raise(Exception("Unrecognized plane to project"))
        fig = plt.figure(figsize=(24, 16))
        if scale:
            if project == 'z':
                plt.xlim(-50, 50)
                plt.ylim(-300, 450)
            elif project == 'y':
                plt.xlim(-100, 100)
                plt.ylim(-100, 100)
            else:
                plt.xlim(-400, 475)
                plt.ylim(-200, 200)
        
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
        plt.show()
    
    
    def _spat_pos(self, nodeList):
        pos = {}
        
        for i in range(len(nodeList)):
            pos[nodeList[i]] = np.array([self.morph_dist[i][self.somaP[i]][1],
                                         self.morph_dist[i][self.somaP[i]][0]])
    
        return pos
        


    
class LengthData:
    length_total = np.empty(len(fp))
    length_branch = []
    length_direct = []
    indMDistLen = []
    indMDistN = []
    
class BranchData:
    branchTrk = []
    branch_dist = []
    indBranchTrk = []
    branchP = []
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

for f in range(len(fp)):
    print(f)
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
    BranchData.branchTrk.append(neu_branchTrk)
    BranchData.branch_dist.append(branch_dist_temp1)
    LengthData.length_branch.append(length_branch_temp)

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
#MorphData.morph_dist_len_EP = np.empty((len(MorphData.morph_dist_len)))


t2 = time.time()

print('checkpoint 2: ' + str(t2-t1))

#(MorphData.regMDist, MorphData.regMDistLen) = utils.segmentMorph(Parameter, BranchData)
#(MorphData.indRegMDist, MorphData.indMDistN) = utils.indSegmentMorph(Parameter, BranchData)


(rGy, cML) = utils.radiusOfGyration(MorphData)

#(rGyEP, cMLEP) = utils.endPointRadiusOfGyration(MorphData, BranchData)

t3 = time.time()

print('checkpoint 3: ' + str(t3-t2))

#(rGyReg, cMLReg) = utils.regularRadiusOfGyration(MorphData.regMDist, MorphData.regMDistLen)

t4 = time.time()

print('checkpoint 4: ' + str(t4-t3))

if Parameter.RUN:
    (OutputData.rGySeg, 
     OutputData.cMLSeg, 
     OutputData.segOrdN, 
     OutputData.randTrk) = utils.regularSegmentRadiusOfGyration(Parameter,
                                                                 BranchData,
                                                                 np.array(MorphData.indMorph_dist_flat), 
                                                                 LengthData.indMDistN, 
                                                                 numScaleSample=Parameter.numScaleSample,
                                                                 stochastic=True,
                                                                 p=indMorph_dist_id)
    if Parameter.SAVE:
        utils.exportMorph(Parameter, t4-t0, MorphData, BranchData, LengthData)

t5 = time.time()

print('checkpoint 5: ' + str(t5-t4))


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
    
    
    branchEndPDict = {'branch': BranchData.branchNum, 'endP': MorphData.endP_len}
    branchEndPDF = pd.DataFrame(data=branchEndPDict)
    fig = plt.figure(figsize=(8,6))
    seaborn.swarmplot(x='branch', y='endP', data=branchEndPDF)
    plt.title("Distribution of Number of Endpoints\n for Given Number of Branches", fontsize=20)
    plt.xlabel("Number of Branches", fontsize=15)
    plt.ylabel("Number of Endpoints", fontsize=15)
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
    plt.scatter(LengthData.length_total, rGy)
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
                                            np.log10(np.sqrt(np.square(rGy))), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    perrR = np.sqrt(np.diag(pcovR))
    fitYR = objFuncPpow(LengthData.length_total, poptR[0], poptR[1])
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(LengthData.length_total, np.sqrt(np.square(rGy)))
    plt.plot(LengthData.length_total, fitYR, color='tab:red')
    plt.yscale('log')
    plt.xscale('log')
#    plt.xlim(10, 10000)
#    plt.ylim(7, 4000)
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
    nSize_lf = [(i + j)/2 for i, j in zip(nSize_l[:-1], nSize_l[1:])]
    
    rGyRegSeg_avg = np.empty(len(nSize_l)-1)
    for i in range(len(nSize_l)-1):
        RS_s = np.where((OutputData.segOrdLen <= nSize_l[i+1]) &
                        (OutputData.segOrdLen >= nSize_l[i]))[0]
        rGyRegSeg_avg[i] = np.average(OutputData.rGySeg[RS_s])
    
    poptRS1, pcovRS1 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(OutputData.segOrdLen), 
                                              np.log10(np.sqrt(np.square(OutputData.rGySeg))), 
                                              p0=[1., 0.], 
                                              maxfev=100000)
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
    ax1.scatter(nSize_lf, 
                np.sqrt(np.square(rGyRegSeg_avg)), 
                color='tab:orange')
    ax1.plot(np.unique(OutputData.segOrdLen), fitYRS1, color='tab:red', lw=2, linestyle='--')
#    ax1.plot(np.unique(OutputData.segOrdN[RS2])*Parameter.sSize, fitYRS2, color='tab:red', lw=2, linestyle='--')
#    ax1.plot(np.unique(OutputData.segOrdN[RS3])*Parameter.sSize, fitYRS3, color='tab:red', lw=2, linestyle='--')
#    ax1.vlines(0.8, 0.01, 11000, linestyles='dashed')
#    ax1.vlines(0.4, 0.01, 11000, linestyles='dashed')
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
            RS_x.append(np.average(nSize_l[i:i+shift_N]))
            
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
bcML = np.empty((len(mdistl_nz)*Parameter.numBranchSample, 3))
brGy = np.empty(len(mdistl_nz)*Parameter.numBranchSample)
cnt = 0

for m in mdistl_nz:
    bsrand = np.random.choice(BranchData.branchP[m], size=Parameter.numBranchSample, replace=False)
    bstrk_temp = []
    ibind_temp = []
    bstrk_len_temp = []
    for b in range(Parameter.numBranchSample):
        bstrkval_temp = []
        ibind_temp = []
        for k in range(len(BranchData.indBranchTrk[m])):
            bsw = np.where(np.array(BranchData.indBranchTrk[m][k]) == bsrand[b])[0]
            if len(bsw) > 0:
                bstrkval_temp.append(BranchData.indBranchTrk[m][k][:bsw[0]])
                ibind_temp.append(k)
    
        bstrkval_u = np.unique([item for sublist in bstrkval_temp for item in sublist])
        bstrk_temp.append(np.where(np.isin(MorphData.morph_id[m], bstrkval_u))[0])
        ibind_temp.append(ibind_temp)
    
    for r in range(len(bstrk_temp)):
        val = np.array(MorphData.morph_dist[m])[bstrk_temp[r]]
        x = val[:,0]
        y = val[:,1]
        z = val[:,2]
        
        xd = [j-i for i, j in zip(x[:-1], x[1:])]
        yd = [j-i for i, j in zip(y[:-1], y[1:])]
        zd = [j-i for i, j in zip(z[:-1], z[1:])]
        dist = np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
        bstrk_len_temp.append(dist)
    
    for i in range(Parameter.numBranchSample):
        bcML[cnt] = (np.sum(np.array(MorphData.morph_dist[m])[bstrk_temp[i]], 
                                                axis=0)/len(np.array(MorphData.morph_dist[m])[bstrk_temp[i]]))
        rList = scipy.spatial.distance.cdist(np.array(MorphData.morph_dist[m])[bstrk_temp[i]], 
                                             np.array([bcML[cnt]])).flatten()
        brGy[cnt] = np.sqrt(np.sum(np.square(rList))/len(rList))
        cnt += 1
    
    bstrk.append(bstrk_temp)
    ibind.append(ibind_temp)
    bstrk_len.append(bstrk_len_temp)


#%%
    
bstrk_len_flat = [item for sublist in bstrk_len for item in sublist]
    
#nSize_l = [0.1, 1, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 
#           300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550]
#nSize_lf = [(i + j)/2 for i, j in zip(nSize_l[:-1], nSize_l[1:])]
#
#rGyRegSeg_avg = np.empty(len(nSize_l)-1)
#for i in range(len(nSize_l)-1):
#    RS_s = np.where((OutputData.segOrdLen <= nSize_l[i+1]) &
#                    (OutputData.segOrdLen >= nSize_l[i]))[0]
#    rGyRegSeg_avg[i] = np.average(OutputData.rGySeg[RS_s])

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
#ax1.scatter(nSize_lf, 
#            np.sqrt(np.square(rGyRegSeg_avg)), 
#            color='tab:orange')
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



t7 = time.time()

print('checkpoint 7: ' + str(t7-t6))
