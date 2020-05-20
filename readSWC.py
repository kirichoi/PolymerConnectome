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
from sklearn import neighbors
from collections import Counter
import multiprocessing as mp
import time

os.chdir(os.path.dirname(__file__))

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
        self.LHdist = []
        self.ALdist = []
    
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
            ax.set_ylim(150, 400)
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
        plt.show()
        
    
    def plotAllNeuron(self, showPoint=False):
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        ax.set_xlim(400, 600)
        ax.set_ylim(150, 400)
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
        

    def plotNeuronFromListPoints(self, multListOfPoints, scale=False, showPoint=False):
        """
        plot 3-D neuron morphology plot using a list of coordinates.
        
        :param listOfPoints: List of 3-D coordinates
        :param showPoint: Flag to visualize points
        """
        
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        if scale:
            ax.set_xlim(400, 600)
            ax.set_ylim(150, 400)
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
        plt.show()
        
        
    def plotNeuron(self, idx, scale=False, cmass=False, showPoint=False, lw=1, label=True, show=True, save=False):
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        if scale:
            ax.set_xlim(400, 600)
            ax.set_ylim(150, 400)
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
            
        if save:
            plt.savefig(Parameter.outputdir + '/neuron_' + str(idx) + '.png', dpi=300, bbox_inches='tight')
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
        

#%%
    
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
                if ((np.array(branch_dist_temp2)[:,0] > 475).all() and (np.array(branch_dist_temp2)[:,0] < 550).all() and
                    (np.array(branch_dist_temp2)[:,1] < 260).all() and (np.array(branch_dist_temp2)[:,2] > 150).all()):
                    MorphData.calyxdist.append(branch_dist_temp2)
                elif ((np.array(branch_dist_temp2)[:,0] < 475).all() and (np.array(branch_dist_temp2)[:,1] < 260).all() and
                    (np.array(branch_dist_temp2)[:,1] > 180).all() and (np.array(branch_dist_temp2)[:,2] > 125).all()):
                    MorphData.LHdist.append(branch_dist_temp2)
                elif ((np.array(branch_dist_temp2)[:,0] > 475).all() and (np.array(branch_dist_temp2)[:,0] < 600).all() and 
                      (np.array(branch_dist_temp2)[:,1] > 280).all() and (np.array(branch_dist_temp2)[:,1] < 400).all() and
                      (np.array(branch_dist_temp2)[:,2] < 90).all()):
                    MorphData.ALdist.append(branch_dist_temp2)
                
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



#%% Cluster Center of Mass Calculation

calyxdist_flat = [item for sublist in MorphData.calyxdist for item in sublist]
LHdist_flat = [item for sublist in MorphData.LHdist for item in sublist]
ALdist_flat = [item for sublist in MorphData.ALdist for item in sublist]

calyxCM = (np.sum(np.array(calyxdist_flat), axis=0)/len(np.array(calyxdist_flat)))
LHCM = (np.sum(np.array(LHdist_flat), axis=0)/len(np.array(LHdist_flat)))
ALCM = (np.sum(np.array(ALdist_flat), axis=0)/len(np.array(ALdist_flat)))

fullCM = np.average(OutputData.cMLSeg, axis=0)

#%% Cluster Spread Calculation
        
radiussize = np.logspace(0, 2, 100)[20:85]

spheredist_calyx_sum = np.empty((len(MorphData.neuron_id), len(radiussize)))
spheredist_LH_sum = np.empty((len(MorphData.neuron_id), len(radiussize)))
spheredist_AL_sum = np.empty((len(MorphData.neuron_id), len(radiussize)))

for m in range(len(MorphData.neuron_id)):
    for b in range(len(radiussize)):
        spheredist_calyx_temp = []
        spheredist_LH_temp = []
        spheredist_AL_temp = []
        
        for ib in range(len(BranchData.branch_dist[m])):
            inbound_calyx = np.where(np.sqrt(np.square(np.array(BranchData.branch_dist[m][ib])[:,0] - calyxCM[0]) +
                                        np.square(np.array(BranchData.branch_dist[m][ib])[:,1] - calyxCM[1]) +
                                        np.square(np.array(BranchData.branch_dist[m][ib])[:,2] - calyxCM[2])) <= radiussize[b])[0]
            inbound_LH = np.where(np.sqrt(np.square(np.array(BranchData.branch_dist[m][ib])[:,0] - LHCM[0]) +
                                        np.square(np.array(BranchData.branch_dist[m][ib])[:,1] - LHCM[1]) +
                                        np.square(np.array(BranchData.branch_dist[m][ib])[:,2] - LHCM[2])) <= radiussize[b])[0]
            inbound_AL = np.where(np.sqrt(np.square(np.array(BranchData.branch_dist[m][ib])[:,0] - ALCM[0]) +
                                        np.square(np.array(BranchData.branch_dist[m][ib])[:,1] - ALCM[1]) +
                                        np.square(np.array(BranchData.branch_dist[m][ib])[:,2] - ALCM[2])) <= radiussize[b])[0]
            
            if len(inbound_calyx) > 1:
                val = np.array(BranchData.branch_dist[m][ib])[inbound_calyx]
                x = val[:,0]
                y = val[:,1]
                z = val[:,2]
                
                xd = [j-i for i, j in zip(x[:-1], x[1:])]
                yd = [j-i for i, j in zip(y[:-1], y[1:])]
                zd = [j-i for i, j in zip(z[:-1], z[1:])]
                dist_calyx = np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                spheredist_calyx_temp.append(dist_calyx)
            else:
                spheredist_calyx_temp.append(0)
                
            if len(inbound_LH) > 1:
                val = np.array(BranchData.branch_dist[m][ib])[inbound_LH]
                x = val[:,0]
                y = val[:,1]
                z = val[:,2]
                
                xd = [j-i for i, j in zip(x[:-1], x[1:])]
                yd = [j-i for i, j in zip(y[:-1], y[1:])]
                zd = [j-i for i, j in zip(z[:-1], z[1:])]
                dist_LH = np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                spheredist_LH_temp.append(dist_LH)
            else:
                spheredist_LH_temp.append(0)
                
            if len(inbound_AL) > 1:
                val = np.array(BranchData.branch_dist[m][ib])[inbound_AL]
                x = val[:,0]
                y = val[:,1]
                z = val[:,2]
                
                xd = [j-i for i, j in zip(x[:-1], x[1:])]
                yd = [j-i for i, j in zip(y[:-1], y[1:])]
                zd = [j-i for i, j in zip(z[:-1], z[1:])]
                dist_AL = np.sum(np.sqrt(np.square(xd) + np.square(yd) + np.square(zd)))
                spheredist_AL_temp.append(dist_AL)
            else:
                spheredist_AL_temp.append(0)
            
        spheredist_calyx_sum[m][b] = np.sum(spheredist_calyx_temp)
        spheredist_LH_sum[m][b] = np.sum(spheredist_LH_temp)
        spheredist_AL_sum[m][b] = np.sum(spheredist_AL_temp)

#%% 
   
radiussize_inv = radiussize#np.divide(1, 4/3*np.pi*np.power(radiussize, 3))

spheredist_calyx_sum[spheredist_calyx_sum == 0] = np.nan
spheredist_LH_sum[spheredist_LH_sum == 0] = np.nan
spheredist_AL_sum[spheredist_AL_sum == 0] = np.nan

spheredist_calyx_sum_avg = np.nanmean(spheredist_calyx_sum, axis=0)
spheredist_LH_sum_avg = np.nanmean(spheredist_LH_sum, axis=0)
spheredist_AL_sum_avg = np.nanmean(spheredist_AL_sum, axis=0)

#spheredist_calyx_sum_avg = spheredist_calyx_sum_avg[np.count_nonzero(~np.isnan(spheredist_calyx_sum), axis=0) >= 10]
#spheredist_LH_sum_avg = spheredist_LH_sum_avg[np.count_nonzero(~np.isnan(spheredist_LH_sum), axis=0) >= 10]
#spheredist_AL_sum_avg = spheredist_AL_sum_avg[np.count_nonzero(~np.isnan(spheredist_AL_sum), axis=0) >= 10]

poptD_calyx_all = []
poptD_LH_all = []
poptD_AL_all = []

poptD_calyx, pcovD_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                    np.log10(radiussize_inv[12:49]), 
                                                    np.log10(spheredist_calyx_sum_avg[12:49]), 
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_calyx = np.sqrt(np.diag(pcovD_calyx))

poptD_LH, pcovD_LH = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize_inv[12:49]), 
                                              np.log10(spheredist_LH_sum_avg[12:49]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_LH = np.sqrt(np.diag(pcovD_LH))

poptD_AL1, pcovD_AL1 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize_inv[29:]), 
                                              np.log10(spheredist_AL_sum_avg[29:]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL1 = np.sqrt(np.diag(pcovD_AL1))

poptD_AL2, pcovD_AL2 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize_inv[12:22]), 
                                              np.log10(spheredist_AL_sum_avg[12:22]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL2 = np.sqrt(np.diag(pcovD_AL2))


fitYD_calyx = objFuncPpow(radiussize_inv, poptD_calyx[0], poptD_calyx[1])
fitYD_LH = objFuncPpow(radiussize_inv, poptD_LH[0], poptD_LH[1])
fitYD_AL1 = objFuncPpow(radiussize_inv, poptD_AL1[0], poptD_AL1[1])
fitYD_AL2 = objFuncPpow(radiussize_inv, poptD_AL2[0], poptD_AL2[1])

fig = plt.figure(figsize=(12,8))

plt.scatter(radiussize_inv, 
                    spheredist_calyx_sum_avg, color='tab:blue', facecolors='none')
plt.scatter(radiussize_inv, 
                    spheredist_LH_sum_avg, color='tab:orange', facecolors='none')
plt.scatter(radiussize_inv[22:29], 
                    spheredist_AL_sum_avg[22:29], color='tab:green')
plt.scatter(radiussize_inv[29:], 
                    spheredist_AL_sum_avg[29:], color='tab:green', facecolors='none')
plt.scatter(radiussize_inv[:22], 
                    spheredist_AL_sum_avg[:22], color='tab:green', facecolors='none')

plt.plot(radiussize_inv, fitYD_calyx, lw=2, linestyle='--', color='tab:blue')
plt.plot(radiussize_inv, fitYD_LH, lw=2, linestyle='--', color='tab:orange')
plt.plot(radiussize_inv, fitYD_AL1, lw=2, linestyle='--', color='tab:green')
plt.plot(radiussize_inv, fitYD_AL2, lw=2, linestyle='--', color='tab:green')
plt.yscale('log')
plt.xscale('log')
plt.legend(['Calyx: ' + str(round(poptD_calyx[0], 3)) + '$\pm$' + str(round(perrD_calyx[0], 3)),
            'LH: ' + str(round(poptD_LH[0], 3)) + '$\pm$' + str(round(perrD_LH[0], 3)),
            'AL1: ' + str(round(poptD_AL1[0], 3)) + '$\pm$' + str(round(perrD_AL1[0], 3)),
            'AL2: ' + str(round(poptD_AL2[0], 3)) + '$\pm$' + str(round(perrD_AL2[0], 3))], fontsize=15)
#plt.xlim(1, 75)
#plt.ylim(3, 1500)
#plt.tight_layout()
plt.xlabel("Radius", fontsize=15)
plt.ylabel("Length", fontsize=15)
plt.show()


#%% Moving window

Calyxmw = []
Calyxmwerr = []
LHmw = []
LHmwerr = []
ALmw = []
ALmwerr = []
mwx = []
shiftN = 11
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
plt.plot(mwx, Calyxmw, lw=2)
plt.plot(mwx, LHmw, lw=2)
plt.plot(mwx, ALmw, lw=2)
plt.fill_between(mwx, np.array(Calyxmw)-np.array(Calyxmwerr), np.array(Calyxmw)+np.array(Calyxmwerr), alpha=0.3)
plt.fill_between(mwx, np.array(LHmw)-np.array(LHmwerr), np.array(LHmw)+np.array(LHmwerr), alpha=0.3)
plt.fill_between(mwx, np.array(ALmw)-np.array(ALmwerr), np.array(ALmw)+np.array(ALmwerr), alpha=0.3)
plt.xscale('log')
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

#dist_len_calyx_dim = dist_len_calyx_dim[:,~np.any(np.isnan(dist_len_calyx_dim), axis=0)]
#dist_len_LH_dim = dist_len_LH_dim[:,~np.any(np.isnan(dist_len_LH_dim), axis=0)]
#dist_len_AL_dim = dist_len_AL_dim[:,~np.any(np.isnan(dist_len_AL_dim), axis=0)]

dist_len_calyx_dim_avg = np.nanmean(dist_len_calyx_dim, axis=1)
dist_len_LH_dim_avg = np.nanmean(dist_len_LH_dim, axis=1)
dist_len_AL_dim_avg = np.nanmean(dist_len_AL_dim, axis=1)

#dist_len_calyx_dim_avg = dist_len_calyx_dim_avg[~np.isnan(dist_len_calyx_dim_avg)]
#dist_len_LH_dim_avg = dist_len_LH_dim_avg[~np.isnan(dist_len_LH_dim_avg)]
#dist_len_AL_dim_avg = dist_len_AL_dim_avg[~np.isnan(dist_len_AL_dim_avg)]

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

morph_dist_flat = np.array([item for sublist in MorphData.morph_dist for item in sublist])

xmax_all = np.max(morph_dist_flat[:,0])
xmin_all = np.min(morph_dist_flat[:,0])
ymax_all = np.max(morph_dist_flat[:,1])
ymin_all = np.min(morph_dist_flat[:,1])
zmax_all = np.max(morph_dist_flat[:,2])
zmin_all = np.min(morph_dist_flat[:,2])

hlist = []
hlist_count = []
hlist_numbox = []

for b in range(len(binsize)):
    xbin = np.arange(xmin_all, xmax_all+binsize[b], binsize[b])
    ybin = np.arange(ymin_all, ymax_all+binsize[b], binsize[b])
    zbin = np.arange(zmin_all, zmax_all+binsize[b], binsize[b])
    if len(xbin) == 1:
        xbin = [-1000, 1000]
    if len(ybin) == 1:
        ybin = [-1000, 1000]
    if len(zbin) == 1:
        zbin = [-1000, 1000]
        
    h, e = np.histogramdd(morph_dist_flat, 
                          bins=[xbin, 
                                ybin,
                                zbin])
    # hlist.append(h)
    hlist_count.append(np.count_nonzero(h))
    # hlist_numbox.append((len(xbin)-1)*
    #                     (len(ybin)-1)*
    #                     (len(zbin)-1))


#%%

poptBcount_all, pcovBcount_all = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[8:21]), 
                                                        np.log10(hlist_count[8:21]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_all = np.sqrt(np.diag(pcovBcount_all))

fitYBcount_all = objFuncPpow(binsize, poptBcount_all[0], poptBcount_all[1])
    
fig = plt.figure(figsize=(12,8))
plt.scatter(binsize, hlist_count)
plt.plot(binsize, fitYBcount_all, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['All: ' + str(round(poptBcount_all[0], 3)) + '$\pm$' + str(round(perrBcount_all[0], 3))], fontsize=15)
#plt.xlim(0.1, 20)
#plt.tight_layout()
plt.xlabel("Box Size", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()





#%% Fractal dimension using binary box counting for each region

binsize = np.logspace(-1, 3, 100)[13:90:3]

calyx_dist_flat = np.array([item for sublist in MorphData.calyxdist for item in sublist])
LH_dist_flat = np.array([item for sublist in MorphData.LHdist for item in sublist])
AL_dist_flat = np.array([item for sublist in MorphData.ALdist for item in sublist])

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

hlist_calyx = []
hlist_calyx_count = []
hlist_calyx_numbox = []
hlist_LH = []
hlist_LH_count = []
hlist_LH_numbox = []
hlist_AL = []
hlist_AL_count = []
hlist_AL_numbox = []

for b in range(len(binsize)):
    xbin_calyx = np.arange(xmin_calyx, xmax_calyx+binsize[b], binsize[b])
    ybin_calyx = np.arange(ymin_calyx, ymax_calyx+binsize[b], binsize[b])
    zbin_calyx = np.arange(zmin_calyx, zmax_calyx+binsize[b], binsize[b])
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
    # hlist_calyx.append(hc)
    hlist_calyx_count.append(np.count_nonzero(hc))
    # hlist_calyx_numbox.append((len(xbin_calyx)-1)*
    #                           (len(ybin_calyx)-1)*
    #                           (len(zbin_calyx)-1))
    
    xbin_LH = np.arange(xmin_LH, xmax_LH+binsize[b], binsize[b])
    ybin_LH = np.arange(ymin_LH, ymax_LH+binsize[b], binsize[b])
    zbin_LH = np.arange(zmin_LH, zmax_LH+binsize[b], binsize[b])
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
    # hlist_LH.append(hh)
    hlist_LH_count.append(np.count_nonzero(hh))
    # hlist_LH_numbox.append((len(xbin_LH)-1)*
    #                        (len(ybin_LH)-1)*
    #                        (len(zbin_LH)-1))
    
    xbin_AL = np.arange(xmin_AL, xmax_AL+binsize[b], binsize[b])
    ybin_AL = np.arange(ymin_AL, ymax_AL+binsize[b], binsize[b])
    zbin_AL = np.arange(zmin_AL, zmax_AL+binsize[b], binsize[b])
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
    # hlist_AL.append(ha)
    hlist_AL_count.append(np.count_nonzero(ha))
    # hlist_AL_numbox.append((len(xbin_AL)-1)*
    #                        (len(ybin_AL)-1)*
    #                        (len(zbin_AL)-1))




#%%
    
    
poptBcount_calyx, pcovBcount_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_calyx_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_calyx = np.sqrt(np.diag(pcovBcount_calyx))

poptBcount_LH, pcovBcount_LH = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_LH_count[7:20]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_LH = np.sqrt(np.diag(pcovBcount_LH))

poptBcount_AL, pcovBcount_AL = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:20]), 
                                                        np.log10(hlist_AL_count[7:20]),
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



#%% Binary Box-counting for Sub-physiological Region Length Scale

binsize = np.logspace(-1, 3, 100)[13:75:1]

sp_l = np.arange(5, 60, 1)
bbr = 5

hlist_calyx_b_count = np.empty((bbr, len(sp_l), len(binsize)), dtype=int)
hlist_LH_b_count = np.empty((bbr, len(sp_l), len(binsize)), dtype=int)
hlist_AL_b_count = np.empty((bbr, len(sp_l), len(binsize)), dtype=int)

for r in range(bbr):
    for l in range(len(sp_l)):
        
        calyx_rand = np.array([np.random.uniform(-(xmax_calyx - xmin_calyx)/20, (xmax_calyx - xmin_calyx)/20), 
                               np.random.uniform(-(ymax_calyx - ymin_calyx)/20, (ymax_calyx - ymin_calyx)/20),
                               np.random.uniform(-(zmax_calyx - zmin_calyx)/20, (zmax_calyx - zmin_calyx)/20)])
        max_calyx_b = calyxCM + sp_l[l] + calyx_rand
        min_calyx_b = calyxCM - sp_l[l] + calyx_rand
        
        LH_rand = np.array([np.random.uniform(-(xmax_LH - xmin_LH)/20, (xmax_LH - xmin_LH)/20), 
                            np.random.uniform(-(ymax_LH - ymin_LH)/20, (ymax_LH - ymin_LH)/20),
                            np.random.uniform(-(zmax_LH - zmin_LH)/20, (zmax_LH - zmin_LH)/20)])
        
        max_LH_b = LHCM + sp_l[l] + LH_rand
        min_LH_b = LHCM - sp_l[l] + LH_rand
        
        AL_rand = np.array([np.random.uniform(-(xmax_AL - xmin_AL)/20, (xmax_AL - xmin_AL)/20), 
                            np.random.uniform(-(ymax_AL - ymin_AL)/20, (ymax_AL - ymin_AL)/20),
                            np.random.uniform(-(zmax_AL - zmin_AL)/20, (zmax_AL - zmin_AL)/20)])
        
        max_AL_b = ALCM + sp_l[l] + AL_rand
        min_AL_b = ALCM - sp_l[l] + AL_rand
        
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
        farg = np.argwhere(np.array(hlist_calyx_b_count[r][l]) > 1)[-1][0]
        iarg = farg - 30
        if iarg < 0:
            iarg = 0
        poptBcount_calyx_b_t, pcovBcount_calyx_b_t = scipy.optimize.curve_fit(objFuncGL, 
                                                                np.log10(binsize[iarg:farg]), 
                                                                np.log10(hlist_calyx_b_count[r][l][iarg:farg]),
                                                                p0=[0.1, 0.1], 
                                                                maxfev=10000)
        perrBcount_calyx_b_t = np.sqrt(np.diag(pcovBcount_calyx_b_t))
        
        farg = np.argwhere(np.array(hlist_LH_b_count[r][l]) > 1)[-1][0]
        iarg = farg - 30
        if iarg < 0:
            iarg = 0
        poptBcount_LH_b_t, pcovBcount_LH_b_t = scipy.optimize.curve_fit(objFuncGL, 
                                                                np.log10(binsize[iarg:farg]), 
                                                                np.log10(hlist_LH_b_count[r][l][iarg:farg]),
                                                                p0=[0.1, 0.1], 
                                                                maxfev=10000)
        perrBcount_LH_b_t = np.sqrt(np.diag(pcovBcount_LH_b_t))
        
        farg = np.argwhere(np.array(hlist_AL_b_count[r][l]) > 1)[-1][0]
        iarg = farg - 30
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
    
fig = plt.figure(figsize=(12,8))
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

fig = plt.figure(figsize=(12,8))
plt.plot(sp_l, poptBcount_calyx_b_avg, lw=2, linestyle='--', color='tab:blue')
plt.plot(sp_l, poptBcount_LH_b_avg, lw=2, linestyle='--', color='tab:orange')
plt.plot(sp_l, poptBcount_AL_b_avg, lw=2, linestyle='--', color='tab:green')
plt.fill_between(sp_l, poptBcount_calyx_b_avg-perrBcount_calyx_b_avg, poptBcount_calyx_b_avg+perrBcount_calyx_b_avg, alpha=0.3)
plt.fill_between(sp_l, poptBcount_LH_b_avg-perrBcount_LH_b_avg, poptBcount_LH_b_avg+perrBcount_LH_b_avg, alpha=0.3)
plt.fill_between(sp_l, poptBcount_AL_b_avg-perrBcount_AL_b_avg, poptBcount_AL_b_avg+perrBcount_AL_b_avg, alpha=0.3)
plt.legend(['Calyx', 'LH', 'AL'], fontsize=15)
#plt.tight_layout()
plt.xlabel("Box Size", fontsize=15)
plt.ylabel("Dimension", fontsize=15)
plt.show()

t9 = time.time()

print('checkpoint 9: ' + str(t9-t8))

#%% Single Neuron Dimnesion Calculation using Binary Box-counting

binsize = np.logspace(-2, 3, 100)[25:95:3]

hlist_single = []
hlist_single_count = np.empty((len(MorphData.morph_dist), len(binsize)))
hlist_single_numbox = np.empty((len(MorphData.morph_dist), len(binsize)))

for i in range(len(MorphData.morph_dist)):
    morph_dist_single = np.array(MorphData.morph_dist[i])
    z
    xmax_single = np.max(morph_dist_single[:,0])
    xmin_single = np.min(morph_dist_single[:,0])
    ymax_single = np.max(morph_dist_single[:,1])
    ymin_single = np.min(morph_dist_single[:,1])
    zmax_single = np.max(morph_dist_single[:,2])
    zmin_single = np.min(morph_dist_single[:,2])
    
    for b in range(len(binsize)):
        xbin = np.arange(xmin_single, xmax_single+binsize[b], binsize[b])
        ybin = np.arange(ymin_single, ymax_single+binsize[b], binsize[b])
        zbin = np.arange(zmin_single, zmax_single+binsize[b], binsize[b])
        if len(xbin) == 1:
            xbin = [-1000, 1000]
        if len(ybin) == 1:
            ybin = [-1000, 1000]
        if len(zbin) == 1:
            zbin = [-1000, 1000]
            
        h, e = np.histogramdd(morph_dist_single, 
                              bins=[xbin, 
                                    ybin,
                                    zbin])
        hlist_single_count[i][b] = np.count_nonzero(h)
        # hlist_single_numbox[i][b] = ((len(xbin)-1)*
        #                              (len(ybin)-1)*
        #                              (len(zbin)-1))
   


#%%

cmap = cm.get_cmap('viridis', len(MorphData.morph_dist))

poptBcount_single_list = []
pcovBcount_single_list = []

fig = plt.figure(figsize=(12,8))
for i in range(len(MorphData.morph_dist)):
    farg = np.argwhere(hlist_single_count[i] > 1)[-1][0]
    iarg = farg - 5
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
#plt.ylim(0.1, 100000)
#plt.tight_layout()
plt.xlabel("Box Count", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()

poptBcount_single_all = np.sort(np.array(poptBcount_single_list)[:,0])[:-2]
xval = np.linspace(min(poptBcount_single_all)-0.1, max(poptBcount_single_all)+0.1, 300)

kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.05).fit(poptBcount_single_all.reshape((len(poptBcount_single_all),1)))

log_dens = kde.score_samples(xval.reshape((len(xval),1)))

fig = plt.figure(figsize=(12,8))
plt.hist(poptBcount_single_all, bins=int(len(hlist_single_count)/5), density=True)
plt.plot(xval, np.exp(log_dens), lw=3)
plt.vlines(xval[np.argmax(np.exp(log_dens))], 0, 5, linestyle='--', label=str(round(xval[np.argmax(np.exp(log_dens))], 3)), color='tab:red')
plt.ylim(0, 4.5)
plt.legend(fontsize=15)
plt.show()


t10 = time.time()

print('checkpoint 10: ' + str(t10-t9))

#%% Branching point and tip coordinate collection

BranchData.branchP_dist = []
MorphData.endP_dist = []
        
for i in range(len(BranchData.branchP)):
    branchP_dist_t = []
    for j in range(len(BranchData.branchP[i])):
        branchP_dist_t.append(MorphData.morph_dist[i][MorphData.morph_id[i].index(BranchData.branchP[i][j])])
    if len(branchP_dist_t) > 0:
        BranchData.branchP_dist.append(branchP_dist_t)
    
for i in range(len(MorphData.endP)):
    endP_dist_t = []
    for j in range(len(MorphData.endP[i])):
        endP_dist_t.append(MorphData.morph_dist[i][MorphData.morph_id[i].index(MorphData.endP[i][j])])
    if len(endP_dist_t) > 0:
        MorphData.endP_dist.append(endP_dist_t)

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
    if ((branchP_dist_flat[i][0] > 475).all() and (branchP_dist_flat[i][0] < 550).all() and
        (branchP_dist_flat[i][1] < 260).all() and (branchP_dist_flat[i][2] > 150).all()):
        branchP_calyx_dist.append(branchP_dist_flat[i])
    elif ((branchP_dist_flat[i][0] < 475).all() and (branchP_dist_flat[i][1] < 260).all() and
          (branchP_dist_flat[i][1] > 180).all() and (branchP_dist_flat[i][2] > 125).all()):
        branchP_LH_dist.append(branchP_dist_flat[i])
    elif ((branchP_dist_flat[i][0] > 475).all() and (branchP_dist_flat[i][0] < 600).all() and 
          (branchP_dist_flat[i][1] > 280).all() and (branchP_dist_flat[i][1] < 400).all() and
          (branchP_dist_flat[i][2] < 90).all()):
        branchP_AL_dist.append(branchP_dist_flat[i])


endP_calyx_dist = []
endP_LH_dist = []
endP_AL_dist = []

for i in range(len(endP_dist_flat)):
    if ((endP_dist_flat[i][0] > 475).all() and (endP_dist_flat[i][0] < 550).all() and
        (endP_dist_flat[i][1] < 260).all() and (endP_dist_flat[i][2] > 150).all()):
        endP_calyx_dist.append(endP_dist_flat[i])
    elif ((endP_dist_flat[i][0] < 475).all() and (endP_dist_flat[i][1] < 260).all() and
          (endP_dist_flat[i][1] > 180).all() and (endP_dist_flat[i][2] > 125).all()):
        endP_LH_dist.append(endP_dist_flat[i])
    elif ((endP_dist_flat[i][0] > 475).all() and (endP_dist_flat[i][0] < 600).all() and 
          (endP_dist_flat[i][1] > 280).all() and (endP_dist_flat[i][1] < 400).all() and
          (endP_dist_flat[i][2] < 90).all()):
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



#%% Regional dist categorization

glo_info = pd.read_excel(os.path.join(Parameter.PATH, '../all_skeletons_type_list_180919.xlsx'))

glo_list = []
glo_idx = []

for f in range(len(MorphData.neuron_id)):
    idx = np.where(glo_info.skid == int(MorphData.neuron_id[f]))[0][0]
    if 'glomerulus' in glo_info['old neuron name'][idx]:
        if glo_info['type'][idx] in glo_list:
            glo_idx[glo_list.index(glo_info['type'][idx])].append(f)
        else:
            glo_list.append(glo_info['type'][idx])
            glo_idx.append([f])

morph_dist_calyx = []
morph_dist_LH = []
morph_dist_AL = []

for i in range(len(glo_list)):
    morph_dist_calyx_temp = []
    morph_dist_LH_temp = []
    morph_dist_AL_temp = []
    for j in range(len(glo_idx[i])):
        morph_dist_calyx_temp2 = []
        morph_dist_LH_temp2 = []
        morph_dist_AL_temp2 = []
        for p in range(len(MorphData.morph_dist[glo_idx[i][j]])):
            if ((np.array(MorphData.morph_dist[glo_idx[i][j]][p])[0] > 475) and (np.array(MorphData.morph_dist[glo_idx[i][j]][p])[0] < 550) and
                (np.array(MorphData.morph_dist[glo_idx[i][j]][p])[1] < 260) and (np.array(MorphData.morph_dist[glo_idx[i][j]][p])[2] > 150)):
                morph_dist_calyx_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
            elif ((np.array(MorphData.morph_dist[glo_idx[i][j]][p])[0] < 475) and (np.array(MorphData.morph_dist[glo_idx[i][j]][p])[1] < 260) and
                (np.array(MorphData.morph_dist[glo_idx[i][j]][p])[1] > 180) and (np.array(MorphData.morph_dist[glo_idx[i][j]][p])[2] > 125)):
                morph_dist_LH_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
            elif ((np.array(MorphData.morph_dist[glo_idx[i][j]][p])[0] > 475) and (np.array(MorphData.morph_dist[glo_idx[i][j]][p])[0] < 600) and 
                  (np.array(MorphData.morph_dist[glo_idx[i][j]][p])[1] > 280) and (np.array(MorphData.morph_dist[glo_idx[i][j]][p])[1] < 400) and
                  (np.array(MorphData.morph_dist[glo_idx[i][j]][p])[2] < 90)):
                morph_dist_AL_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
        
        morph_dist_calyx_temp.append(morph_dist_calyx_temp2)
        morph_dist_LH_temp.append(morph_dist_LH_temp2)
        morph_dist_AL_temp.append(morph_dist_AL_temp2)
                
    morph_dist_calyx.append(morph_dist_calyx_temp)
    morph_dist_LH.append(morph_dist_LH_temp)
    morph_dist_AL.append(morph_dist_AL_temp)
            


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
    if np.isnan(morph_dist_LH_CM[f]).any():
        pass
    else:
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
plt.show()



#%% Cluster quantification

morph_dist_calyx_CM_flat = [item for sublist in morph_dist_calyx_CM for item in sublist]

morph_dist_calyx_r = scipy.spatial.distance.cdist(np.array(morph_dist_calyx_CM_flat), np.array(morph_dist_calyx_CM_flat))
calyxclusterstat = []
calyxdist_cluster_u_full = []
calyxdist_noncluster_u_full = []

idx = np.arange(len(morph_dist_calyx_r))
trk1 = 0
trk2 = 0

for f in range(len(glo_list)):
    dist_cluster = []
    dist_noncluster = []
    for i in range(len(morph_dist_calyx_CM[f])):
        dist_cluster.append(morph_dist_calyx_r[trk1:trk1+len(morph_dist_calyx_CM[f]),trk1:trk1+len(morph_dist_calyx_CM[f])])
        dist_noncluster.append(morph_dist_calyx_r[trk1:trk1+len(morph_dist_calyx_CM[f]),np.delete(idx, np.arange(trk1,trk1+len(morph_dist_calyx_CM[f])))])
    trk1 += len(morph_dist_calyx_CM[f])
    
    dist_cluster_u = np.unique(dist_cluster)
    dist_noncluster_u = np.unique(dist_noncluster)
    
    calyxdist_cluster_u_full.append(dist_cluster_u)
    calyxdist_noncluster_u_full.append(dist_noncluster_u)
    
    calyxclusterstat.append([np.mean(dist_cluster_u), np.std(dist_cluster_u), np.mean(dist_noncluster_u), np.std(dist_noncluster_u)])

calyxdist_cluster_u_full_flat = [item for sublist in calyxdist_cluster_u_full for item in sublist]
calyxdist_noncluster_u_full_flat = [item for sublist in calyxdist_noncluster_u_full for item in sublist]

calyxdist_cluster_u_full_flat = np.array(calyxdist_cluster_u_full_flat)[np.nonzero(calyxdist_cluster_u_full_flat)[0]]
calyxdist_noncluster_u_full_flat = np.array(calyxdist_noncluster_u_full_flat)[np.nonzero(calyxdist_noncluster_u_full_flat)[0]]

print("Calyx cluster Mean: " + str(np.mean(calyxdist_cluster_u_full_flat)) + ", STD: " + str(np.std(calyxdist_cluster_u_full_flat)))
print("Calyx noncluster Mean: " + str(np.mean(calyxdist_noncluster_u_full_flat)) + ", STD: " + str(np.std(calyxdist_noncluster_u_full_flat)))


morph_dist_LH_CM_flat = [item for sublist in morph_dist_LH_CM for item in sublist]
morph_dist_LH_CM_flat = [x for x in morph_dist_LH_CM_flat if str(x) != 'nan'] # Remove nan

morph_dist_LH_r = scipy.spatial.distance.cdist(np.array(morph_dist_LH_CM_flat), np.array(morph_dist_LH_CM_flat))
LHclusterstat = []
LHdist_cluster_u_full = []
LHdist_noncluster_u_full = []

idx = np.arange(len(morph_dist_LH_r))
trk1 = 0
trk2 = 0

for f in range(len(glo_list)-1):
    dist_cluster = []
    dist_noncluster = []
    for i in range(len(morph_dist_LH_CM[f])):
        dist_cluster.append(morph_dist_LH_r[trk1:trk1+len(morph_dist_LH_CM[f]),trk1:trk1+len(morph_dist_LH_CM[f])])
        dist_noncluster.append(morph_dist_LH_r[trk1:trk1+len(morph_dist_LH_CM[f]),np.delete(idx, np.arange(trk1,trk1+len(morph_dist_LH_CM[f])))])
    trk1 += len(morph_dist_LH_CM[f])
    
    dist_cluster_u = np.unique(dist_cluster)
    dist_noncluster_u = np.unique(dist_noncluster)
    
    LHdist_cluster_u_full.append(dist_cluster_u)
    LHdist_noncluster_u_full.append(dist_noncluster_u)
    
    LHclusterstat.append([np.mean(dist_cluster_u), np.std(dist_cluster_u), np.mean(dist_noncluster_u), np.std(dist_noncluster_u)])

LHdist_cluster_u_full_flat = [item for sublist in LHdist_cluster_u_full for item in sublist]
LHdist_noncluster_u_full_flat = [item for sublist in LHdist_noncluster_u_full for item in sublist]

LHdist_cluster_u_full_flat = np.array(LHdist_cluster_u_full_flat)[np.nonzero(LHdist_cluster_u_full_flat)[0]]
LHdist_noncluster_u_full_flat = np.array(LHdist_noncluster_u_full_flat)[np.nonzero(LHdist_noncluster_u_full_flat)[0]]

print("LH cluster Mean: " + str(np.mean(LHdist_cluster_u_full_flat)) + ", STD: " + str(np.std(LHdist_cluster_u_full_flat)))
print("LH noncluster Mean: " + str(np.mean(LHdist_noncluster_u_full_flat)) + ", STD: " + str(np.std(LHdist_noncluster_u_full_flat)))


#%%

fig, ax = plt.subplots()
labels = ['Calyx', 'LH']
x = np.arange(len(labels))  # the label locations
width = .3  # the width of the bars
cmeans = [np.mean(calyxdist_cluster_u_full_flat), np.mean(LHdist_cluster_u_full_flat)]
cerr = [np.std(calyxdist_cluster_u_full_flat), np.std(LHdist_cluster_u_full_flat)]
ncmeans = [np.mean(calyxdist_noncluster_u_full_flat), np.mean(LHdist_noncluster_u_full_flat)]
ncerr = [np.std(calyxdist_noncluster_u_full_flat), np.std(LHdist_noncluster_u_full_flat)]
ax.bar(x - width/2, cmeans, width, yerr=cerr, capsize=5, label='Cluster')
ax.bar(x + width/2, ncmeans, width, yerr=ncerr, capsize=5, label='Non-Cluster')
ax.set_ylabel('Distance')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()


#%% Cluster quantification heatmap visualization

from scipy.stats import kde
import logging
from mayavi import mlab

nbins=10
gi=1

morph_dist_calyx_n_flat = [item for sublist in morph_dist_calyx[gi] for item in sublist]

x = np.array(morph_dist_calyx_n_flat)[:,0]
y = np.array(morph_dist_calyx_n_flat)[:,1]
z = np.array(morph_dist_calyx_n_flat)[:,2]

xyz = np.vstack([x,y,z])
kdecalyx = kde.gaussian_kde(xyz)

# Evaluate kde on a grid
xi, yi, zi = np.mgrid[450:580:nbins*1j, 170:280:nbins*1j, 120:230:nbins*1j]
coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
density = kdecalyx(coords).reshape(xi.shape)

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot')

grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
mlab.pipeline.volume(grid, color=(0.2, 0.4, 0.5))

mlab.axes()
mlab.show()




#%% 2D heatmap of spatial distribution of each neuron in calyx, LH, and AL

from scipy.stats import kde

nbins=100
gi=1

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

