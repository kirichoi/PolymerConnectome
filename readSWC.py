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
import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost
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

t5 = time.time()

print('checkpoint 5: ' + str(t5-t4))





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



#%% Cluster Center of Mass Calculation

calyxdist_flat = [item for sublist in MorphData.calyxdist for item in sublist]
LHdist_flat = [item for sublist in MorphData.LHdist for item in sublist]
ALdist_flat = [item for sublist in MorphData.ALdist for item in sublist]

calyxCM = (np.sum(np.array(calyxdist_flat), axis=0)/len(np.array(calyxdist_flat)))
LHCM = (np.sum(np.array(LHdist_flat), axis=0)/len(np.array(LHdist_flat)))
ALCM = (np.sum(np.array(ALdist_flat), axis=0)/len(np.array(ALdist_flat)))

fullCM = np.average(OutputData.cMLSeg, axis=0)

#%% Cluster Spread Calculation
        
radiussize = np.logspace(0, 2, 100)[34:95]

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
                                                    np.log10(radiussize_inv[0:35]), 
                                                    np.log10(spheredist_calyx_sum_avg[0:35]), 
                                                    p0=[-0.1, 0.1], 
                                                    maxfev=10000)
perrD_calyx = np.sqrt(np.diag(pcovD_calyx))

poptD_LH, pcovD_LH = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize_inv[0:35]), 
                                              np.log10(spheredist_LH_sum_avg[0:35]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_LH = np.sqrt(np.diag(pcovD_LH))

poptD_AL1, pcovD_AL1 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize_inv[14:35]), 
                                              np.log10(spheredist_AL_sum_avg[14:35]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL1 = np.sqrt(np.diag(pcovD_AL1))

poptD_AL2, pcovD_AL2 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(radiussize_inv[0:9]), 
                                              np.log10(spheredist_AL_sum_avg[0:9]), 
                                              p0=[-0.1, 0.1], 
                                              maxfev=10000)
perrD_AL2 = np.sqrt(np.diag(pcovD_AL2))


fitYD_calyx = objFuncPpow(radiussize_inv, poptD_calyx[0], poptD_calyx[1])
fitYD_LH = objFuncPpow(radiussize_inv, poptD_LH[0], poptD_LH[1])
fitYD_AL1 = objFuncPpow(radiussize_inv, poptD_AL1[0], poptD_AL1[1])
fitYD_AL2 = objFuncPpow(radiussize_inv, poptD_AL2[0], poptD_AL2[1])

fig = plt.figure(figsize=(12,8))

plt.scatter(radiussize_inv[:49], 
                    spheredist_calyx_sum_avg[:49], color='tab:blue', facecolors='none')
plt.scatter(radiussize_inv[:53], 
                    spheredist_LH_sum_avg[:53], color='tab:orange', facecolors='none')
plt.scatter(radiussize_inv[8:15], 
                    spheredist_AL_sum_avg[8:15], color='tab:green')
plt.scatter(radiussize_inv[15:], 
                    spheredist_AL_sum_avg[15:], color='tab:green', facecolors='none')
plt.scatter(radiussize_inv[:8], 
                    spheredist_AL_sum_avg[:8], color='tab:green', facecolors='none')

plt.plot(radiussize_inv, fitYD_calyx, lw=2, linestyle='--', color='tab:blue')
plt.plot(radiussize_inv, fitYD_LH, lw=2, linestyle='--', color='tab:orange')
plt.plot(radiussize_inv, fitYD_AL1, lw=2, linestyle='--', color='tab:green')
plt.plot(radiussize_inv, fitYD_AL2, lw=2, linestyle='dashdot', color='tab:green')
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
shiftN = 4
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
                                                        np.log10(binsize[8:27]), 
                                                        np.log10(hlist_count[8:27]),
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

hlist_calyx_count = []
hlist_LH_count = []
hlist_AL_count = []

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
    hlist_calyx_count.append(np.count_nonzero(hc))
    
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
    hlist_LH_count.append(np.count_nonzero(hh))
    
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
    hlist_AL_count.append(np.count_nonzero(ha))




#%%
    
    
poptBcount_calyx, pcovBcount_calyx = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:21]), 
                                                        np.log10(hlist_calyx_count[7:21]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_calyx = np.sqrt(np.diag(pcovBcount_calyx))

poptBcount_LH, pcovBcount_LH = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:21]), 
                                                        np.log10(hlist_LH_count[7:21]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_LH = np.sqrt(np.diag(pcovBcount_LH))

poptBcount_AL, pcovBcount_AL = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[7:23]), 
                                                        np.log10(hlist_AL_count[7:23]),
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

binsize = np.logspace(-1, 3, 100)[13:85:3]

sp_l = 30

max_calyx_b = calyxCM + sp_l
min_calyx_b = calyxCM - sp_l

max_LH_b = LHCM + sp_l
min_LH_b = LHCM - sp_l

max_AL_b = ALCM + sp_l
min_AL_b = ALCM - sp_l

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
    
farg = np.argwhere(np.array(hlist_calyx_b_count) > 1)[-1][0] + 2
iarg = farg - 14
poptBcount_calyx_b, pcovBcount_calyx_b = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_calyx_b_count[iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_calyx_b = np.sqrt(np.diag(pcovBcount_calyx_b))

farg = np.argwhere(np.array(hlist_LH_b_count) > 1)[-1][0] + 2
iarg = farg - 14
poptBcount_LH_b, pcovBcount_LH_b = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_LH_b_count[iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_LH_b = np.sqrt(np.diag(pcovBcount_LH_b))

farg = np.argwhere(np.array(hlist_AL_b_count) > 1)[-1][0] + 2
iarg = farg - 14
poptBcount_AL_b, pcovBcount_AL_b = scipy.optimize.curve_fit(objFuncGL, 
                                                        np.log10(binsize[iarg:farg]), 
                                                        np.log10(hlist_AL_b_count[iarg:farg]),
                                                        p0=[0.1, 0.1], 
                                                        maxfev=10000)
perrBcount_AL_b = np.sqrt(np.diag(pcovBcount_AL_b))

fitYBcount_calyx_b = objFuncPpow(binsize, poptBcount_calyx_b[0], poptBcount_calyx_b[1])
fitYBcount_LH_b = objFuncPpow(binsize, poptBcount_LH_b[0], poptBcount_LH_b[1])
fitYBcount_AL_b = objFuncPpow(binsize, poptBcount_AL_b[0], poptBcount_AL_b[1])
    
fig = plt.figure(figsize=(12,8))
plt.scatter(binsize, hlist_calyx_b_count)
plt.scatter(binsize, hlist_LH_b_count)
plt.scatter(binsize, hlist_AL_b_count)
plt.plot(binsize, fitYBcount_calyx_b, lw=2, linestyle='--')
plt.plot(binsize, fitYBcount_LH_b, lw=2, linestyle='--')
plt.plot(binsize, fitYBcount_AL_b, lw=2, linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['Calyx: ' + str(round(poptBcount_calyx_b[0], 3)) + '$\pm$' + str(round(perrBcount_calyx_b[0], 3)),
            'LH: ' + str(round(poptBcount_LH_b[0], 3)) + '$\pm$' + str(round(perrBcount_LH_b[0], 3)),
            'AL: ' + str(round(poptBcount_AL_b[0], 3)) + '$\pm$' + str(round(perrBcount_AL_b[0], 3))], fontsize=15)
plt.xlim(0.2, 300)
#plt.tight_layout()
plt.xlabel("Box Size", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()



#%% Binary Box-counting for Sub-physiological Region Length Scale

binsize = np.logspace(-1, 3, 100)[10:95]

sp_l = np.arange(5, 120, 1)
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
        farg = np.argwhere(np.array(hlist_calyx_b_count[r][l]) > 1)[-1][0] + 2
        iarg = farg - 30
        if iarg < 0:
            iarg = 0
        poptBcount_calyx_b_t, pcovBcount_calyx_b_t = scipy.optimize.curve_fit(objFuncGL, 
                                                                np.log10(binsize[iarg:farg]), 
                                                                np.log10(hlist_calyx_b_count[r][l][iarg:farg]),
                                                                p0=[0.1, 0.1], 
                                                                maxfev=10000)
        perrBcount_calyx_b_t = np.sqrt(np.diag(pcovBcount_calyx_b_t))
        
        farg = np.argwhere(np.array(hlist_LH_b_count[r][l]) > 1)[-1][0] + 2
        iarg = farg - 30
        if iarg < 0:
            iarg = 0
        poptBcount_LH_b_t, pcovBcount_LH_b_t = scipy.optimize.curve_fit(objFuncGL, 
                                                                np.log10(binsize[iarg:farg]), 
                                                                np.log10(hlist_LH_b_count[r][l][iarg:farg]),
                                                                p0=[0.1, 0.1], 
                                                                maxfev=10000)
        perrBcount_LH_b_t = np.sqrt(np.diag(pcovBcount_LH_b_t))
        
        farg = np.argwhere(np.array(hlist_AL_b_count[r][l]) > 1)[-1][0] + 2
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
        farg = np.argwhere(hlist_single_count[i] > 1)[-1][0] + 2
        iarg = farg - 8
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
plt.show()

poptBcount_single_all = np.sort(np.array(poptBcount_single_list)[:,0])
xval = np.linspace(min(poptBcount_single_all)-0.1, max(poptBcount_single_all)+0.1, 300)

kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.05).fit(poptBcount_single_all.reshape((len(poptBcount_single_all),1)))

log_dens = kde.score_samples(xval.reshape((len(xval),1)))

fig = plt.figure(figsize=(12,8))
plt.hist(poptBcount_single_all, bins=int(len(hlist_single_count)/5), density=True)
plt.plot(xval, np.exp(log_dens), lw=3)
# plt.vlines(xval[np.argmax(np.exp(log_dens))], 0, 5, linestyle='--', label=str(round(xval[np.argmax(np.exp(log_dens))], 3)), color='tab:red')
plt.vlines(np.mean(poptBcount_single_all), 0, 5, linestyle='--', label=str(round(np.mean(poptBcount_single_all), 3)), color='tab:red')
plt.legend(fontsize=15)
plt.show()



# t10 = time.time()

# print('checkpoint 10: ' + str(t10-t9))

#%% Single Neuron Dimnesion Calculation by calyx, LH, and AL using Binary Box-counting

MorphData.calyxdist_neuron = []
MorphData.LHdist_neuron = []
MorphData.ALdist_neuron = []

for i in range(len(MorphData.morph_dist)):
    calyxdist_neuron_t = []
    LHdist_neuron_t = []
    ALdist_neuron_t = []
    for j in range(len(MorphData.morph_dist[i])):
        if ((np.array(MorphData.morph_dist[i][j])[0] > 475).all() and (np.array(MorphData.morph_dist[i][j])[0] < 550).all() and
            (np.array(MorphData.morph_dist[i][j])[1] < 260).all() and (np.array(MorphData.morph_dist[i][j])[2] > 150).all()):
            calyxdist_neuron_t.append(MorphData.morph_dist[i][j])
        elif ((np.array(MorphData.morph_dist[i][j])[0] < 475).all() and (np.array(MorphData.morph_dist[i][j])[1] < 260).all() and
              (np.array(MorphData.morph_dist[i][j])[1] > 180).all() and (np.array(MorphData.morph_dist[i][j])[2] > 125).all()):
            LHdist_neuron_t.append(MorphData.morph_dist[i][j])
        elif ((np.array(MorphData.morph_dist[i][j])[0] > 475).all() and (np.array(MorphData.morph_dist[i][j])[0] < 600).all() and 
              (np.array(MorphData.morph_dist[i][j])[1] > 280).all() and (np.array(MorphData.morph_dist[i][j])[1] < 400).all() and
              (np.array(MorphData.morph_dist[i][j])[2] < 90).all()):
            ALdist_neuron_t.append(MorphData.morph_dist[i][j])
    if len(calyxdist_neuron_t) > 700:
        MorphData.calyxdist_neuron.append(calyxdist_neuron_t)
    if len(LHdist_neuron_t) > 700:
        MorphData.LHdist_neuron.append(LHdist_neuron_t)
    if len(ALdist_neuron_t) > 700:
        MorphData.ALdist_neuron.append(ALdist_neuron_t)


#%% Single Neuron Dimnesion Calculation by calyx

binsize = np.logspace(-2, 3, 100)[20:85:2]

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
    farg = np.argwhere(hlist_single_count_calyx[i] > 1)[-1][0] + 2
    iarg = farg - 9
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

poptBcount_single_all_calyx = np.sort(np.array(poptBcount_single_list_calyx)[:,0])
xval_calyx = np.linspace(min(poptBcount_single_all_calyx)-0.1, max(poptBcount_single_all_calyx)+0.1, 300)

kde_calyx = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.05).fit(poptBcount_single_all_calyx.reshape((len(poptBcount_single_all_calyx),1)))

log_dens_calyx = kde_calyx.score_samples(xval_calyx.reshape((len(xval_calyx),1)))

fig = plt.figure(figsize=(12,8))
plt.hist(poptBcount_single_all_calyx, bins=int(len(hlist_single_count_calyx)/5), density=True)
plt.plot(xval_calyx, np.exp(log_dens_calyx), lw=3)
plt.vlines(np.mean(poptBcount_single_all_calyx), 0, 5, linestyle='--', 
           label=str(round(np.mean(poptBcount_single_all_calyx), 3)), 
           color='tab:red')
plt.ylim(0, 4.5)
plt.legend(fontsize=15)
plt.show()




#%% Single Neuron Dimnesion Calculation by LH

binsize = np.logspace(-2, 3, 100)[20:85:2]

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
    farg = np.argwhere(hlist_single_count_LH[i] > 1)[-1][0] + 2
    iarg = farg - 9
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

poptBcount_single_all_LH = np.sort(np.array(poptBcount_single_list_LH)[:,0])
xval_LH = np.linspace(min(poptBcount_single_all_LH)-0.1, max(poptBcount_single_all_LH)+0.1, 300)

kde_LH = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.05).fit(poptBcount_single_all_LH.reshape((len(poptBcount_single_all_LH),1)))

log_dens_LH = kde_LH.score_samples(xval_LH.reshape((len(xval_LH),1)))

fig = plt.figure(figsize=(12,8))
plt.hist(poptBcount_single_all_LH, bins=int(len(hlist_single_count_LH)/5), density=True)
plt.plot(xval_LH, np.exp(log_dens_LH), lw=3)
plt.vlines(np.mean(poptBcount_single_all_LH), 0, 5, linestyle='--', 
           label=str(round(np.mean(poptBcount_single_all_LH), 3)), 
           color='tab:red')
plt.ylim(0, 4.5)
plt.legend(fontsize=15)
plt.show()



#%% Single Neuron Dimnesion Calculation by AL

binsize = np.logspace(-2, 3, 100)[20:85:2]

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
    farg = np.argwhere(hlist_single_count_AL[i] > 1)[-1][0] + 2
    iarg = farg - 7
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

poptBcount_single_all_AL = np.sort(np.array(poptBcount_single_list_AL)[:,0])
xval_AL = np.linspace(min(poptBcount_single_all_AL)-0.1, max(poptBcount_single_all_AL)+0.1, 300)

kde_AL = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.05).fit(poptBcount_single_all_AL.reshape((len(poptBcount_single_all_AL),1)))

log_dens_AL = kde_AL.score_samples(xval_AL.reshape((len(xval_AL),1)))

fig = plt.figure(figsize=(12,8))
plt.hist(poptBcount_single_all_AL, bins=int(len(hlist_single_count_AL)/5), density=True)
plt.plot(xval_AL, np.exp(log_dens_AL), lw=3)
plt.vlines(np.mean(poptBcount_single_all_AL), 0, 5, linestyle='--', 
           label=str(round(np.mean(poptBcount_single_all_AL), 3)), 
           color='tab:red')
plt.ylim(0, 4.5)
plt.legend(fontsize=15)
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

(rGy_calyx, cML_calyx) = utils.radiusOfGyration(MorphData.calyxdist)
(rGy_LH, cML_LH) = utils.radiusOfGyration(MorphData.LH_dist)
(rGy_AL, cML_AL) = utils.radiusOfGyration(MorphData.AL_dist)

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


#%% Regional dist categorization

glo_info = pd.read_excel(os.path.join(Parameter.PATH, '../all_skeletons_type_list_180919.xlsx'))

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
        
        for q in range(len(BranchData.branchP_dist[glo_idx[i][j]])):
            if ((np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[0] > 475).all() and (np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[0] < 550).all() and
                (np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[1] < 260).all() and (np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[2] > 150).all()):
                morph_dist_calyx_bp_temp2.append(np.array(BranchData.branchP_dist[glo_idx[i][j]][q]))
            elif ((np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[0] < 475).all() and (np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[1] < 260).all() and
                  (np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[1] > 180).all() and (np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[2] > 125).all()):
                morph_dist_LH_bp_temp2.append(np.array(BranchData.branchP_dist[glo_idx[i][j]][q]))
            elif ((np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[0] > 475).all() and (np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[0] < 600).all() and 
                  (np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[1] > 280).all() and (np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[1] < 400).all() and
                  (np.array(BranchData.branchP_dist[glo_idx[i][j]][q])[2] < 90).all()):
                morph_dist_AL_bp_temp2.append(np.array(BranchData.branchP_dist[glo_idx[i][j]][q]))
        
        for r in range(len(MorphData.endP_dist[glo_idx[i][j]])):
            if ((np.array(MorphData.endP_dist[glo_idx[i][j]][r])[0] > 475).all() and (np.array(MorphData.endP_dist[glo_idx[i][j]][r])[0] < 550).all() and
                (np.array(MorphData.endP_dist[glo_idx[i][j]][r])[1] < 260).all() and (np.array(MorphData.endP_dist[glo_idx[i][j]][r])[2] > 150).all()):
                morph_dist_calyx_ep_temp2.append(np.array(MorphData.endP_dist[glo_idx[i][j]][r]))
            elif ((np.array(MorphData.endP_dist[glo_idx[i][j]][r])[0] < 475).all() and (np.array(MorphData.endP_dist[glo_idx[i][j]][r])[1] < 260).all() and
                  (np.array(MorphData.endP_dist[glo_idx[i][j]][r])[1] > 180).all() and (np.array(MorphData.endP_dist[glo_idx[i][j]][r])[2] > 125).all()):
                morph_dist_LH_ep_temp2.append(np.array(MorphData.endP_dist[glo_idx[i][j]][r]))
            elif ((np.array(MorphData.endP_dist[glo_idx[i][j]][r])[0] > 475).all() and (np.array(MorphData.endP_dist[glo_idx[i][j]][r])[0] < 600).all() and 
                  (np.array(MorphData.endP_dist[glo_idx[i][j]][r])[1] > 280).all() and (np.array(MorphData.endP_dist[glo_idx[i][j]][r])[1] < 400).all() and
                  (np.array(MorphData.endP_dist[glo_idx[i][j]][r])[2] < 90).all()):
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

morph_dist_calyx_flat = [item for sublist in morph_dist_calyx for item in sublist]
morph_dist_calyx_flat = [item for sublist in morph_dist_calyx_flat for item in sublist]

mdcalyx_xmax = np.max(np.array(morph_dist_calyx_flat)[:,0])
mdcalyx_xmin = np.min(np.array(morph_dist_calyx_flat)[:,0])
mdcalyx_ymax = np.max(np.array(morph_dist_calyx_flat)[:,1])
mdcalyx_ymin = np.min(np.array(morph_dist_calyx_flat)[:,1])
mdcalyx_zmax = np.max(np.array(morph_dist_calyx_flat)[:,2])
mdcalyx_zmin = np.min(np.array(morph_dist_calyx_flat)[:,2])

morph_dist_LH_flat = [item for sublist in morph_dist_LH for item in sublist]
morph_dist_LH_flat = [item for sublist in morph_dist_LH_flat for item in sublist]

mdLH_xmax = np.max(np.array(morph_dist_LH_flat)[:,0])
mdLH_xmin = np.min(np.array(morph_dist_LH_flat)[:,0])
mdLH_ymax = np.max(np.array(morph_dist_LH_flat)[:,1])
mdLH_ymin = np.min(np.array(morph_dist_LH_flat)[:,1])
mdLH_zmax = np.max(np.array(morph_dist_LH_flat)[:,2])
mdLH_zmin = np.min(np.array(morph_dist_LH_flat)[:,2])

morph_dist_AL_flat = [item for sublist in morph_dist_AL for item in sublist]
morph_dist_AL_flat = [item for sublist in morph_dist_AL_flat for item in sublist]

mdAL_xmax = np.max(np.array(morph_dist_AL_flat)[:,0])
mdAL_xmin = np.min(np.array(morph_dist_AL_flat)[:,0])
mdAL_ymax = np.max(np.array(morph_dist_AL_flat)[:,1])
mdAL_ymin = np.min(np.array(morph_dist_AL_flat)[:,1])
mdAL_zmax = np.max(np.array(morph_dist_AL_flat)[:,2])
mdAL_zmin = np.min(np.array(morph_dist_AL_flat)[:,2])

hull_calyx = ConvexHull(np.array(morph_dist_calyx_flat))
calyx_vol = hull_calyx.volume
calyx_area = hull_calyx.area

hull_LH = ConvexHull(np.array(morph_dist_LH_flat))
LH_vol = hull_LH.volume
LH_area = hull_LH.area

hull_AL = ConvexHull(np.array(morph_dist_AL_flat))
AL_vol = hull_AL.volume
AL_area = hull_AL.area

    
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



#%% Cluster quantification

morph_dist_calyx_CM_flat = np.array([item for sublist in morph_dist_calyx_CM for item in sublist])
morph_dist_LH_CM_flat = np.array([item for sublist in morph_dist_LH_CM for item in sublist])
morph_dist_AL_CM_flat = np.array([item for sublist in morph_dist_AL_CM for item in sublist])

morph_dist_calyx_r = scipy.spatial.distance.cdist(morph_dist_calyx_CM_flat, morph_dist_calyx_CM_flat)
morph_dist_LH_r = scipy.spatial.distance.cdist(morph_dist_LH_CM_flat, morph_dist_LH_CM_flat)
morph_dist_AL_r = scipy.spatial.distance.cdist(morph_dist_AL_CM_flat, morph_dist_AL_CM_flat)

glo_len = [len(arr) for arr in glo_idx]
glo_lb = [sum(glo_len[0:i]) for i in range(len(glo_len)+1)]
glo_lbs = np.subtract(glo_lb, glo_lb[0])
glo_float = np.divide(glo_lbs, glo_lbs[-1])

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


print("Calyx cluster Mean: " + str(np.mean(calyxdist_cluster_u_full_flat)) + ", STD: " + str(np.std(calyxdist_cluster_u_full_flat)))
print("Calyx noncluster Mean: " + str(np.mean(calyxdist_noncluster_u_full_flat)) + ", STD: " + str(np.std(calyxdist_noncluster_u_full_flat)))

print("LH cluster Mean: " + str(np.mean(LHdist_cluster_u_full_flat)) + ", STD: " + str(np.std(LHdist_cluster_u_full_flat)))
print("LH noncluster Mean: " + str(np.mean(LHdist_noncluster_u_full_flat)) + ", STD: " + str(np.std(LHdist_noncluster_u_full_flat)))

print("AL cluster Mean: " + str(np.mean(ALdist_cluster_u_full_flat)) + ", STD: " + str(np.std(ALdist_cluster_u_full_flat)))
print("AL noncluster Mean: " + str(np.mean(ALdist_noncluster_u_full_flat)) + ", STD: " + str(np.std(ALdist_noncluster_u_full_flat)))


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


fig = plt.figure(figsize=(6, 4))
plt.hist(calyxdist_cluster_u_full_flat, alpha=0.5, density=True)
plt.hist(calyxdist_noncluster_u_full_flat, alpha=0.5, density=True)
plt.xlabel('Distance')
plt.ylabel('Probability')
plt.title('Distances between within and outside cluster distances of calyx')
plt.legend(['Cluster', 'Non-Cluster'])
plt.show()

fig = plt.figure(figsize=(6, 4))
plt.hist(LHdist_cluster_u_full_flat, alpha=0.5, density=True)
plt.hist(LHdist_noncluster_u_full_flat, alpha=0.5, density=True)
plt.xlabel('Distance')
plt.ylabel('Probability')
plt.title('Distances between within and outside cluster distances of LH')
plt.legend(['Cluster', 'Non-Cluster'])
plt.show()

fig = plt.figure(figsize=(6, 4))
plt.hist(ALdist_cluster_u_full_flat, alpha=0.5, density=True)
plt.hist(ALdist_noncluster_u_full_flat, alpha=0.5, density=True)
plt.xlabel('Distance')
plt.ylabel('Probability')
plt.title('Distances between within and outside cluster distances of AL')
plt.legend(['Cluster', 'Non-Cluster'])
plt.show()


fig, ax = plt.subplots()
labels = ['Calyx', 'LH', 'AL']
x = np.arange(len(labels))
width = .3

cmeans = [np.median(calyxdist_cluster_u_full_flat), np.median(LHdist_cluster_u_full_flat), np.median(ALdist_cluster_u_full_flat)]
cerr = [scipy.stats.median_absolute_deviation(calyxdist_cluster_u_full_flat, center=np.median), 
        scipy.stats.median_absolute_deviation(LHdist_cluster_u_full_flat, center=np.median), 
        scipy.stats.median_absolute_deviation(ALdist_cluster_u_full_flat, center=np.median)]
ncmeans = [np.median(calyxdist_noncluster_u_full_flat), np.median(LHdist_noncluster_u_full_flat), np.median(ALdist_noncluster_u_full_flat)]
ncerr = [scipy.stats.median_absolute_deviation(calyxdist_noncluster_u_full_flat, center=np.median), 
         scipy.stats.median_absolute_deviation(LHdist_noncluster_u_full_flat, center=np.median), 
         scipy.stats.median_absolute_deviation(ALdist_noncluster_u_full_flat, center=np.median)]

ax.bar(x - width/2, cmeans, width, yerr=cerr, capsize=5, label='Cluster')
ax.bar(x + width/2, ncmeans, width, yerr=ncerr, capsize=5, label='Non-Cluster')
ax.set_ylabel('Distance')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_title('Median distance within and outside cluster')
plt.tight_layout()
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

for i in range(len(morph_dist_AL_r)):
    ALcalyx_corr.append(np.corrcoef(morph_dist_calyx_r[i], morph_dist_AL_r[i])[0][1])
    ALLH_corr.append(np.corrcoef(morph_dist_LH_r[i], morph_dist_AL_r[i])[0][1])

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
ax.bar(x, ALcalyx_corr_glo_avg, width, yerr=ALcalyx_corr_glo_std, label='Calyx-AL', alpha=0.5, error_kw=dict(ecolor='tab:blue', lw=1, capsize=2, capthick=1))
ax.bar(x, ALLH_corr_glo_avg, width, yerr=ALLH_corr_glo_std, label='LH-AL', alpha=0.5, error_kw=dict(ecolor='tab:orange', lw=1, capsize=2, capthick=1))
ax.set_ylabel('Distance')
ax.set_xticks(x)
ax.set_xticklabels(glo_list, rotation=90, fontsize=7)
ax.legend()
ax.set_title('Distance correlation between calyx/LH and AL by glomerulus')
plt.xlim(0-0.5, len(glo_list)-0.5)
plt.tight_layout()
plt.show()


validx = np.argwhere(np.array(ALLH_corr_glo_avg) > 0.5).T[0]

diffidx = np.argwhere(np.subtract(ALLH_corr_glo_avg, ALcalyx_corr_glo_avg) > 0.5).T[0]

print(np.sort(np.array(glo_list)[validx]))
print(np.sort(np.array(glo_list)[diffidx]))


#%% Correlation matrix cluster

glo_list_neuron = np.repeat(glo_list, glo_len)
glo_lb_idx = []

for i in range(len(glo_lb)-1):
    glo_lb_idx.append(np.arange(glo_lb[i],glo_lb[i+1]))

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

L = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_AL_r_avg), method='complete')
ind = scipy.cluster.hierarchy.fcluster(L, 100, 'maxclust')
columns = [morph_dist_AL_r_avg_df.columns.tolist()[i] for i in list((np.argsort(ind)))]

glo_list_cluster = np.array(glo_list)[columns]
# glo_list_cluster = ['DL3', 'DA1', 'VM7d', 'VM7v', 'VC4', 'VM5v', 'VM5d', 'DM6', 'DM2', 'DM5', 
#   'DA2', 'DC1', 'DA4l', 'VC1', 'VA6', 'DC2', 'DC4', 'DL5', 'D', 'DL1', 'DA3', 'DA4m',
#   'DL4', 'VA1v', 'VA1d', 'DC3', 'VL2p', 'VL2a', 'VA7l', 'VA3', 'VA5', 'VA7m', 'VM1',
#   'VC3l', 'VC3m', 'VM4', 'VM6', 'VL1', 'V', 'DL2d', 'DL2v', 'VM2', 'VM3', 'DP1l', 'VA4',
#   'VC2', 'VA2', 'DP1m', 'DM3', 'DM4', 'DM1']

# columns = [glo_list.index(glo_list_cluster[i]) for i in range(len(glo_list_cluster))]

glo_len_cluster = np.array(glo_len)[columns]

# Custom ordering based on unsupervised clustering using NBLAST from manuscript
# glo_list_cluster = ['DL3', 'DA1', 'VM7d', 'VM7v', 'VC4', 'VM5v', 'VM5d', 'DM6', 'DM2', 'DM5', 
#   'DA2', 'DC1', 'DA4l', 'VC1', 'VA6', 'DC2', 'DC4', 'DL5', 'D', 'DL1', 'DA3', 'DA4m',
#   'DL4', 'VA1v', 'VA1d', 'DC3', 'VL2p', 'VL2a', 'VA7l', 'VA3', 'VA5', 'VA7m', 'VM1',
#   'VC3l', 'VC3m', 'VM4', 'VM6', 'VL1', 'V', 'DL2d', 'DL2v', 'VM2', 'VM3', 'DP1l', 'VA4',
#   'VC2', 'VA2', 'DP1m', 'DM3', 'DM4', 'DM1']

# glo_idx_cluster = []
# for i in range(len(glo_list_cluster)):
#     glo_idx_cluster.append(glo_idx[np.argwhere(glo_list_cluster[i] == np.array(glo_list))[0][0]])

# glo_list = glo_list_cluster
# glo_idx = glo_idx_cluster

glo_idx_cluster = []
for i in range(len(glo_list)):
    glo_idx_cluster.append(glo_lb_idx[np.argwhere(np.array(glo_list)[columns][i] == np.array(glo_list))[0][0]])

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

fig = plt.figure()
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(morph_dist_calyx_r_df)#, vmax=np.max(morph_dist_calyx_r))
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
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=4, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=4, rotation_mode='default')
plt.colorbar()
plt.title("Reorganized inter-cluster distance calyx", pad=40)
plt.show()

fig = plt.figure()
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(morph_dist_LH_r_df)#, vmax=np.max(morph_dist_LH_r))
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
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=4, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=4, rotation_mode='default')
plt.colorbar()
plt.title("Reorganized inter-cluster distance LH", pad=40)
plt.show()

fig = plt.figure()
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(morph_dist_AL_r_df)#, vmax=np.max(morph_dist_AL_r))
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
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=4, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=4, rotation_mode='default')
plt.colorbar()
plt.title("Reorganized inter-cluster distance AL", pad=40)
plt.show()



fig, ax = plt.subplots()
x = np.arange(len(glo_list))
width = 1.
ax.bar(x, np.array(ALcalyx_corr_glo_avg)[columns], width, 
       yerr=np.array(ALcalyx_corr_glo_std)[columns], label='Calyx-AL', alpha=0.5, 
       error_kw=dict(ecolor='tab:blue', lw=1, capsize=2, capthick=1))
ax.bar(x, np.array(ALLH_corr_glo_avg)[columns], width, 
       yerr=np.array(ALLH_corr_glo_std)[columns], label='LH-AL', alpha=0.5, 
       error_kw=dict(ecolor='tab:orange', lw=1, capsize=2, capthick=1))
ax.set_ylabel('Distance')
ax.set_xticks(x)
ax.set_xticklabels(glo_list_cluster, rotation=90, fontsize=7)
ax.legend()
ax.set_title('Distance correlation between calyx/LH and AL by glomerulus')
plt.xlim(0-0.5, len(glo_list)-0.5)
plt.tight_layout()
plt.show()




#%% Entropy

morph_dist_calyx_hist_x = []
morph_dist_calyx_hist_y = []
morph_dist_calyx_hist_z = []

for i in range(len(morph_dist_calyx)):
    morph_dist_calyx_hist_x_t = []
    morph_dist_calyx_hist_y_t = []
    morph_dist_calyx_hist_z_t = []
    for j in range(len(morph_dist_calyx[i])):
        xval = np.linspace(mdcalyx_xmin-1, mdcalyx_xmax+1, 300)
        yval = np.linspace(mdcalyx_ymin-1, mdcalyx_ymax+1, 300)
        zval = np.linspace(mdcalyx_zmin-1, mdcalyx_zmax+1, 300)
        
        hx = np.array(morph_dist_calyx[i][j])[:,0]
        hy = np.array(morph_dist_calyx[i][j])[:,1]
        hz = np.array(morph_dist_calyx[i][j])[:,2]
        
        kdecalyx_hx = neighbors.KernelDensity(kernel='gaussian', bandwidth=2.0).fit(hx.reshape((len(hx),1)))
        kdecalyx_hy = neighbors.KernelDensity(kernel='gaussian', bandwidth=2.0).fit(hy.reshape((len(hy),1)))
        kdecalyx_hz = neighbors.KernelDensity(kernel='gaussian', bandwidth=2.0).fit(hz.reshape((len(hz),1)))
        
        log_dens_hx = kdecalyx_hx.score_samples(xval.reshape((len(xval),1)))
        log_dens_hy = kdecalyx_hy.score_samples(yval.reshape((len(yval),1)))
        log_dens_hz = kdecalyx_hz.score_samples(zval.reshape((len(zval),1)))
        
        morph_dist_calyx_hist_x_t.append(np.exp(log_dens_hx)*(xval[1]-xval[0]))
        morph_dist_calyx_hist_y_t.append(np.exp(log_dens_hy)*(yval[1]-yval[0]))
        morph_dist_calyx_hist_z_t.append(np.exp(log_dens_hz)*(zval[1]-zval[0]))
    
    morph_dist_calyx_hist_x.append(morph_dist_calyx_hist_x_t)
    morph_dist_calyx_hist_y.append(morph_dist_calyx_hist_y_t)
    morph_dist_calyx_hist_z.append(morph_dist_calyx_hist_z_t)

calyx_ent_cluster = []
calyx_ent_noncluster = []

morph_dist_calyx_hist_x_flat = np.array([item for sublist in morph_dist_calyx_hist_x for item in sublist])
morph_dist_calyx_hist_y_flat = np.array([item for sublist in morph_dist_calyx_hist_y for item in sublist])
morph_dist_calyx_hist_z_flat = np.array([item for sublist in morph_dist_calyx_hist_z for item in sublist])

ent_calyx_x = []
ent_calyx_y = []
ent_calyx_z = []

for i in range(len(morph_dist_calyx_hist_x_flat)):
    ent_calyx_x_t = []
    ent_calyx_y_t = []
    ent_calyx_z_t = []
    
    for j in range(len(morph_dist_calyx_hist_x_flat)):
        ent_x = scipy.stats.entropy(morph_dist_calyx_hist_x_flat[i], qk=morph_dist_calyx_hist_x_flat[j])
        ent_y = scipy.stats.entropy(morph_dist_calyx_hist_y_flat[i], qk=morph_dist_calyx_hist_y_flat[j])
        ent_z = scipy.stats.entropy(morph_dist_calyx_hist_z_flat[i], qk=morph_dist_calyx_hist_z_flat[j])
        
        ent_calyx_x_t.append(ent_x)
        ent_calyx_y_t.append(ent_y)
        ent_calyx_z_t.append(ent_z)
        
    ent_calyx_x.append(ent_calyx_x_t)
    ent_calyx_y.append(ent_calyx_y_t)
    ent_calyx_z.append(ent_calyx_z_t)

ent_calyx = np.add(np.add(np.array(ent_calyx_x), np.array(ent_calyx_y)), np.array(ent_calyx_z))

calyx_ent_clusterstat = []
calyx_ent_cluster = []
calyx_ent_noncluster = []

idx = np.arange(len(ent_calyx))
trk1 = 0

for f in range(len(glo_list)):
    dist_cluster = []
    dist_noncluster = []
    for i in range(len(morph_dist_calyx_hist_x[f])):
        dist_cluster.append(ent_calyx[trk1:trk1+len(morph_dist_calyx_hist_x[f]),trk1:trk1+len(morph_dist_calyx_hist_x[f])])
        dist_noncluster.append(ent_calyx[trk1:trk1+len(morph_dist_calyx_hist_x[f]),np.delete(idx, np.arange(trk1,trk1+len(morph_dist_calyx_hist_x[f])))])
    trk1 += len(morph_dist_calyx_hist_x[f])
    
    dist_cluster_u = np.unique(dist_cluster)
    dist_noncluster_u = np.unique(dist_noncluster)
    
    calyx_ent_cluster.append(dist_cluster_u)
    calyx_ent_noncluster.append(dist_noncluster_u)
    
    calyx_ent_clusterstat.append([np.mean(dist_cluster_u), np.std(dist_cluster_u), np.mean(dist_noncluster_u), np.std(dist_noncluster_u)])

calyx_ent_cluster_flat = [item for sublist in calyx_ent_cluster for item in sublist]
calyx_ent_noncluster_flat = [item for sublist in calyx_ent_noncluster for item in sublist]

calyx_ent_cluster_flat = np.array(calyx_ent_cluster_flat)[np.nonzero(calyx_ent_cluster_flat)[0]]
calyx_ent_noncluster_flat = np.array(calyx_ent_noncluster_flat)[np.nonzero(calyx_ent_noncluster_flat)[0]]

print("Calyx cluster Mean: " + str(np.median(calyx_ent_cluster_flat)) + ", STD: " + str(scipy.stats.median_absolute_deviation(calyx_ent_cluster_flat)))
print("Calyx noncluster Mean: " + str(np.median(calyx_ent_noncluster_flat)) + ", STD: " + str(scipy.stats.median_absolute_deviation(calyx_ent_noncluster_flat)))


morph_dist_LH_hist_x = []
morph_dist_LH_hist_y = []
morph_dist_LH_hist_z = []

for i in range(len(morph_dist_LH)):
    morph_dist_LH_hist_x_t = []
    morph_dist_LH_hist_y_t = []
    morph_dist_LH_hist_z_t = []
    for j in range(len(morph_dist_LH[i])):
        xval = np.linspace(mdLH_xmin-1, mdLH_xmax+1, 300)
        yval = np.linspace(mdLH_ymin-1, mdLH_ymax+1, 300)
        zval = np.linspace(mdLH_zmin-1, mdLH_zmax+1, 300)
        
        hx = np.array(morph_dist_LH[i][j])[:,0]
        hy = np.array(morph_dist_LH[i][j])[:,1]
        hz = np.array(morph_dist_LH[i][j])[:,2]
        
        kdeLH_hx = neighbors.KernelDensity(kernel='gaussian', bandwidth=2.0).fit(hx.reshape((len(hx),1)))
        kdeLH_hy = neighbors.KernelDensity(kernel='gaussian', bandwidth=2.0).fit(hy.reshape((len(hy),1)))
        kdeLH_hz = neighbors.KernelDensity(kernel='gaussian', bandwidth=2.0).fit(hz.reshape((len(hz),1)))
        
        log_dens_hx = kdeLH_hx.score_samples(xval.reshape((len(xval),1)))
        log_dens_hy = kdeLH_hy.score_samples(yval.reshape((len(yval),1)))
        log_dens_hz = kdeLH_hz.score_samples(zval.reshape((len(zval),1)))
        
        morph_dist_LH_hist_x_t.append(np.exp(log_dens_hx)*(xval[1]-xval[0]))
        morph_dist_LH_hist_y_t.append(np.exp(log_dens_hy)*(yval[1]-yval[0]))
        morph_dist_LH_hist_z_t.append(np.exp(log_dens_hz)*(zval[1]-zval[0]))
    
    morph_dist_LH_hist_x.append(morph_dist_LH_hist_x_t)
    morph_dist_LH_hist_y.append(morph_dist_LH_hist_y_t)
    morph_dist_LH_hist_z.append(morph_dist_LH_hist_z_t)

LH_ent_cluster = []
LH_ent_noncluster = []

morph_dist_LH_hist_x_flat = np.array([item for sublist in morph_dist_LH_hist_x for item in sublist])
morph_dist_LH_hist_y_flat = np.array([item for sublist in morph_dist_LH_hist_y for item in sublist])
morph_dist_LH_hist_z_flat = np.array([item for sublist in morph_dist_LH_hist_z for item in sublist])

ent_LH_x = []
ent_LH_y = []
ent_LH_z = []

for i in range(len(morph_dist_LH_hist_x_flat)):
    ent_LH_x_t = []
    ent_LH_y_t = []
    ent_LH_z_t = []
    
    for j in range(len(morph_dist_LH_hist_x_flat)):
        ent_x = scipy.stats.entropy(morph_dist_LH_hist_x_flat[i], qk=morph_dist_LH_hist_x_flat[j])
        ent_y = scipy.stats.entropy(morph_dist_LH_hist_y_flat[i], qk=morph_dist_LH_hist_y_flat[j])
        ent_z = scipy.stats.entropy(morph_dist_LH_hist_z_flat[i], qk=morph_dist_LH_hist_z_flat[j])
        
        ent_LH_x_t.append(ent_x)
        ent_LH_y_t.append(ent_y)
        ent_LH_z_t.append(ent_z)
        
    ent_LH_x.append(ent_LH_x_t)
    ent_LH_y.append(ent_LH_y_t)
    ent_LH_z.append(ent_LH_z_t)

ent_LH = np.add(np.add(np.array(ent_LH_x), np.array(ent_LH_y)), np.array(ent_LH_z))

LH_ent_clusterstat = []
LH_ent_cluster = []
LH_ent_noncluster = []

idx = np.arange(len(ent_LH))
trk1 = 0

for f in range(len(glo_list)):
    dist_cluster = []
    dist_noncluster = []
    for i in range(len(morph_dist_LH_hist_x[f])):
        dist_cluster.append(ent_LH[trk1:trk1+len(morph_dist_LH_hist_x[f]),trk1:trk1+len(morph_dist_LH_hist_x[f])])
        dist_noncluster.append(ent_LH[trk1:trk1+len(morph_dist_LH_hist_x[f]),np.delete(idx, np.arange(trk1,trk1+len(morph_dist_LH_hist_x[f])))])
    trk1 += len(morph_dist_LH_hist_x[f])
    
    dist_cluster_u = np.unique(dist_cluster)
    dist_noncluster_u = np.unique(dist_noncluster)
    
    LH_ent_cluster.append(dist_cluster_u)
    LH_ent_noncluster.append(dist_noncluster_u)
    
    LH_ent_clusterstat.append([np.mean(dist_cluster_u), np.std(dist_cluster_u), np.mean(dist_noncluster_u), np.std(dist_noncluster_u)])

LH_ent_cluster_flat = [item for sublist in LH_ent_cluster for item in sublist]
LH_ent_noncluster_flat = [item for sublist in LH_ent_noncluster for item in sublist]

LH_ent_cluster_flat = np.array(LH_ent_cluster_flat)[np.nonzero(LH_ent_cluster_flat)[0]]
LH_ent_noncluster_flat = np.array(LH_ent_noncluster_flat)[np.nonzero(LH_ent_noncluster_flat)[0]]

print("LH cluster Mean: " + str(np.median(LH_ent_cluster_flat)) + ", STD: " + str(scipy.stats.median_absolute_deviation(LH_ent_cluster_flat)))
print("LH noncluster Mean: " + str(np.median(LH_ent_noncluster_flat)) + ", STD: " + str(scipy.stats.median_absolute_deviation(LH_ent_noncluster_flat)))


morph_dist_AL_hist_x = []
morph_dist_AL_hist_y = []
morph_dist_AL_hist_z = []

for i in range(len(morph_dist_AL)):
    morph_dist_AL_hist_x_t = []
    morph_dist_AL_hist_y_t = []
    morph_dist_AL_hist_z_t = []
    for j in range(len(morph_dist_AL[i])):
        xval = np.linspace(mdAL_xmin-1, mdAL_xmax+1, 300)
        yval = np.linspace(mdAL_ymin-1, mdAL_ymax+1, 300)
        zval = np.linspace(mdAL_zmin-1, mdAL_zmax+1, 300)
        
        hx = np.array(morph_dist_AL[i][j])[:,0]
        hy = np.array(morph_dist_AL[i][j])[:,1]
        hz = np.array(morph_dist_AL[i][j])[:,2]
        
        kdeAL_hx = neighbors.KernelDensity(kernel='gaussian', bandwidth=3.0).fit(hx.reshape((len(hx),1)))
        kdeAL_hy = neighbors.KernelDensity(kernel='gaussian', bandwidth=3.0).fit(hy.reshape((len(hy),1)))
        kdeAL_hz = neighbors.KernelDensity(kernel='gaussian', bandwidth=3.0).fit(hz.reshape((len(hz),1)))
        
        log_dens_hx = kdeAL_hx.score_samples(xval.reshape((len(xval),1)))
        log_dens_hy = kdeAL_hy.score_samples(yval.reshape((len(yval),1)))
        log_dens_hz = kdeAL_hz.score_samples(zval.reshape((len(zval),1)))
        
        morph_dist_AL_hist_x_t.append(np.exp(log_dens_hx)*(xval[1]-xval[0]))
        morph_dist_AL_hist_y_t.append(np.exp(log_dens_hy)*(yval[1]-yval[0]))
        morph_dist_AL_hist_z_t.append(np.exp(log_dens_hz)*(zval[1]-zval[0]))
    
    morph_dist_AL_hist_x.append(morph_dist_AL_hist_x_t)
    morph_dist_AL_hist_y.append(morph_dist_AL_hist_y_t)
    morph_dist_AL_hist_z.append(morph_dist_AL_hist_z_t)

AL_ent_cluster = []
AL_ent_noncluster = []

morph_dist_AL_hist_x_flat = np.array([item for sublist in morph_dist_AL_hist_x for item in sublist])
morph_dist_AL_hist_y_flat = np.array([item for sublist in morph_dist_AL_hist_y for item in sublist])
morph_dist_AL_hist_z_flat = np.array([item for sublist in morph_dist_AL_hist_z for item in sublist])

ent_AL_x = []
ent_AL_y = []
ent_AL_z = []

for i in range(len(morph_dist_AL_hist_x_flat)):
    ent_AL_x_t = []
    ent_AL_y_t = []
    ent_AL_z_t = []
    
    for j in range(len(morph_dist_AL_hist_x_flat)):
        ent_x = scipy.stats.entropy(morph_dist_AL_hist_x_flat[i], qk=morph_dist_AL_hist_x_flat[j])
        ent_y = scipy.stats.entropy(morph_dist_AL_hist_y_flat[i], qk=morph_dist_AL_hist_y_flat[j])
        ent_z = scipy.stats.entropy(morph_dist_AL_hist_z_flat[i], qk=morph_dist_AL_hist_z_flat[j])
        
        ent_AL_x_t.append(ent_x)
        ent_AL_y_t.append(ent_y)
        ent_AL_z_t.append(ent_z)
        
    ent_AL_x.append(ent_AL_x_t)
    ent_AL_y.append(ent_AL_y_t)
    ent_AL_z.append(ent_AL_z_t)

ent_AL = np.add(np.add(np.array(ent_AL_x), np.array(ent_AL_y)), np.array(ent_AL_z))

AL_ent_clusterstat = []
AL_ent_cluster = []
AL_ent_noncluster = []

idx = np.arange(len(ent_AL))
trk1 = 0

for f in range(len(glo_list)):
    dist_cluster = []
    dist_noncluster = []
    for i in range(len(morph_dist_AL_hist_x[f])):
        dist_cluster.append(ent_AL[trk1:trk1+len(morph_dist_AL_hist_x[f]),trk1:trk1+len(morph_dist_AL_hist_x[f])])
        dist_noncluster.append(ent_AL[trk1:trk1+len(morph_dist_AL_hist_x[f]),np.delete(idx, np.arange(trk1,trk1+len(morph_dist_AL_hist_x[f])))])
    trk1 += len(morph_dist_AL_hist_x[f])
    
    dist_cluster_u = np.unique(dist_cluster)
    dist_noncluster_u = np.unique(dist_noncluster)
    
    AL_ent_cluster.append(dist_cluster_u)
    AL_ent_noncluster.append(dist_noncluster_u)
    
    AL_ent_clusterstat.append([np.mean(dist_cluster_u), np.std(dist_cluster_u), np.mean(dist_noncluster_u), np.std(dist_noncluster_u)])

AL_ent_cluster_flat = [item for sublist in AL_ent_cluster for item in sublist]
AL_ent_noncluster_flat = [item for sublist in AL_ent_noncluster for item in sublist]

AL_ent_cluster_flat = np.array(AL_ent_cluster_flat)[np.nonzero(AL_ent_cluster_flat)[0]]
AL_ent_noncluster_flat = np.array(AL_ent_noncluster_flat)[np.nonzero(AL_ent_noncluster_flat)[0]]

print("AL cluster Mean: " + str(np.median(AL_ent_cluster_flat)) + ", STD: " + str(scipy.stats.median_absolute_deviation(AL_ent_cluster_flat)))
print("AL noncluster Mean: " + str(np.median(AL_ent_noncluster_flat)) + ", STD: " + str(scipy.stats.median_absolute_deviation(AL_ent_noncluster_flat)))


#%%

fig, ax = plt.subplots()
labels = ['Calyx', 'LH', 'AL']
x = np.arange(len(labels))
width = .3

cmeans = [np.median(calyx_ent_cluster_flat), np.median(LH_ent_cluster_flat), np.median(AL_ent_cluster_flat)]
cerr = [scipy.stats.median_absolute_deviation(calyx_ent_cluster_flat, center=np.median), 
        scipy.stats.median_absolute_deviation(LH_ent_cluster_flat, center=np.median), 
        scipy.stats.median_absolute_deviation(AL_ent_cluster_flat, center=np.median)]
ncmeans = [np.median(calyx_ent_noncluster_flat), np.median(LH_ent_noncluster_flat), np.median(AL_ent_noncluster_flat)]
ncerr = [scipy.stats.median_absolute_deviation(calyx_ent_noncluster_flat, center=np.median), 
         scipy.stats.median_absolute_deviation(LH_ent_noncluster_flat, center=np.median), 
         scipy.stats.median_absolute_deviation(AL_ent_noncluster_flat, center=np.median)]

ax.bar(x - width/2, cmeans, width, yerr=cerr, capsize=5, label='Cluster')
ax.bar(x + width/2, ncmeans, width, yerr=ncerr, capsize=5, label='Non-Cluster')
ax.set_ylabel('Entropy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
lab = ['Calyx C', 'Calyx NC', 'LH C', 'LH NC', 'AL C', 'AL NC']
plt.boxplot([calyx_ent_cluster_flat, calyx_ent_noncluster_flat, 
             LH_ent_cluster_flat, LH_ent_noncluster_flat, 
             AL_ent_cluster_flat, AL_ent_noncluster_flat], 
            notch=True, labels=lab, bootstrap=500)
plt.tight_layout()
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
ax.scatter3D(np.array(morph_dist_calyx_ep_mean)[:,0], np.array(morph_dist_calyx_ep_mean)[:,1], np.array(morph_dist_calyx_ep_mean)[:,2], color='tab:blue')
ax.scatter3D(np.array(morph_dist_LH_ep_mean)[:,0], np.array(morph_dist_LH_ep_mean)[:,1], np.array(morph_dist_LH_ep_mean)[:,2], color='tab:orange')
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
ax.scatter3D(np.array(morph_dist_calyx_ep_mean)[:,0], np.array(morph_dist_calyx_ep_mean)[:,1], np.array(morph_dist_calyx_ep_mean)[:,2], color='tab:blue')
ax.scatter3D(np.array(morph_dist_LH_ep_mean)[:,0], np.array(morph_dist_LH_ep_mean)[:,1], np.array(morph_dist_LH_ep_mean)[:,2], color='tab:orange')
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
gi=52

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

