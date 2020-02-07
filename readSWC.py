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
    numSample = 10
    RN = '1'
    
    sSize = 1000
    nSize = [100, 1000, 10000, 100000]
    dSize = 100
    
    SEED = 1234
    
    outputdir = './output_TEMCA/RN_' + str(RN)

fp = [f for f in os.listdir(Parameter.PATH) if os.path.isfile(os.path.join(Parameter.PATH, f))]
fp = [os.path.join(Parameter.PATH, f) for f in fp]

#fp.pop(17)
fp = fp[:5]

class MorphData():
    
    def __init__(self):
        self.morph_id = []
        self.morph_parent = []
        self.morph_prox = []
        self.morph_dist = []
        self.neuron_id = []
        self.neuron_type = []
        self.endP = []
        self.somaP = []
        self.indRegMDist = None
        self.indRegMDistLen = None
    
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
            listOfPoints = self.indRegMDist[multListOfPoints[i][0]][multListOfPoints[i][1]:multListOfPoints[i][2]]
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
    
class BranchData:
    branchTrk = []
    branch_dist = []
    indMorph_dist = []
    indBranchTrk = []
    branchP = []
    branchNum = np.empty(len(fp))

class OutputData:
    rGyRegSegs = None
    cMLRegSegs = None
    regSegOrdNs = None
    randTrks = None
    rGyRegSegi = None
    cMLRegSegi = None
    regSegOrdNi = None
    randTrki = None
    rGyRegSegm = None
    cMLRegSegm = None
    regSegOrdNm = None
    randTrkm = None
    


MorphData = MorphData()

t0 = time.time()

for f in range(len(fp)):
    print(f)
    morph_neu_id = []
    morph_neu_parent = []
    morph_neu_prox = []
    morph_neu_dist = []
    
    df = pd.read_csv(fp[f], delimiter=' ', header=None)
    
    MorphData.neuron_id.append(os.path.basename(fp[f]).split('.')[0])
    
    scall = df.iloc[np.where(df[6] == -1)[0]]
    
    MorphData.morph_id.append(df[0].tolist())
    MorphData.morph_parent.append(df[6].tolist())
    MorphData.morph_dist.append(np.divide(np.array(df[[2,3,4]]), 1000).tolist()) # Scale
    ctr = Counter(df[6].tolist())
    ctrVal = list(ctr.values())
    ctrKey = list(ctr.keys())
    BranchData.branchNum[f] = sum(i > 1 for i in ctrVal)
    branchInd = np.array(ctrKey)[np.where(np.array(ctrVal) > 1)[0]]
    
    neu_branchTrk = []
    branch_dist_temp1 = []
    length_branch_temp = []
    
    list_end = np.setdiff1d(MorphData.morph_id[f], MorphData.morph_parent[f])
    
    BranchData.branchP.append(branchInd)
    MorphData.endP.append(list_end)
    bPoint = np.append(branchInd, list_end)
    
    for bp in range(len(bPoint)):
        neu_branchTrk_temp = []
        branch_dist_temp2 = []
        dist = 0
        
        neu_branchTrk_temp.append(bPoint[bp])
        branch_dist_temp2.append(MorphData.morph_dist[f][MorphData.morph_id[f].index(bPoint[bp])])
        parentTrck = bPoint[bp]
        while (parentTrck not in branchInd or bPoint[bp] in branchInd) and (parentTrck != -1):
            parentTrck = MorphData.morph_parent[f][MorphData.morph_id[f].index(parentTrck)]
            if parentTrck != -1:
                neu_branchTrk_temp.append(parentTrck)
                rhs = branch_dist_temp2[-1]
                lhs = MorphData.morph_dist[f][MorphData.morph_id[f].index(parentTrck)]
                branch_dist_temp2.append(lhs)
                dist +=  np.linalg.norm(np.subtract(rhs, lhs))
                
        if len(neu_branchTrk_temp) > 1:
            neu_branchTrk.append(neu_branchTrk_temp)
            branch_dist_temp1.append(branch_dist_temp2)
            length_branch_temp.append(dist)
    BranchData.branchTrk.append(neu_branchTrk)
    BranchData.branch_dist.append(branch_dist_temp1)
    LengthData.length_branch.append(length_branch_temp)
    

#    for ep in range(len(list_end)):
#        neu_indBranchTrk_temp = []
#        neu_indBranchTrk_temp.append(list_end[ep])
#        parentTrck = list_end[ep]
#        while parentTrck != int(scall.values[0][0]):
#            parentTrck = MorphData.morph_parent[f][MorphData.morph_id[f].index(parentTrck)]
#            neu_indBranchTrk_temp.append(parentTrck)
#        if len(neu_indBranchTrk_temp) > 1:
#            neu_indBranchTrk_temp.reverse()
#            neu_indBranchTrk.append(neu_indBranchTrk_temp)
#    BranchData.indBranchTrk.append(neu_indBranchTrk)


#for b in range(len(BranchData.branchTrk)):
#    branch_dist_temp1 = []
#    length_branch_temp = []
#    for sb in range(len(BranchData.branchTrk[b])):
#        dist = 0
#        branch_dist_temp2 = []
#        for sbp in range(len(BranchData.branchTrk[b][sb])):
#            branch_dist_temp2.append(np.array(MorphData.morph_dist[b])[np.where(MorphData.morph_id[b] 
#                                  == np.array(BranchData.branchTrk[b][sb][sbp]))[0]].flatten().tolist())
#            bid = MorphData.morph_id[b].index(BranchData.branchTrk[b][sb][sbp])
#            if MorphData.morph_parent[b][bid] != -1:
#                pid = MorphData.morph_id[b].index(MorphData.morph_parent[b][bid])
#                rhs = MorphData.morph_dist[b][pid]
#                lhs = MorphData.morph_dist[b][bid]
#                
#                dist += np.linalg.norm(np.subtract(rhs, lhs))
#        branch_dist_temp2.reverse()
#        branch_dist_temp1.append(branch_dist_temp2)
#        length_branch_temp.append(dist)
#    BranchData.branch_dist.append(branch_dist_temp1)
#    LengthData.length_branch.append(length_branch_temp)

#branch_dist_flat = [item for sublist in branch_dist for item in sublist]

LengthData.length_branch_flat = [item for sublist in LengthData.length_branch for item in sublist]
LengthData.length_average = np.empty(len(fp))

for lb in range(len(LengthData.length_branch)):
    LengthData.length_total[lb] = np.sum(LengthData.length_branch[lb])
    LengthData.length_average[lb] = np.average(LengthData.length_branch[lb])

MorphData.morph_dist_len = np.array([len(arr) for arr in MorphData.morph_dist])
MorphData.morph_dist_len_EP = np.empty((len(MorphData.morph_dist_len)))
MorphData.endP_len = [len(arr) for arr in MorphData.endP]

#indMorph_dist_p_us = []
#indMorph_dist_id = []
#indMorph_dist_id_s = []
#indMorph_dist_id_i = []
#indMorph_dist_id_m = []

#for i in range(len(BranchData.indBranchTrk)):
#    indMorph_dist_temp1 = []
#    for j in range(len(BranchData.indBranchTrk[i])):
#        indMorph_dist_temp2 = []
#        indMorph_dist_p_us.append(1/len(BranchData.indBranchTrk[i]))
#        for k in range(len(BranchData.indBranchTrk[i][j])):
#            indMorph_dist_temp2.append(np.array(MorphData.morph_dist[i])[np.where(MorphData.morph_id[i] 
#                                    == np.array(BranchData.indBranchTrk[i][j][k]))[0]].flatten().tolist())
#    
#        indMorph_dist_id.append(i)
#        if i in MorphData.sensory:
#            indMorph_dist_id_s.append(len(indMorph_dist_id)-1)
#        elif i in MorphData.inter:
#            indMorph_dist_id_i.append(len(indMorph_dist_id)-1)
#        elif i in MorphData.motor:
#            indMorph_dist_id_m.append(len(indMorph_dist_id)-1)
#            
#        indMorph_dist_temp1.append(indMorph_dist_temp2)
#    BranchData.indMorph_dist.append(indMorph_dist_temp1)

#BranchData.indMorph_dist_p_us = np.array(indMorph_dist_p_us)
#BranchData.indMorph_dist_flat = [item for sublist in BranchData.indMorph_dist for item in sublist]

#t1 = time.time()

#print('checkpoint 1: ' + str(t1-t0))

np.random.seed(Parameter.SEED)

#(MorphData.regMDist, MorphData.regMDistLen) = utils.segmentMorph(Parameter, BranchData)
#(MorphData.indRegMDist, MorphData.indRegMDistLen) = utils.indSegmentMorph(Parameter, BranchData)

t2 = time.time()

#print('checkpoint 2: ' + str(t2-t1))

(rGy, cML) = utils.radiusOfGyration(MorphData)

#(rGyEP, cMLEP) = utils.endPointRadiusOfGyration(MorphData, BranchData)

t3 = time.time()

print('checkpoint 3: ' + str(t3-t2))

#(rGyReg, cMLReg) = utils.regularRadiusOfGyration(MorphData.regMDist, MorphData.regMDistLen)

#t4 = time.time()

#print('checkpoint 4: ' + str(t4-t3))

#if Parameter.RUN:
#    (OutputData.rGyRegSeg, 
#     OutputData.cMLRegSeg, 
#     OutputData.regSegOrdN, 
#     OutputData.randTrks) = utils.regularSegmentRadiusOfGyration(Parameter,
#                        BranchData,
#                        np.array(MorphData.indRegMDist), 
#                        MorphData.indRegMDistLen, 
#                        numSample=Parameter.numSample,
#                        stochastic=True)
#    if Parameter.SAVE:
#        utils.exportOutput(Parameter, OutputData)
#        
#else:
#    (OutputData.rGyRegSegs, OutputData.regSegOrdNs, OutputData.randTrks, 
#     OutputData.rGyRegSegi, OutputData.regSegOrdNi, OutputData.randTrki, 
#     OutputData.rGyRegSegm, OutputData.regSegOrdNm, OutputData.randTrkm) = utils.importData(Parameter)
#
#OutputData.rGyRegSeg = np.concatenate((OutputData.rGyRegSegs, 
#                                       OutputData.rGyRegSegi,
#                                       OutputData.rGyRegSegm))
#OutputData.regSegOrdN = np.concatenate((OutputData.regSegOrdNs,
#                                        OutputData.regSegOrdNi, 
#                                        OutputData.regSegOrdNm))

#t5 = time.time()

#print('checkpoint 5: ' + str(t5-t4))



if Parameter.PLOT:

    fig, ax = plt.subplots(1, 2, figsize=(20,6))
    hist0 = ax[0].hist(LengthData.length_total, 
              bins=int((np.max(LengthData.length_total) - np.min(LengthData.length_total))/10),
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
              bins=int((np.max(LengthData.length_average) - np.min(LengthData.length_average))/10),
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
    
    fig, ax = plt.subplots(4, 1, figsize=(18,24))
    ax[0][0].scatter(LengthData.length_total, BranchData.branchNum)
    ax[0][0].set_xlabel("Total Length", fontsize=15)
    ax[0][0].set_ylabel("Number of Branches", fontsize=15)
#    ax[0][0].set_xlim(-50, 1000)
#    ax[0][0].set_ylim(-1, 8)

    ax[1][0].scatter(LengthData.length_average, BranchData.branchNum)
    ax[1][0].set_xlabel("Average Segment Length", fontsize=15)
    ax[1][0].set_ylabel("Number of Branches", fontsize=15)
#    ax[1][0].set_xlim(-50, 1000)
#    ax[1][0].set_ylim(-1, 8)
    
    for i in range(len(np.unique(BranchData.branchNum))):
        scttrInd = np.where(BranchData.branchNum ==
                            np.unique(BranchData.branchNum)[i])[0]
        ax[2][0].scatter(LengthData.length_average[scttrInd], 
                         LengthData.length_total[scttrInd])
        fitX = np.linspace(0, 1000, 1000)
    ax[2][0].set_xlabel("Average Segment Length", fontsize=15)
    ax[2][0].set_ylabel("Total Length", fontsize=15)
    ax[2][0].legend(np.unique(BranchData.branchNum)[:-1], fontsize=15)
    for i in range(len(np.unique(BranchData.branchNum))):
        scttrInd = np.where(BranchData.branchNum == 
                            np.unique(BranchData.branchNum)[i])[0]
        if np.unique(BranchData.branchNum)[i] == 0:
            fitY = objFuncL(fitX, 1)
            ax[2][0].plot(fitX, fitY)
        elif (np.unique(BranchData.branchNum)[i] == 1 or 
              np.unique(BranchData.branchNum)[i] == 2):
            popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                    LengthData.length_average[scttrInd], 
                                                    LengthData.length_total[scttrInd],
                                                    p0=[1.],
                                                    maxfev=10000)
            fitY = objFuncL(fitX, popt[0])
            ax[2][0].plot(fitX, fitY)
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
        ax[3][0].scatter([item for sublist in np.array(LengthData.length_branch)[scttrInd].tolist() for item in sublist], 
                         repeated_length_total_sensory)
    ax[3][0].set_xlabel("Segment Length", fontsize=15)
    ax[3][0].set_ylabel("Total Length", fontsize=15)
    ax[3][0].legend(np.unique(BranchData.branchNum)[:-1], fontsize=15)
    for i in range(len(np.unique(BranchData.branchNum))):
        scttrInd = np.where(BranchData.branchNum == 
                            np.unique(BranchData.branchNum)[i])[0]
        length_branch_len_sensory = [len(arr) for arr in np.array(LengthData.length_branch)[scttrInd]]
        repeated_length_total_sensory = np.repeat(LengthData.length_total[scttrInd], 
                                                  length_branch_len[scttrInd])
        if np.unique(BranchData.branchNum)[i] == 0:
            fitY = objFuncL(fitX, 1)
            ax[3][0].plot(fitX, fitY)
        elif (np.unique(BranchData.branchNum)[i] == 1 or 
            np.unique(BranchData.branchNum)[i] == 2):
            popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                  [item for sublist in 
                                                   np.array(LengthData.length_branch)[scttrInd].tolist() for item in sublist], 
                                                  repeated_length_total_sensory,
                                                  p0=[1.],
                                                  maxfev=10000)
            fitY = objFuncL(fitX, popt[0])
            ax[3][0].plot(fitX, fitY)
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
    plt.legend(['Sensory', 'Inter', 'Motor'], fontsize=15)
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
                                            np.log10(MorphData.morph_dist_len), 
                                            np.log10(np.sqrt(np.square(rGy))), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR = objFuncPpow(MorphData.morph_dist_len, poptR[0], poptR[1])
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(MorphData.morph_dist_len, np.sqrt(np.square(rGy)))
    plt.plot(MorphData.morph_dist_len, fitYregR, color='tab:red')
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
    
    
    rGyRegSeg_avg = np.empty(len(Parameter.nSize))
    for i in range(len(Parameter.nSize)):
        RStemp = np.where(OutputData.regSegOrdN == Parameter.nSize[i])[0]
        rGyRegSeg_avg[i] = np.average(OutputData.rGyRegSeg[RStemp])
    
    RS1 = np.where(OutputData.regSegOrdN > 7)[0]
    RS2 = np.where((OutputData.regSegOrdN <= 8) & (OutputData.regSegOrdN >= 4))[0]
    RS3 = np.where(OutputData.regSegOrdN < 5)[0]
    
    poptRS1, pcovRS1 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(OutputData.regSegOrdN[RS1]*Parameter.sSize), 
                                              np.log10(np.sqrt(np.square(OutputData.rGyRegSeg[RS1])*1/Parameter.sSize)), 
                                              p0=[1., 0.], 
                                              maxfev=100000)
    fitYregRS1 = objFuncPpow(np.unique(OutputData.regSegOrdN[RS1])*Parameter.sSize, poptRS1[0], poptRS1[1])
    
    fitYregRS12 = objFuncPpow(np.unique(OutputData.regSegOrdN[RS2])*Parameter.sSize, poptRS1[0], poptRS1[1])
    
    poptRS2, pcovRS2 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(OutputData.regSegOrdN[RS2]*Parameter.sSize), 
                                              np.log10(np.sqrt(np.square(OutputData.rGyRegSeg[RS2])*1/Parameter.sSize)), 
                                              p0=[1., 0.], 
                                              maxfev=100000)
    fitYregRS2 = objFuncPpow(np.unique(OutputData.regSegOrdN[RS2])*Parameter.sSize, poptRS2[0], poptRS2[1])
    
    poptRS3, pcovRS3 = scipy.optimize.curve_fit(objFuncGL, 
                                              np.log10(OutputData.regSegOrdN[RS3]*Parameter.sSize), 
                                              np.log10(np.sqrt(np.square(OutputData.rGyRegSeg[RS3])*1/Parameter.sSize)), 
                                              p0=[1., 0.], 
                                              maxfev=100000)
    fitYregRS3 = objFuncPpow(np.unique(OutputData.regSegOrdN[RS3])*Parameter.sSize, poptRS3[0], poptRS3[1])
    
    fitYregRS32 = objFuncPpow(np.unique(OutputData.regSegOrdN[RS2])*Parameter.sSize, poptRS3[0], poptRS3[1])
    
    
    fig, ax1 = plt.subplots(figsize=(12,8))
    ax1.xaxis.label.set_fontsize(15)
    ax1.xaxis.set_tick_params(which='major', length=7)
    ax1.xaxis.set_tick_params(which='minor', length=5)
    ax1.yaxis.label.set_fontsize(15)
    ax1.yaxis.set_tick_params(which='major', length=7)
    ax1.yaxis.set_tick_params(which='minor', length=5)
    ax1.scatter(MorphData.regMDistLen*Parameter.sSize, 
                np.sqrt(np.square(rGyReg)*1/Parameter.sSize), color='tab:blue')
    ax1.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR, color='tab:red', lw=2)
    ax1.scatter(OutputData.regSegOrdN*Parameter.sSize, 
                np.sqrt(np.square(OutputData.rGyRegSeg)*1/Parameter.sSize), 
                color='tab:blue',
                facecolors='none')
    ax1.scatter(np.array(Parameter.nSize)*Parameter.sSize, 
                np.sqrt(np.square(rGyRegSeg_avg)*1/Parameter.sSize), 
                color='tab:orange')
    ax1.plot(np.unique(OutputData.regSegOrdN[RS1])*Parameter.sSize, fitYregRS1, color='tab:red', lw=2, linestyle='--')
    ax1.plot(np.unique(OutputData.regSegOrdN[RS2])*Parameter.sSize, fitYregRS2, color='tab:red', lw=2, linestyle='--')
    ax1.plot(np.unique(OutputData.regSegOrdN[RS3])*Parameter.sSize, fitYregRS3, color='tab:red', lw=2, linestyle='--')
    ax1.vlines(0.8, 0.01, 11000, linestyles='dashed')
    ax1.vlines(0.4, 0.01, 11000, linestyles='dashed')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    #ax1.xlim(0.01, 10500)
    ax1.set_ylim(0.03, 10000)
    
    ax2 = plt.axes([0, 0, 1, 1])
    ip1 = InsetPosition(ax1, [0.01, 0.57, 0.4, 0.4])
    ax2.set_axes_locator(ip1)
    mark_inset(ax1, ax2, loc1=3, loc2=4, fc="none", ec='0.5')
    
    ax2.scatter(OutputData.regSegOrdN*Parameter.sSize, 
                np.sqrt(np.square(OutputData.rGyRegSeg)*1/Parameter.sSize), 
                color='tab:blue', 
                facecolors='none')
    ax2.scatter(np.array(Parameter.nSize)[1:11]*Parameter.sSize,
                np.sqrt(np.square(rGyRegSeg_avg)[1:11]*1/Parameter.sSize), 
                color='tab:orange')
    ax2.plot(np.unique(OutputData.regSegOrdN[RS1])[:4]*Parameter.sSize, fitYregRS1[:4], color='tab:red', lw=2, linestyle='--')
    ax2.plot(np.unique(OutputData.regSegOrdN[RS2])*Parameter.sSize, fitYregRS2, color='tab:red', lw=2, linestyle='--')
    ax2.plot(np.unique(OutputData.regSegOrdN[RS3])*Parameter.sSize, fitYregRS3, color='tab:red', lw=2, linestyle='--')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.vlines(0.8, 0.01, 5, linestyles='dashed')
    ax2.vlines(0.4, 0.01, 5, linestyles='dashed')
    ax2.set_xlim(0.24, 1.4)
    ax2.set_ylim(0.28, 1.6)
    
    ax3 = plt.axes([1, 1, 2, 2])
    ip2 = InsetPosition(ax1, [0.57, 0.02, 0.4, 0.4])
    ax3.set_axes_locator(ip2)
    mark_inset(ax1, ax3, loc1=2, loc2=3, fc="none", ec='0.5')
    
    ax3.plot(np.unique(OutputData.regSegOrdN[RS1])[:4]*Parameter.sSize, fitYregRS1[:4], color='tab:red', lw=2, linestyle='-')
    ax3.plot(np.unique(OutputData.regSegOrdN[RS2])*Parameter.sSize, fitYregRS12, color='tab:red', lw=2, linestyle='--')
    ax3.plot(np.unique(OutputData.regSegOrdN[RS2])*Parameter.sSize, fitYregRS2, color='tab:green', lw=2, linestyle='-')
    ax3.plot(np.unique(OutputData.regSegOrdN[RS3])*Parameter.sSize, fitYregRS3, color='tab:blue', lw=2, linestyle='-')
    ax3.plot(np.unique(OutputData.regSegOrdN[RS2])*Parameter.sSize, fitYregRS32, color='tab:blue', lw=2, linestyle='--')
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.vlines(0.8, 0.01, 1, linestyles='dashed')
    ax3.vlines(0.4, 0.01, 1, linestyles='dashed')
    ax3.set_xlim(0.36, 0.89)
    ax3.set_ylim(0.42, 0.95)
    
    ax1.set_xlabel(r"Length ($\lambda N$)", fontsize=15)
    ax1.set_ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
    #plt.tight_layout()
    if Parameter.SAVE:
        plt.savefig('./images/regSegRG_morphScale_' + str(Parameter.RN) + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    
    #==============================================================================
    
    
    
    fig, ax1 = plt.subplots(figsize=(12,8))
    ax1.xaxis.label.set_fontsize(15)
    ax1.xaxis.set_tick_params(which='major', length=7)
    ax1.xaxis.set_tick_params(which='minor', length=5)
    ax1.yaxis.label.set_fontsize(15)
    ax1.yaxis.set_tick_params(which='major', length=7)
    ax1.yaxis.set_tick_params(which='minor', length=5)
    #ax1.scatter(regMDistLen[MorphData.sensory]*Parameter.sSize, np.sqrt(np.square(rGyReg)[MorphData.sensory]*1/Parameter.sSize))
    #ax1.scatter(regMDistLen[MorphData.inter]*Parameter.sSize, np.sqrt(np.square(rGyReg)[MorphData.inter]*1/Parameter.sSize))
    #ax1.scatter(regMDistLen[MorphData.motor]*Parameter.sSize, np.sqrt(np.square(rGyReg)[MorphData.motor]*1/Parameter.sSize))
    ax1.scatter(OutputData.regSegOrdNi*Parameter.sSize, 
                np.sqrt(np.square(OutputData.rGyRegSegi)*1/Parameter.sSize), 
                color='tab:orange',
                facecolors='none')
    ax1.scatter(OutputData.regSegOrdNm*Parameter.sSize, 
                np.sqrt(np.square(OutputData.rGyRegSegm)*1/Parameter.sSize),
                color='tab:green', 
                facecolors='none')
    ax1.scatter(OutputData.regSegOrdNs*Parameter.sSize,
                np.sqrt(np.square(OutputData.rGyRegSegs)*1/Parameter.sSize),
                color='tab:blue', 
                facecolors='none')
    ax1.legend(["Sensory Neuron", "Interneuron", "Motor Neuron"], fontsize=15)
    ax1.vlines(0.8, 0.01, 11000, linestyles='dashed')
    ax1.vlines(0.4, 0.01, 11000, linestyles='dashed')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    #ax1.xlim(0.01, 10500)
    ax1.set_ylim(0.03, 100)
    ax1.set_xlabel(r"Length ($\lambda N$)", fontsize=15)
    ax1.set_ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
    if Parameter.SAVE:
        plt.savefig('./images/regSegRG_morphScale_sep_' + str(Parameter.RN) + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    #==============================================================================
    
    
    shift_N = 4
    poptRS_sl = []
    RS_x = []
    for i in range(len(Parameter.nSize) - shift_N):
        RS_s = np.where((OutputData.regSegOrdN <= Parameter.nSize[i+shift_N]) &
                        (OutputData.regSegOrdN >= Parameter.nSize[i]))[0]
        
        RS_x.append(np.average(Parameter.nSize[i:i+shift_N]))
        
        poptRS_s, pcovRS_s = scipy.optimize.curve_fit(objFuncGL, 
                                                      np.log10(OutputData.regSegOrdN[RS_s]*Parameter.sSize), 
                                                      np.log10(np.sqrt(np.square(OutputData.rGyRegSeg[RS_s])*1/Parameter.sSize)), 
                                                      p0=[1., 0.], 
                                                      maxfev=100000)
        poptRS_sl.append(poptRS_s[0])
    
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(RS_x, poptRS_sl)
    #plt.plot(regMDistLen*Parameter.sSize, fitYregR, color='tab:red')
    #plt.yscale('log')
    plt.hlines(poptR[0], 0.1, 1000, linestyles='--', color='tab:red')
    plt.hlines(poptRS1[0], 0.1, 1000, linestyles='--', color='tab:green')
    plt.hlines(poptRS3[0], 0.1, 1000, linestyles='--', color='tab:orange')
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1, 200)
    #plt.ylim(0.005, 1000)
    #plt.title(r"Scaling Behavior of Regularized $R_{g}$ to Regularized $N$", fontsize=20)
    plt.xlabel(r"Average Length ($\lambda N_{avg}$)", fontsize=15)
    plt.ylabel(r"Slope ($\nu$)", fontsize=15)
    #plt.tight_layout()
    if Parameter.SAVE:
        plt.savefig('./images/regSegRG_slope_' + str(Parameter.RN) + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    #==============================================================================
    
    
    poptRS_sl_sep_sen = []
    poptRS_sl_sep_int = []
    poptRS_sl_sep_mot = []
    RS_x_sep = []
    for i in range(len(Parameter.nSize) - shift_N):
        RS_s_sep_sen = np.where((OutputData.regSegOrdNs <= Parameter.nSize[i+shift_N]) & 
                                (OutputData.regSegOrdNs >= Parameter.nSize[i]))[0]
        RS_s_sep_int = np.where((OutputData.regSegOrdNi <= Parameter.nSize[i+shift_N]) & 
                                (OutputData.regSegOrdNi >= Parameter.nSize[i]))[0]
        RS_s_sep_mot = np.where((OutputData.regSegOrdNm <= Parameter.nSize[i+shift_N]) &
                                (OutputData.regSegOrdNm >= Parameter.nSize[i]))[0]
        
        RS_x_sep.append(np.average(Parameter.nSize[i:i+shift_N]))
        
        poptRS_s_sep_sen, pcovRS_s_sep_sep = scipy.optimize.curve_fit(objFuncGL, 
                                                      np.log10(OutputData.regSegOrdNs[RS_s_sep_sen]*Parameter.sSize), 
                                                      np.log10(np.sqrt(np.square(OutputData.rGyRegSegs[RS_s_sep_sen])*1/Parameter.sSize)), 
                                                      p0=[1., 0.], 
                                                      maxfev=100000)
        poptRS_s_sep_int, pcovRS_s_sep_int = scipy.optimize.curve_fit(objFuncGL, 
                                                      np.log10(OutputData.regSegOrdNi[RS_s_sep_int]*Parameter.sSize), 
                                                      np.log10(np.sqrt(np.square(OutputData.rGyRegSegi[RS_s_sep_int])*1/Parameter.sSize)), 
                                                      p0=[1., 0.], 
                                                      maxfev=100000)
        poptRS_s_sep_mot, pcovRS_s_sep_mot = scipy.optimize.curve_fit(objFuncGL, 
                                                      np.log10(OutputData.regSegOrdNm[RS_s_sep_mot]*Parameter.sSize), 
                                                      np.log10(np.sqrt(np.square(OutputData.rGyRegSegm[RS_s_sep_mot])*1/Parameter.sSize)), 
                                                      p0=[1., 0.], 
                                                      maxfev=100000)
        
        poptRS_sl_sep_sen.append(poptRS_s_sep_sen[0])
        poptRS_sl_sep_int.append(poptRS_s_sep_int[0])
        poptRS_sl_sep_mot.append(poptRS_s_sep_mot[0])
    
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(RS_x_sep, poptRS_sl_sep_sen)
    plt.scatter(RS_x_sep, poptRS_sl_sep_int)
    plt.scatter(RS_x_sep, poptRS_sl_sep_mot)
    plt.legend(["Sensory Neuron", "Interneuron", "Motor Neuron"], fontsize=15)
    #plt.plot(regMDistLen*Parameter.sSize, fitYregR, color='tab:red')
    #plt.yscale('log')
    #plt.hlines(poptR[0], 0.1, 1000, linestyles='--', color='tab:red')
    #plt.hlines(poptRS1[0], 0.1, 1000, linestyles='--', color='tab:green')
    #plt.hlines(poptRS3[0], 0.1, 1000, linestyles='--', color='tab:orange')
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1, 200)
    #plt.ylim(0.005, 1000)
    #plt.title(r"Scaling Behavior of Regularized $R_{g}$ to Regularized $N$", fontsize=20)
    plt.xlabel(r"Average Length ($\lambda N_{avg}$)", fontsize=15)
    plt.ylabel(r"Slope ($\nu$)", fontsize=15)
    #plt.tight_layout()
    if Parameter.SAVE:
        plt.savefig('./images/regSegRG_slope_sep_' + str(Parameter.RN) + '.png', dpi=300, bbox_inches='tight')
    plt.show()







#==============================================================================

#sRnTrkIRge = np.array(randTrk)[np.array(sChoice)[np.where((np.array(sChoice) > 180000) & (np.array(sChoice) < 210000))[0]]]




#fig = plt.figure()
#ax = plt.gca()
#ax.scatter(cTargetCVal, BranchData.branchNum[cTargetNeuronCorr])
##ax.scatter(cTargetCVal, LengthData.length_total[cTargetNeuronCorr])
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.show()
#
#fig = plt.figure()
#ax = plt.gca()
#ax.scatter(np.array(cTargetCVal)[cTargetS], BranchData.branchNum[cTargetNeuronCorrS])
##ax.scatter(np.array(cTargetCVal)[cTargetNeuronCorrS], LengthData.length_total[MorphData.sensory])
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.show()
#
#fig = plt.figure()
#ax = plt.gca()
#ax.scatter(np.array(cTargetCVal)[cTargetI], BranchData.branchNum[cTargetNeuronCorrI])
##ax.scatter(np.array(cTargetCVal)[cTargetNeuronCorrS], LengthData.length_total[MorphData.sensory])
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.show()
#
#fig = plt.figure()
#ax = plt.gca()
#ax.scatter(np.array(cTargetCVal)[cTargetM], BranchData.branchNum[cTargetNeuronCorrM])
##ax.scatter(np.array(cTargetCVal)[cTargetNeuronCorrS], LengthData.length_total[MorphData.sensory])
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.show()


