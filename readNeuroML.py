# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:32:04 2019

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

PATH = r'./CElegansNeuroML-SNAPSHOT_030213/CElegans/generatedNeuroML2'

RUN = True
SAVE = True
RN = '4'


fp = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]
fp = [f for f in fp if "Acetylcholine" not in f]
fp = [f for f in fp if "CElegans" not in f]
fp = [f for f in fp if "Dopamine" not in f]
fp = [f for f in fp if "FMRFamide" not in f]
fp = [f for f in fp if "GABA" not in f]
fp = [f for f in fp if "Glutamate" not in f]
fp = [f for f in fp if "LeakConductance" not in f]
fp = [f for f in fp if "Octapamine" not in f]
fp = [f for f in fp if "Serotonin" not in f]
fp = [f for f in fp if "README" not in f]
fp = [os.path.join(PATH, f) for f in fp]

morph_id = []
morph_parent = []
morph_prox = []
morph_dist = []
length_total = []
length_branch = []
length_direct = []
branchTrk = []
branch_dist = []
indMorph_dist = []
indBranchTrk = []
branchP = []
endP = []
somaP = []
neuron_id = []
neuron_type = []
branchNum = []
sensory = []
inter = []
motor = []
polymodal = []
other = []

t0 = time.time()

for f in range(len(fp)):
    morph_neu_id = []
    morph_neu_parent = []
    morph_neu_prox = []
    morph_neu_dist = []
    doc = loaders.NeuroMLLoader.load(fp[f])
    neuron_id.append(doc.cells[0].id)
    neuron_type.append(doc.cells[0].notes.strip())
    if doc.cells[0].notes.strip() == "SensoryNeuron":
        sensory.append(f)
    elif doc.cells[0].notes.strip() == "Interneuron":
        inter.append(f)
    elif doc.cells[0].notes.strip() == "Motor Neuron":
        motor.append(f)
    elif doc.cells[0].notes.strip() == "PolymodalNeuron":
        polymodal.append(f)
    else:
        other.append(f)
    sgmts = doc.cells[0].morphology
    for s in range(sgmts.num_segments):
        sgmt = doc.cells[0].morphology.segments[s]
        morph_neu_id.append(sgmt.id)
        if sgmt.parent != None:
            morph_neu_parent.append(sgmt.parent.segments)
        else:
            morph_neu_parent.append(-1)
            somaP.append(s)
        if sgmt.proximal != None:
            morph_neu_prox.append([sgmt.proximal.x, 
                                   sgmt.proximal.y, 
                                   sgmt.proximal.z, 
                                   sgmt.proximal.diameter])
        else:
            morph_neu_prox.append([])
        if sgmt.distal != None:
            morph_neu_dist.append([sgmt.distal.x, 
                                   sgmt.distal.y, 
                                   sgmt.distal.z, 
                                   sgmt.distal.diameter])
        else:
            morph_neu_dist.append([])
    
    morph_id.append(morph_neu_id)
    morph_parent.append(morph_neu_parent)
    morph_prox.append(morph_neu_prox)
    morph_dist.append(morph_neu_dist)
    ctr = Counter(morph_neu_parent)
    ctrVal = list(ctr.values())
    ctrKey = list(ctr.keys())
    branchNum.append(sum(i > 1 for i in ctrVal))
    branchInd = np.array(ctrKey)[np.where(np.array(ctrVal) > 1)[0]]
    
    neu_branchTrk = []
    neu_indBranchTrk = []
    
    list_end = np.setdiff1d(morph_id[f], morph_parent[f])
    
    branchP.append(branchInd)
    endP.append(list_end)
    bPoint = np.append(branchInd, list_end)
    
    for bp in range(len(bPoint)):
        neu_branchTrk_temp = []
        neu_branchTrk_temp.append(bPoint[bp])
        parentTrck = bPoint[bp]
        while (parentTrck not in branchInd or bPoint[bp] in branchInd) and (parentTrck != -1):
            parentTrck = morph_parent[f][morph_id[f].index(parentTrck)]
            if parentTrck != -1:
                neu_branchTrk_temp.append(parentTrck)
        if len(neu_branchTrk_temp) > 1:
            neu_branchTrk.append(neu_branchTrk_temp)
    branchTrk.append(neu_branchTrk)
    
    for ep in range(len(list_end)):
        neu_indBranchTrk_temp = []
        neu_indBranchTrk_temp.append(list_end[ep])
        parentTrck = list_end[ep]
        while parentTrck != 0:
            parentTrck = morph_parent[f][morph_id[f].index(parentTrck)]
            neu_indBranchTrk_temp.append(parentTrck)
        if len(neu_indBranchTrk_temp) > 1:
            neu_indBranchTrk_temp.reverse()
            neu_indBranchTrk.append(neu_indBranchTrk_temp)
    indBranchTrk.append(neu_indBranchTrk)


for b in range(len(branchTrk)):
    branch_dist_temp1 = []
    length_branch_temp = []
    for sb in range(len(branchTrk[b])):
        dist = 0
        branch_dist_temp2 = []
        for sbp in range(len(branchTrk[b][sb])):
            branch_dist_temp2.append(np.array(morph_dist[b])[np.where(morph_id[b] 
                                  == np.array(branchTrk[b][sb][sbp]))[0]].flatten().tolist())
            bid = morph_id[b].index(branchTrk[b][sb][sbp])
            if morph_parent[b][bid] != -1:
                pid = morph_id[b].index(morph_parent[b][bid])
                rhs = morph_dist[b][pid][:3]
                lhs = morph_dist[b][bid][:3]
                
                dist += np.linalg.norm(np.subtract(rhs, lhs))
        branch_dist_temp2.reverse()
        branch_dist_temp1.append(branch_dist_temp2)
        length_branch_temp.append(dist)
    branch_dist.append(branch_dist_temp1)
    length_branch.append(length_branch_temp)

#branch_dist_flat = [item for sublist in branch_dist for item in sublist]

length_branch_flat = [item for sublist in length_branch for item in sublist]
length_branch_sensory = []
length_branch_inter = []
length_branch_motor = []
length_branch_polymodal = []
length_branch_other = []
length_average = []

for lb in range(len(length_branch)):
    length_total.append(np.sum(length_branch[lb]))
    length_average.append(np.average(length_branch[lb]))
    if lb in sensory:
        length_branch_sensory.append(length_branch[lb])
    elif lb in inter:
        length_branch_inter.append(length_branch[lb])
    elif lb in motor:
        length_branch_motor.append(length_branch[lb])
    elif lb in polymodal:
        length_branch_polymodal.append(length_branch[lb])
    elif lb in other:
        length_branch_other.append(length_branch[lb])


length_branch_sensory_flat = [item for sublist in length_branch_sensory for item in sublist]
length_branch_inter_flat = [item for sublist in length_branch_inter for item in sublist]
length_branch_motor_flat = [item for sublist in length_branch_motor for item in sublist]
length_branch_polymodal_flat = [item for sublist in length_branch_polymodal for item in sublist]
length_branch_other_flat = [item for sublist in length_branch_other for item in sublist]

morph_dist_len = [len(arr) for arr in morph_dist]
morph_dist_len_EP = np.empty((len(morph_dist_len)))
endP_len = [len(arr) for arr in endP]

indMorph_dist_p = []
indMorph_dist_id = []

for i in range(len(indBranchTrk)):
    indMorph_dist_temp1 = []
    for j in range(len(indBranchTrk[i])):
        indMorph_dist_temp2 = []
        indMorph_dist_p.append(1/len(indBranchTrk[i]))
        for k in range(len(indBranchTrk[i][j])):
            indMorph_dist_temp2.append(np.array(morph_dist[i])[np.where(morph_id[i] 
                                    == np.array(indBranchTrk[i][j][k]))[0]].flatten().tolist())
    
        indMorph_dist_id.append(i)
        indMorph_dist_temp1.append(indMorph_dist_temp2)
    indMorph_dist.append(indMorph_dist_temp1)

indMorph_dist_flat = [item for sublist in indMorph_dist for item in sublist]

indMorph_dist_p = indMorph_dist_p/np.sum(indMorph_dist_p)

t1 = time.time()

print('checkpoint 1: ' + str(t1-t0))

def segmentMorph(sSize):
    regMDist = []
    
    for i in range(len(branch_dist)):
        regMDist_temp1 = []
        for k in range(len(branch_dist[i])):
            regMDist_temp2 = []
            for j in range(len(branch_dist[i][k])-1):
                dist = np.linalg.norm(np.array(branch_dist[i][k])[j+1][:3]-
                                      np.array(branch_dist[i][k])[j][:3])
                l1 = np.linspace(0,1,max(1, int(dist/sSize)))
                nArr = np.array(branch_dist[i][k])[j][:3]+(np.array(branch_dist[i][k])[j+1][:3]-
                                np.array(branch_dist[i][k])[j][:3])*l1[:,None]
                regMDist_temp2.append(nArr.tolist())
            regMDist_temp1.append([item for sublist in regMDist_temp2 for item in sublist])
        regMDist_temp_flatten = [item for sublist in regMDist_temp1 for item in sublist]
        _, Uidx = np.unique(np.array(regMDist_temp_flatten), return_index=True, axis=0)
        uniqueUSorted = np.array(regMDist_temp_flatten)[np.sort(Uidx)].tolist()
        regMDist.append(uniqueUSorted)
    
    regMDistLen = [len(arr) for arr in regMDist]
    
    return regMDist, regMDistLen
   
def indSegmentMorph(sSize):
    indRegMDist = []
    
    for i in range(len(indMorph_dist_flat)):
        indRegMDist_temp = []
        for j in range(len(indMorph_dist_flat[i])-1):
            dist = np.linalg.norm(np.array(indMorph_dist_flat[i])[j+1][:3]-
                                  np.array(indMorph_dist_flat[i])[j][:3])
            l1 = np.linspace(0,1,max(1, int(dist/sSize)))
            nArr = np.array(indMorph_dist_flat[i])[j][:3]+(np.array(indMorph_dist_flat[i])[j+1][:3]-
                            np.array(indMorph_dist_flat[i])[j][:3])*l1[:,None]
            indRegMDist_temp.append(nArr.tolist())
        indRegMDist_temp_flatten = [item for sublist in indRegMDist_temp for item in sublist]
        _, Uidx = np.unique(np.array(indRegMDist_temp_flatten), return_index=True, axis=0)
        uniqueUSorted = np.array(indRegMDist_temp_flatten)[np.sort(Uidx)].tolist()
        indRegMDist.append(uniqueUSorted)
    
    indRegMDistLen = [len(arr) for arr in indRegMDist]
    
    return indRegMDist, indRegMDistLen


def radiusOfGyration():
    cML = []
    rGy = []
    for i in range(len(morph_dist)):
        cML.append(np.sum(np.array(morph_dist[i]), axis=0)[:3]/len(np.array(morph_dist[i])))
        rList = scipy.spatial.distance.cdist(np.array(morph_dist[i])[:,:3], np.array([cML[i]])).flatten()
        rGy.append(np.sqrt(np.sum(np.square(rList))/len(rList)))
    
    return (rGy, cML)

def endPointRadiusOfGyration():
    cMLEP = []
    rGyEP = []
    for i in range(len(morph_dist)):
        distInd = np.where(np.isin(np.unique(np.hstack([endP[i], somaP[i], branchP[i]])), morph_id[i]))[0]
        morph_dist_len_EP[i] = len(distInd)
        cMLEP.append(np.sum(np.array(morph_dist[i])[distInd], axis=0)[:3]/len(np.array(morph_dist[i])[distInd]))
        rList_EP = scipy.spatial.distance.cdist(np.array(morph_dist[i])[distInd,:3], np.array([cMLEP[i]])).flatten()
        rGyEP.append(np.sqrt(np.sum(np.square(rList_EP))/len(rList_EP)))
    
    return (rGyEP, cMLEP)

def regularRadiusOfGyration(regMDist, regMDistLen):
    
    cMLReg = []
    rGyReg = []
    for i in range(len(regMDist)):
        cMLReg.append(np.sum(np.array(regMDist[i]), axis=0)/regMDistLen[i])
        rList_reg = scipy.spatial.distance.cdist(np.array(regMDist[i]), np.array([cMLReg[i]])).flatten()
        rGyReg.append(np.sqrt(np.sum(np.square(rList_reg))/regMDistLen[i]))
    
    return (rGyReg, cMLReg)

def regularSegmentRadiusOfGyration(indRegMDist, indRegMDistLen, nSize, dSize, stochastic=True):

    cMLRegSeg = []
    rGyRegSeg = []
    nSize = np.array(nSize)+1
    regSegOrdN = []
    sensory_choice = []
    inter_choice = []
    motor_choice = []
    
    if stochastic:
        for i in range(len(nSize)):
            for j in range(10000):
                randIdx1 = np.random.choice(np.arange(0, len(indRegMDist)), 
                                            1, 
                                            p=indMorph_dist_p)[0]
                while len(indRegMDist[randIdx1]) <= nSize[i]:
                    randIdx1 = np.random.choice(np.arange(0, len(indRegMDist)),
                                                1, 
                                                p=indMorph_dist_p)[0]
                
                if indMorph_dist_id[randIdx1] in sensory:
                    sensory_choice.append(randIdx1)
                elif indMorph_dist_id[randIdx1] in inter:
                    inter_choice.append(randIdx1)
                elif indMorph_dist_id[randIdx1] in motor:
                    motor_choice.append(randIdx1)
                    
                randIdx2 = np.random.choice(np.arange(0, len(indRegMDist[randIdx1])-nSize[i]), 1)[0]
                
                regSegOrdN.append(nSize[i]-1)
                cMLRegSeg.append(np.sum(np.array(indRegMDist[randIdx1])[randIdx2:randIdx2+nSize[i]], axis=0)/nSize[i])
                rList_reg_seg = scipy.spatial.distance.cdist(np.array(indRegMDist[randIdx1])[randIdx2:randIdx2+nSize[i]], 
                                                             np.array([cMLRegSeg[-1]])).flatten()
                rGyRegSeg.append(np.sqrt(np.sum(np.square(rList_reg_seg))/nSize[i]))
    else:
        for i in range(len(nSize)):
            randIdx = np.sort(np.random.choice(np.arange(0, len(indRegMDistLen)), 
                                               100, 
                                               p=indMorph_dist_p, 
                                               replace=False))
            for j in randIdx:
                dInt = np.arange(0, indRegMDistLen[j]-nSize[i], dSize)
                for k in range(len(dInt)-1):
                    regSegOrdN.append(nSize[i]-1)
                    cMLRegSeg.append(np.sum(np.array(indRegMDist[i])[dInt[k]:dInt[k]+nSize[i]], axis=0)/nSize[i])
                    rList_reg_seg = scipy.spatial.distance.cdist(np.array(indRegMDist[i])[dInt[k]:dInt[k]+nSize[i]], 
                                                                 np.array([cMLRegSeg[-1]])).flatten()
                    rGyRegSeg.append(np.sqrt(np.sum(np.square(rList_reg_seg))/nSize[i]))
        
    return (rGyRegSeg, cMLRegSeg, regSegOrdN, sensory_choice, inter_choice, motor_choice)


np.random.seed(1234)

sSize = 0.01
nSize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 50, 100, 500, 1000]
dSize = 10


(regMDist, regMDistLen) = segmentMorph(sSize)
(indRegMDist, indRegMDistLen) = indSegmentMorph(sSize)

t2 = time.time()

print('checkpoint 2: ' + str(t2-t1))

(rGy, cML) = radiusOfGyration()

(rGyEP, cMLEP) = endPointRadiusOfGyration()

t3 = time.time()

print('checkpoint 3: ' + str(t3-t2))

(rGyReg, cMLReg) = regularRadiusOfGyration(regMDist, regMDistLen)

t4 = time.time()

print('checkpoint 4: ' + str(t4-t3))

if RUN:
    (rGyRegSeg, 
     cMLRegSeg, 
     regSegOrdN, 
     sensory_choice, 
     inter_choice,
     motor_choice) = regularSegmentRadiusOfGyration(indRegMDist, 
                                                    indRegMDistLen, 
                                                    nSize, 
                                                    dSize, 
                                                    stochastic=True)
else:
    rGyRegSeg = np.genfromtxt('./rGyRegSeg_' + str(RN) + '.csv', delimiter=',')
    regSegOrdN = np.genfromtxt('./regSegOrdN_' + str(RN) + '.csv', delimiter=',')
    
t5 = time.time()

print('checkpoint 5: ' + str(t5-t4))

if SAVE:
    np.savetxt('./rGyRegSeg_' + str(RN) + '.csv', rGyRegSeg, delimiter=",")
    np.savetxt('./regSegOrdN_' + str(RN) + '.csv', regSegOrdN, delimiter=",")


fig, ax = plt.subplots(1, 2, figsize=(20,6))
hist1 = ax[0].hist(length_total, 
          bins=int((np.max(length_total) - np.min(length_total))/10),
          density=True)
ax[0].set_title("Distribution of Segment Length", fontsize=20)
ax[0].set_ylabel("Normalized Density", fontsize=15)
ax[0].set_xlabel("Total Length", fontsize=15)
ax[0].set_xlim(0, 1000)
plt.tight_layout()
plt.show()



# Segment Length Histogram

fig, ax = plt.subplots(1, 2, figsize=(20,6))
hist1 = ax[0].hist(length_branch_flat, 
          bins=int((np.max(length_branch_flat) - np.min(length_branch_flat))/10),
          density=True)
ax[0].set_title("Distribution of Segment Length", fontsize=20)
ax[0].set_ylabel("Normalized Density", fontsize=15)
ax[0].set_xlabel("Segment Length", fontsize=15)

hist2 = ax[1].hist(length_branch_sensory_flat, 
                 bins=int((np.max(length_branch_sensory_flat) - np.min(length_branch_sensory_flat))/10), 
                 density=True, 
                 alpha=0.5)
hist3 = ax[1].hist(length_branch_inter_flat, 
                 bins=int((np.max(length_branch_inter_flat) - np.min(length_branch_inter_flat))/10),
                 density=True, 
                 alpha=0.5)
hist4 = ax[1].hist(length_branch_motor_flat,
                 bins=int((np.max(length_branch_motor_flat) - np.min(length_branch_motor_flat))/10), 
                 density=True,
                 alpha=0.5)
ax[1].set_title("Distribution of Segment Length by Type", fontsize=20)
ax[1].set_ylabel("Normalized Density", fontsize=15)
ax[1].set_xlabel("Segment Length", fontsize=15)
ax[1].legend(['Sensory', 'Inter', 'Motor'], fontsize=15)
plt.tight_layout()
plt.show()

hist1centers = 0.5*(hist1[1][1:] + hist1[1][:-1])
hist2centers = 0.5*(hist2[1][1:] + hist2[1][:-1])
hist3centers = 0.5*(hist3[1][1:] + hist3[1][:-1])
hist4centers = 0.5*(hist4[1][1:] + hist4[1][:-1])


def objFuncP(xdata, a, b):
    y = a*np.power(xdata, b)
    
    return y
    
def objFuncPL(xdata, a):
    y = np.power(xdata, a)
    
    return y

def objFuncPpow(xdata, a, b):
    y = np.power(10, b)*np.power(xdata, a)
    
    return y


popt1, pcov1 = scipy.optimize.curve_fit(objFuncP, hist1centers, hist1[0], p0=[0.1, -0.1], maxfev=10000)
fitX = np.linspace(1, 10000, 1000)
fitY1 = objFuncP(fitX, popt1[0], popt1[1])

popt2, pcov2 = scipy.optimize.curve_fit(objFuncP, hist2centers, hist2[0], p0=[0.1, -0.1], maxfev=10000)
popt3, pcov3 = scipy.optimize.curve_fit(objFuncP, hist3centers, hist3[0], p0=[0.1, -0.1], maxfev=10000)
popt4, pcov4 = scipy.optimize.curve_fit(objFuncP, hist4centers, hist4[0], p0=[0.1, -0.1], maxfev=10000)

fitY2 = objFuncP(fitX, popt2[0], popt2[1])
fitY3 = objFuncP(fitX, popt3[0], popt3[1])
fitY4 = objFuncP(fitX, popt4[0], popt4[1])

# Segment Length in Log-Log

fig, ax = plt.subplots(1, 2, figsize=(20,6))
ax[0].scatter(hist1centers, hist1[0])
ax[0].set_title("Distribution of Segment Length", fontsize=20)
ax[0].set_ylabel("Normalized Density", fontsize=15)
ax[0].set_xlabel("Segment Length", fontsize=15)
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xlim(1, 10000)
ax[0].set_ylim(0.00001, 0.1)
ax[0].plot(fitX, fitY1, 'r')
ax[1].scatter(hist2centers, hist2[0])
ax[1].scatter(hist3centers, hist3[0])
ax[1].scatter(hist4centers, hist4[0])
ax[1].plot(fitX, fitY2)
ax[1].plot(fitX, fitY3)
ax[1].plot(fitX, fitY4)
ax[1].set_title("Distribution of Segment Length by Type", fontsize=20)
ax[1].set_ylabel("Normalized Density", fontsize=15)
ax[1].set_xlabel("Segment Length", fontsize=15)
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].legend(['Sensory', 'Inter', 'Motor'], fontsize=15)
ax[1].set_xlim(1, 10000)
ax[1].set_ylim(0.00001, 0.1)
plt.tight_layout()
plt.show()



# Segment Length in Log-Log by Type

fig, ax = plt.subplots(1, 3, figsize=(24,6))
ax[0].scatter(hist2centers, hist2[0])
ax[0].plot(fitX, fitY2, 'r')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xlim(1, 10000)
ax[0].set_ylim(0.00001, 0.1)
ax[0].set_title("Sensory Neuron Segment Length", fontsize=20)
ax[0].set_ylabel("Normalized Density", fontsize=15)
ax[0].set_xlabel("Segment Length", fontsize=15)

ax[1].scatter(hist3centers, hist3[0])
ax[1].plot(fitX, fitY3, 'r')
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlim(1, 10000)
ax[1].set_ylim(0.00001, 0.1)
ax[1].set_title("Interneuron Segment Length", fontsize=20)
ax[1].set_ylabel("Normalized Density", fontsize=15)
ax[1].set_xlabel("Segment Length", fontsize=15)

ax[2].scatter(hist4centers, hist4[0])
ax[2].plot(fitX, fitY4, 'r')
ax[2].set_yscale('log')
ax[2].set_xscale('log')
ax[2].set_xlim(1, 10000)
ax[2].set_ylim(0.00001, 0.1)
ax[2].set_title("Motor Neuron Segment Length", fontsize=20)
ax[2].set_ylabel("Normalized Density", fontsize=15)
ax[2].set_xlabel("Segment Length", fontsize=15)
plt.tight_layout()
plt.show()


# Average Segment Length Histogram

fig, ax = plt.subplots(1, 2, figsize=(20,6))
hist9 = ax[0].hist(length_average,
          bins=int((np.max(np.array(length_average)) - np.min(np.array(length_average)))/10),
          density=True)
ax[0].set_title("Distribution of Average Segment Length", fontsize=20)
ax[0].set_ylabel("Normalized Density", fontsize=15)
ax[0].set_xlabel("Segment Length", fontsize=15)

hist5 = ax[1].hist(np.array(length_average)[sensory], 
                 bins=int((np.max(np.array(length_average)[sensory]) - np.min(np.array(length_average)[sensory]))/10),
                 density=True,
                 alpha=0.5)
hist6 = ax[1].hist(np.array(length_average)[inter], 
                 bins=int((np.max(np.array(length_average)[inter]) - np.min(np.array(length_average)[inter]))/10),
                 density=True,
                 alpha=0.5)
hist7 = ax[1].hist(np.array(length_average)[motor],
                 bins=int((np.max(np.array(length_average)[motor]) - np.min(np.array(length_average)[motor]))/10),
                 density=True, 
                 alpha=0.5)
ax[1].legend(['Sensory', 'Inter', 'Motor'], fontsize=15)
ax[1].set_title("Distribution of Average Segment Length", fontsize=20)
ax[1].set_ylabel("Normalized Density", fontsize=15)
ax[1].set_xlabel("Average Segment Length", fontsize=15)
plt.tight_layout()
plt.show()

hist5centers = 0.5*(hist5[1][1:] + hist5[1][:-1])
hist6centers = 0.5*(hist6[1][1:] + hist6[1][:-1])
hist7centers = 0.5*(hist7[1][1:] + hist7[1][:-1])
hist9centers = 0.5*(hist9[1][1:] + hist9[1][:-1])


# Average Segment Length in Log-Log

fig, ax = plt.subplots(1, 2, figsize=(20,6))
ax[0].scatter(hist9centers, hist9[0])
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xlim(1, 10000)
ax[0].set_ylim(0.0001, 0.1)
ax[0].set_title("Distribution of Average Segment Length", fontsize=20)
ax[0].set_ylabel("Normalized Density", fontsize=15)
ax[0].set_xlabel("Segment Length", fontsize=15)

ax[1].scatter(hist5centers, hist5[0])
ax[1].scatter(hist6centers, hist6[0])
ax[1].scatter(hist7centers, hist7[0])
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlim(1, 10000)
ax[1].set_ylim(0.0001, 0.1)
ax[1].set_title("Distribution of Average Segment Length by Type", fontsize=20)
ax[1].set_ylabel("Normalized Density", fontsize=15)
ax[1].set_xlabel("Segment Length", fontsize=15)
ax[1].legend(['Sensory', 'Inter', 'Motor'], fontsize=15)
plt.tight_layout()
plt.show()


# Average Segment Length in Log-Log by type

fig, ax = plt.subplots(1, 3, figsize=(24,6))
ax[0].scatter(hist5centers, hist5[0])
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xlim(1, 10000)
ax[0].set_ylim(0.00001, 1)
ax[0].set_title("Average Sensory Neuron Segment Length", fontsize=20)
ax[0].set_ylabel("Normalized Density", fontsize=15)
ax[0].set_xlabel("Average Segment Length", fontsize=15)

ax[1].scatter(hist6centers, hist6[0])
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlim(1, 10000)
ax[1].set_ylim(0.00001, 1)
ax[1].set_title("Average Interneuron Segment Length", fontsize=20)
ax[1].set_ylabel("Normalized Density", fontsize=15)
ax[1].set_xlabel("Average Segment Length", fontsize=15)

ax[2].scatter(hist7centers, hist7[0])
ax[2].set_yscale('log')
ax[2].set_xscale('log')
ax[2].set_xlim(1, 10000)
ax[2].set_ylim(0.00001, 1)
ax[2].set_title("Average Motor Neuron Segment Length", fontsize=20)
ax[2].set_ylabel("Normalized Density", fontsize=15)
ax[2].set_xlabel("Average Segment Length", fontsize=15)
plt.tight_layout()
plt.show()


def objFuncL(xdata, a):
    y = a*xdata
    
    return y

def objFuncGL(xdata, a, b):
    y = a*xdata + b
    
    return y


# BranchNum vs Total Segment Length vs Average Segment Length by Type

poptL = []

fig, ax = plt.subplots(4, 3, figsize=(18,24))
ax[0][0].scatter(np.array(length_total)[sensory], np.array(branchNum)[sensory])
ax[0][0].set_title("Sensory Neuron", fontsize=20)
ax[0][0].set_xlabel("Total Segment Length", fontsize=15)
ax[0][0].set_ylabel("Number of Branches", fontsize=15)
ax[0][0].set_xlim(-50, 1000)
ax[0][0].set_ylim(-1, 8)

ax[0][1].scatter(np.array(length_total)[inter], np.array(branchNum)[inter])
ax[0][1].set_title("Interneuron", fontsize=20)
ax[0][1].set_xlabel("Total Segment Length", fontsize=15)
ax[0][1].set_xlim(-50, 1000)
ax[0][1].set_ylim(-1, 8)

ax[0][2].scatter(np.array(length_total)[motor], np.array(branchNum)[motor])
ax[0][2].set_title("Motor Neuron", fontsize=20)
ax[0][2].set_xlabel("Total Segment Length", fontsize=15)
ax[0][2].set_xlim(-50, 1000)
ax[0][2].set_ylim(-1, 8)

ax[1][0].scatter(np.array(length_average)[sensory], np.array(branchNum)[sensory])
ax[1][0].set_xlabel("Average Segment Length", fontsize=15)
ax[1][0].set_ylabel("Number of Branches", fontsize=15)
ax[1][0].set_xlim(-50, 1000)
ax[1][0].set_ylim(-1, 8)

ax[1][1].scatter(np.array(length_average)[inter], np.array(branchNum)[inter])
ax[1][1].set_xlabel("Average Segment Length", fontsize=15)
ax[1][1].set_xlim(-50, 1000)
ax[1][1].set_ylim(-1, 8)

ax[1][2].scatter(np.array(length_average)[motor], np.array(branchNum)[motor])
ax[1][2].set_xlabel("Average Segment Length", fontsize=15)
ax[1][2].set_xlim(-50, 1000)
ax[1][2].set_ylim(-1, 8)

for i in range(len(np.unique(np.array(branchNum)[sensory]))):
    scttrInd = np.where(np.array(branchNum)[sensory] == np.unique(np.array(branchNum)[sensory])[i])[0]
    ax[2][0].scatter(np.array(length_average)[sensory][scttrInd], np.array(length_total)[sensory][scttrInd])
    fitX = np.linspace(0, 1000, 1000)
ax[2][0].set_xlabel("Average Segment Length", fontsize=15)
ax[2][0].set_ylabel("Total Segment Length", fontsize=15)
ax[2][0].legend(np.unique(np.array(branchNum)[sensory])[:-1], fontsize=15)
for i in range(len(np.unique(np.array(branchNum)[sensory]))):
    scttrInd = np.where(np.array(branchNum)[sensory] == np.unique(np.array(branchNum)[sensory])[i])[0]
    if np.unique(np.array(branchNum)[sensory])[i] == 0:
        fitY = objFuncL(fitX, 1)
        ax[2][0].plot(fitX, fitY)
    elif np.unique(np.array(branchNum)[sensory])[i] == 1 or np.unique(np.array(branchNum)[sensory])[i] == 2:
        popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                np.array(length_average)[sensory][scttrInd], 
                                                np.array(length_total)[sensory][scttrInd],
                                                p0=[1.],
                                                maxfev=10000)
        fitY = objFuncL(fitX, popt[0])
        ax[2][0].plot(fitX, fitY)
        poptL.append(popt[0])
ax[2][0].set_xlim(-50, 1000)
ax[2][0].set_ylim(0, 1000)

for i in range(len(np.unique(np.array(branchNum)[inter]))):
    scttrInd = np.where(np.array(branchNum)[inter] == np.unique(np.array(branchNum)[inter])[i])[0]
    ax[2][1].scatter(np.array(length_average)[inter][scttrInd], np.array(length_total)[inter][scttrInd])
ax[2][1].set_xlabel("Average Segment Length", fontsize=15)
ax[2][1].legend(np.unique(np.array(branchNum)[inter]), fontsize=15)
for i in range(len(np.unique(np.array(branchNum)[inter]))):
    scttrInd = np.where(np.array(branchNum)[inter] == np.unique(np.array(branchNum)[inter])[i])[0]
    if np.unique(np.array(branchNum)[inter])[i] == 0:
        fitY = objFuncL(fitX, 1)
        ax[2][1].plot(fitX, fitY)
    elif np.unique(np.array(branchNum)[inter])[i] == 1 or np.unique(np.array(branchNum)[inter])[i] == 2:
        popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                np.array(length_average)[inter][scttrInd], 
                                                np.array(length_total)[inter][scttrInd],
                                                p0=[1.],
                                                maxfev=10000)
        fitY = objFuncL(fitX, popt[0])
        ax[2][1].plot(fitX, fitY)
        poptL.append(popt[0])
ax[2][1].set_xlim(-50, 1000)
ax[2][1].set_ylim(0, 1000)

for i in range(len(np.unique(np.array(branchNum)[motor]))):
    scttrInd = np.where(np.array(branchNum)[motor] == np.unique(np.array(branchNum)[motor])[i])[0]
    ax[2][2].scatter(np.array(length_average)[motor][scttrInd], np.array(length_total)[motor][scttrInd])
ax[2][2].set_xlabel("Average Segment Length", fontsize=15)
ax[2][2].legend(np.unique(np.array(branchNum)[motor]), fontsize=15)
for i in range(len(np.unique(np.array(branchNum)[motor]))):
    scttrInd = np.where(np.array(branchNum)[motor] == np.unique(np.array(branchNum)[motor])[i])[0]
    if np.unique(np.array(branchNum)[motor])[i] == 0:
        fitY = objFuncL(fitX, 1)
        ax[2][2].plot(fitX, fitY)
    elif np.unique(np.array(branchNum)[motor])[i] == 1 or np.unique(np.array(branchNum)[motor])[i] == 2:
        popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                np.array(length_average)[motor][scttrInd], 
                                                np.array(length_total)[motor][scttrInd],
                                                p0=[1.],
                                                maxfev=10000)
        fitY = objFuncL(fitX, popt[0])
        ax[2][2].plot(fitX, fitY)
        poptL.append(popt[0])
ax[2][2].set_xlim(-50, 1000)
ax[2][2].set_ylim(0, 1000)




length_branch_len = [len(arr) for arr in length_branch]
repeated_length_total = np.repeat(length_total, length_branch_len)

for i in range(len(np.unique(np.array(branchNum)[sensory]))):
    scttrInd = np.where(np.array(branchNum)[sensory] == np.unique(np.array(branchNum)[sensory])[i])[0]
    length_branch_len_sensory = [len(arr) for arr in np.array(length_branch)[sensory][scttrInd]]
    repeated_length_total_sensory = np.repeat(np.array(length_total)[sensory][scttrInd], 
                                              np.array(length_branch_len)[sensory][scttrInd])
    ax[3][0].scatter([item for sublist in np.array(length_branch)[sensory][scttrInd].tolist() for item in sublist], 
                     repeated_length_total_sensory)
ax[3][0].set_xlabel("Segment Length", fontsize=15)
ax[3][0].set_ylabel("Total Segment Length", fontsize=15)
ax[3][0].legend(np.unique(np.array(branchNum)[sensory])[:-1], fontsize=15)
for i in range(len(np.unique(np.array(branchNum)[sensory]))):
    scttrInd = np.where(np.array(branchNum)[sensory] == np.unique(np.array(branchNum)[sensory])[i])[0]
    length_branch_len_sensory = [len(arr) for arr in np.array(length_branch)[sensory][scttrInd]]
    repeated_length_total_sensory = np.repeat(np.array(length_total)[sensory][scttrInd], 
                                              np.array(length_branch_len)[sensory][scttrInd])
    if np.unique(np.array(branchNum)[sensory])[i] == 0:
        fitY = objFuncL(fitX, 1)
        ax[3][0].plot(fitX, fitY)
    elif np.unique(np.array(branchNum)[sensory])[i] == 1 or np.unique(np.array(branchNum)[sensory])[i] == 2:
        popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                              [item for sublist in np.array(length_branch)[sensory][scttrInd].tolist() for item in sublist], 
                                              repeated_length_total_sensory,
                                              p0=[1.],
                                              maxfev=10000)
        fitY = objFuncL(fitX, popt[0])
        ax[3][0].plot(fitX, fitY)
        poptL.append(popt[0])
ax[3][0].set_xlim(-50, 1000)
ax[3][0].set_ylim(0, 1000)

for i in range(len(np.unique(np.array(branchNum)[inter]))):
    scttrInd = np.where(np.array(branchNum)[inter] == np.unique(np.array(branchNum)[inter])[i])[0]
    length_branch_len_inter = [len(arr) for arr in np.array(length_branch)[inter][scttrInd]]
    repeated_length_total_inter = np.repeat(np.array(length_total)[inter][scttrInd], 
                                            np.array(length_branch_len)[inter][scttrInd])
    ax[3][1].scatter([item for sublist in np.array(length_branch)[inter][scttrInd].tolist() for item in sublist], 
                     repeated_length_total_inter)
ax[3][1].set_xlabel("Segment Length", fontsize=15)
ax[3][1].legend(np.unique(np.array(branchNum)[inter]), fontsize=15)
for i in range(len(np.unique(np.array(branchNum)[inter]))):
    scttrInd = np.where(np.array(branchNum)[inter] == np.unique(np.array(branchNum)[inter])[i])[0]
    length_branch_len_inter = [len(arr) for arr in np.array(length_branch)[inter][scttrInd]]
    repeated_length_total_inter = np.repeat(np.array(length_total)[inter][scttrInd], 
                                            np.array(length_branch_len)[inter][scttrInd])
    if np.unique(np.array(branchNum)[inter])[i] == 0:
        fitY = objFuncL(fitX, 1)
        ax[3][1].plot(fitX, fitY)
    elif np.unique(np.array(branchNum)[inter])[i] == 1 or np.unique(np.array(branchNum)[inter])[i] == 2:
        popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                              [item for sublist in np.array(length_branch)[inter][scttrInd].tolist() for item in sublist], 
                                              repeated_length_total_inter,
                                              p0=[1.],
                                              maxfev=10000)
        fitY = objFuncL(fitX, popt[0])
        ax[3][1].plot(fitX, fitY)
        poptL.append(popt[0])
ax[3][1].set_xlim(-50, 1000)
ax[3][1].set_ylim(0, 1000)

for i in range(len(np.unique(np.array(branchNum)[motor]))):
    scttrInd = np.where(np.array(branchNum)[motor] == np.unique(np.array(branchNum)[motor])[i])[0]
    length_branch_len_motor = [len(arr) for arr in np.array(length_branch)[motor][scttrInd]]
    repeated_length_total_motor = np.repeat(np.array(length_total)[motor][scttrInd], 
                                            np.array(length_branch_len)[motor][scttrInd])
    ax[3][2].scatter([item for sublist in np.array(length_branch)[motor][scttrInd].tolist() for item in sublist], 
                     repeated_length_total_motor)
ax[3][2].set_xlabel("Segment Length", fontsize=15)
ax[3][2].legend(np.unique(np.array(branchNum)[motor]), fontsize=15)
for i in range(len(np.unique(np.array(branchNum)[motor]))):
    scttrInd = np.where(np.array(branchNum)[motor] == np.unique(np.array(branchNum)[motor])[i])[0]
    length_branch_len_motor = [len(arr) for arr in np.array(length_branch)[motor][scttrInd]]
    repeated_length_total_motor = np.repeat(np.array(length_total)[motor][scttrInd], 
                                            np.array(length_branch_len)[motor][scttrInd])
    if np.unique(np.array(branchNum)[motor])[i] == 0:
        fitY = objFuncL(fitX, 1)
        ax[3][2].plot(fitX, fitY)
    elif np.unique(np.array(branchNum)[motor])[i] == 1 or np.unique(np.array(branchNum)[motor])[i] == 2:
        popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                              [item for sublist in np.array(length_branch)[motor][scttrInd].tolist() for item in sublist], 
                                              repeated_length_total_motor,
                                              p0=[1.],
                                              maxfev=10000)
        fitY = objFuncL(fitX, popt[0])
        ax[3][2].plot(fitX, fitY)
        poptL.append(popt[0])
ax[3][2].set_xlim(-50, 1000)
ax[3][2].set_ylim(0, 1000)

plt.tight_layout()
plt.show()



branchEndPDict = {'branch': branchNum, 'endP': endP_len}
branchEndPDF = pd.DataFrame(data=branchEndPDict)
fig = plt.figure(figsize=(8,6))
seaborn.swarmplot(x='branch', y='endP', data=branchEndPDF.loc[branchEndPDF['branch'] < 197])
plt.title("Distribution of Number of Endpoints\n for Given Branch Number", fontsize=20)
plt.xlabel("Branch Number", fontsize=15)
plt.ylabel("Number of Endpoints", fontsize=15)
#plt.xlim(-1, 10)
#plt.ylim(-1, 10)
plt.tight_layout()
plt.show()



fig = plt.figure(figsize=(8,6))
seaborn.kdeplot(np.delete(np.array(branchNum)[sensory], np.where(np.array(branchNum)[sensory] == 197)[0]), bw=.6, label="Sensory")
seaborn.kdeplot(np.array(branchNum)[inter], bw=.6, label="Inter")
seaborn.kdeplot(np.array(branchNum)[motor], bw=.6, label="Motor")
plt.xlim(-2, 8)
plt.title("Estimated Distribution of Branch Number by Type", fontsize=20)
plt.xlabel("Number of Branches", fontsize=15)
plt.ylabel("Estimated Probability Density", fontsize=15)
plt.legend(['Sensory', 'Inter', 'Motor'], fontsize=15)
plt.tight_layout()
plt.show()



fig = plt.figure(figsize=(8,6))
plt.scatter(np.array(morph_dist_len), np.array(rGy))
plt.yscale('log')
plt.xscale('log')
#plt.xlim(1, 10000)
#plt.ylim(0.005, 1000)
plt.title("Scaling Behavior of $R_{g}$ to $N$", fontsize=20)
plt.xlabel("Number of Points", fontsize=15)
plt.ylabel("Radius of Gyration", fontsize=15)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8,6))
plt.scatter(np.array(morph_dist_len_EP), np.array(rGyEP))
plt.yscale('log')
plt.xscale('log')
#plt.xlim(1, 10000)
#plt.ylim(0.005, 1000)
plt.title("Scaling Behavior of $R_{g}$ to $N_{EP}$", fontsize=20)
plt.xlabel("Number of Nodes", fontsize=15)
plt.ylabel("Radius of Gyration", fontsize=15)
plt.tight_layout()
plt.show()


#reg_len_scale = np.average(np.divide(regMDistLen, morph_dist_len))
poptR, pcovR = scipy.optimize.curve_fit(objFuncGL, 
                                        np.log10(np.array(regMDistLen)*sSize), 
                                        np.log10(np.sqrt(np.square(np.array(rGyReg))*1/sSize)), 
                                        p0=[1., 0.], 
                                        maxfev=100000)
fitYregR = objFuncPpow(np.array(regMDistLen)*sSize, poptR[0], poptR[1])

fig = plt.figure(figsize=(8,6))
plt.scatter(np.array(regMDistLen)*sSize, np.sqrt(np.square(np.array(rGyReg))*1/sSize))
plt.plot(np.array(regMDistLen)*sSize, fitYregR, color='tab:red')
plt.yscale('log')
plt.xscale('log')
plt.xlim(20, 10000)
plt.ylim(20, 6000)
plt.title(r"Scaling Behavior of Regularized $R_{g}$ to Regularized $N$", fontsize=20)
plt.xlabel(r"Number of Regularized Points ($a*N$)", fontsize=15)
plt.ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
plt.tight_layout()
plt.show()



poptR_sensory, pcovR_sensory = scipy.optimize.curve_fit(objFuncGL, 
                                        np.log10(np.array(regMDistLen)[sensory]*sSize), 
                                        np.log10(np.sqrt(np.square(np.array(rGyReg))[sensory]*1/sSize)), 
                                        p0=[1., 0.], 
                                        maxfev=100000)
fitYregR_sensory = objFuncPpow(np.array(regMDistLen)*sSize, poptR_sensory[0], poptR_sensory[1])

poptR_inter, pcovR_inter = scipy.optimize.curve_fit(objFuncGL, 
                                        np.log10(np.array(regMDistLen)[inter]*sSize), 
                                        np.log10(np.sqrt(np.square(np.array(rGyReg))[inter]*1/sSize)), 
                                        p0=[1., 0.], 
                                        maxfev=100000)
fitYregR_inter = objFuncPpow(np.array(regMDistLen)*sSize, poptR_inter[0], poptR_inter[1])

poptR_motor, pcovR_motor = scipy.optimize.curve_fit(objFuncGL, 
                                        np.log10(np.array(regMDistLen)[motor]*sSize), 
                                        np.log10(np.sqrt(np.square(np.array(rGyReg))[motor]*1/sSize)), 
                                        p0=[1., 0.], 
                                        maxfev=100000)
fitYregR_motor = objFuncPpow(np.array(regMDistLen)*sSize, poptR_motor[0], poptR_motor[1])


fig = plt.figure(figsize=(8,6))
plt.scatter(np.array(regMDistLen)[sensory]*sSize, np.sqrt(np.square(np.array(rGyReg))[sensory]*1/sSize))
plt.scatter(np.array(regMDistLen)[inter]*sSize, np.sqrt(np.square(np.array(rGyReg))[inter]*1/sSize))
plt.scatter(np.array(regMDistLen)[motor]*sSize, np.sqrt(np.square(np.array(rGyReg))[motor]*1/sSize))
plt.legend(["Sensory Neuron", "Interneuron", "Motor Neuron"], fontsize=15)
plt.plot(np.array(regMDistLen)*sSize, fitYregR_sensory, color='tab:blue')
plt.plot(np.array(regMDistLen)*sSize, fitYregR_inter, color='tab:orange')
plt.plot(np.array(regMDistLen)*sSize, fitYregR_motor, color='tab:green')
plt.yscale('log')
plt.xscale('log')
plt.xlim(20, 10000)
plt.ylim(20, 6000)
plt.title(r"Scaling Behavior of Regularized $R_{g}$ to Regularized $N$", fontsize=20)
plt.xlabel(r"Number of Regularized Points ($a*N$)", fontsize=15)
plt.ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
plt.tight_layout()
plt.show()



rGyRegSeg_avg = []
for i in range(len(nSize)):
    RStemp = np.where(np.array(regSegOrdN) == nSize[i])[0]
    rGyRegSeg_avg.append(np.average(np.array(rGyRegSeg)[RStemp]))

RS1 = np.where(np.array(regSegOrdN) > 7)[0]
RS2 = np.where((np.array(regSegOrdN) <= 8) & (np.array(regSegOrdN) >= 4))[0]
RS3 = np.where(np.array(regSegOrdN) < 5)[0]

poptRS1, pcovRS1 = scipy.optimize.curve_fit(objFuncGL, 
                                          np.log10(np.array(regSegOrdN)[RS1]*sSize), 
                                          np.log10(np.sqrt(np.square(np.array(rGyRegSeg)[RS1])*1/sSize)), 
                                          p0=[1., 0.], 
                                          maxfev=100000)
fitYregRS1 = objFuncPpow(np.unique(np.array(regSegOrdN)[RS1])*sSize, poptRS1[0], poptRS1[1])

fitYregRS12 = objFuncPpow(np.unique(np.array(regSegOrdN)[RS2])*sSize, poptRS1[0], poptRS1[1])

poptRS2, pcovRS2 = scipy.optimize.curve_fit(objFuncGL, 
                                          np.log10(np.array(regSegOrdN)[RS2]*sSize), 
                                          np.log10(np.sqrt(np.square(np.array(rGyRegSeg)[RS2])*1/sSize)), 
                                          p0=[1., 0.], 
                                          maxfev=100000)
fitYregRS2 = objFuncPpow(np.unique(np.array(regSegOrdN)[RS2])*sSize, poptRS2[0], poptRS2[1])

poptRS3, pcovRS3 = scipy.optimize.curve_fit(objFuncGL, 
                                          np.log10(np.array(regSegOrdN)[RS3]*sSize), 
                                          np.log10(np.sqrt(np.square(np.array(rGyRegSeg)[RS3])*1/sSize)), 
                                          p0=[1., 0.], 
                                          maxfev=100000)
fitYregRS3 = objFuncPpow(np.unique(np.array(regSegOrdN)[RS3])*sSize, poptRS3[0], poptRS3[1])

fitYregRS32 = objFuncPpow(np.unique(np.array(regSegOrdN)[RS2])*sSize, poptRS3[0], poptRS3[1])


fig, ax1 = plt.subplots(figsize=(12,8))
ax1.xaxis.label.set_fontsize(15)
ax1.xaxis.set_tick_params(which='major', length=7)
ax1.xaxis.set_tick_params(which='minor', length=5)
ax1.yaxis.label.set_fontsize(15)
ax1.yaxis.set_tick_params(which='major', length=7)
ax1.yaxis.set_tick_params(which='minor', length=5)
ax1.scatter(np.array(regMDistLen)*sSize, np.sqrt(np.square(np.array(rGyReg))*1/sSize), color='tab:blue')
ax1.plot(np.array(regMDistLen)*sSize, fitYregR, color='tab:red', lw=2)
ax1.scatter(np.array(regSegOrdN)*sSize, np.sqrt(np.square(np.array(rGyRegSeg))*1/sSize), color='tab:blue', facecolors='none')
ax1.scatter(np.array(nSize)*sSize, np.sqrt(np.square(np.array(rGyRegSeg_avg))*1/sSize), color='tab:orange')
ax1.plot(np.unique(np.array(regSegOrdN)[RS1])*sSize, fitYregRS1, color='tab:red', lw=2, linestyle='--')
ax1.plot(np.unique(np.array(regSegOrdN)[RS2])*sSize, fitYregRS2, color='tab:red', lw=2, linestyle='--')
ax1.plot(np.unique(np.array(regSegOrdN)[RS3])*sSize, fitYregRS3, color='tab:red', lw=2, linestyle='--')
ax1.vlines(0.08, 0.01, 11000, linestyles='dashed')
ax1.vlines(0.04, 0.01, 11000, linestyles='dashed')
ax1.set_yscale('log')
ax1.set_xscale('log')
#ax1.xlim(0.01, 10500)
ax1.set_ylim(0.03, 10000)

ax2 = plt.axes([0, 0, 1, 1])
ip1 = InsetPosition(ax1, [0.03, 0.57, 0.4, 0.4])
ax2.set_axes_locator(ip1)
mark_inset(ax1, ax2, loc1=3, loc2=4, fc="none", ec='0.5')

ax2.scatter(np.array(regSegOrdN)*sSize, np.sqrt(np.square(np.array(rGyRegSeg))*1/sSize), color='tab:blue', facecolors='none')
ax2.scatter(np.array(nSize)[1:11]*sSize, np.sqrt(np.square(np.array(rGyRegSeg_avg))[1:11]*1/sSize), color='tab:orange')
ax2.plot(np.unique(np.array(regSegOrdN)[RS1])[:4]*sSize, fitYregRS1[:4], color='tab:red', lw=2, linestyle='--')
ax2.plot(np.unique(np.array(regSegOrdN)[RS2])*sSize, fitYregRS2, color='tab:red', lw=2, linestyle='--')
ax2.plot(np.unique(np.array(regSegOrdN)[RS3])*sSize, fitYregRS3, color='tab:red', lw=2, linestyle='--')
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.vlines(0.08, 0.01, 1, linestyles='dashed')
ax2.vlines(0.04, 0.01, 1, linestyles='dashed')
ax2.set_xlim(0.022, 0.15)
ax2.set_ylim(0.08, 0.6)

ax3 = plt.axes([1, 1, 2, 2])
ip2 = InsetPosition(ax1, [0.57, 0.08, 0.4, 0.4])
ax3.set_axes_locator(ip2)
mark_inset(ax1, ax3, loc1=2, loc2=4, fc="none", ec='0.5')

ax3.plot(np.unique(np.array(regSegOrdN)[RS1])[:4]*sSize, fitYregRS1[:4], color='tab:red', lw=2, linestyle='-')
ax3.plot(np.unique(np.array(regSegOrdN)[RS2])*sSize, fitYregRS12, color='tab:red', lw=2, linestyle='--')
ax3.plot(np.unique(np.array(regSegOrdN)[RS2])*sSize, fitYregRS2, color='tab:green', lw=2, linestyle='-')
ax3.plot(np.unique(np.array(regSegOrdN)[RS3])*sSize, fitYregRS3, color='tab:blue', lw=2, linestyle='-')
ax3.plot(np.unique(np.array(regSegOrdN)[RS2])*sSize, fitYregRS32, color='tab:blue', lw=2, linestyle='--')
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.vlines(0.08, 0.01, 1, linestyles='dashed')
ax3.vlines(0.04, 0.01, 1, linestyles='dashed')
ax3.set_xlim(0.035, 0.087)
ax3.set_ylim(0.12, 0.28)

ax1.set_xlabel(r"Number of Regularized Points ($\lambda N$)", fontsize=15)
ax1.set_ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
#plt.tight_layout()
if SAVE:
    plt.savefig('./images/regSegRG_morphScale_' + str(RN) + '.png', dpi=300, bbox_inches='tight')
plt.show()





fig, ax1 = plt.subplots(figsize=(12,8))
ax1.xaxis.label.set_fontsize(15)
ax1.xaxis.set_tick_params(which='major', length=7)
ax1.xaxis.set_tick_params(which='minor', length=5)
ax1.yaxis.label.set_fontsize(15)
ax1.yaxis.set_tick_params(which='major', length=7)
ax1.yaxis.set_tick_params(which='minor', length=5)
ax1.scatter(np.array(regMDistLen)[sensory]*sSize, np.sqrt(np.square(np.array(rGyReg))[sensory]*1/sSize))
ax1.scatter(np.array(regMDistLen)[inter]*sSize, np.sqrt(np.square(np.array(rGyReg))[inter]*1/sSize))
ax1.scatter(np.array(regMDistLen)[motor]*sSize, np.sqrt(np.square(np.array(rGyReg))[motor]*1/sSize))
ax1.legend(["Sensory Neuron", "Interneuron", "Motor Neuron"], fontsize=15)
ax1.scatter(np.array(regSegOrdN)[sensory_choice]*sSize, np.sqrt(np.square(np.array(rGyRegSeg))[sensory_choice]*1/sSize), color='tab:blue', facecolors='none')
ax1.scatter(np.array(regSegOrdN)[inter_choice]*sSize, np.sqrt(np.square(np.array(rGyRegSeg))[inter_choice]*1/sSize), color='tab:orange', facecolors='none')
ax1.scatter(np.array(regSegOrdN)[motor_choice]*sSize, np.sqrt(np.square(np.array(rGyRegSeg))[motor_choice]*1/sSize), color='tab:green', facecolors='none')
ax1.vlines(0.08, 0.01, 11000, linestyles='dashed')
ax1.vlines(0.04, 0.01, 11000, linestyles='dashed')
ax1.set_yscale('log')
ax1.set_xscale('log')
#ax1.xlim(0.01, 10500)
ax1.set_ylim(0.03, 10000)
if SAVE:
    plt.savefig('./images/regSegRG_morphScale_type_' + str(RN) + '.png', dpi=300, bbox_inches='tight')
plt.show()





shift_N = 1
poptRS_sl = []
RS_x = []
for i in range(len(nSize) - shift_N):
    RS_s = np.where((np.array(regSegOrdN) <= nSize[i+shift_N]) & (np.array(regSegOrdN) >= nSize[i]))[0]
    
    RS_x.append(np.average(nSize[i:i+shift_N]))
    
    poptRS_s, pcovRS_s = scipy.optimize.curve_fit(objFuncGL, 
                                                  np.log10(np.array(regSegOrdN)[RS_s]*sSize), 
                                                  np.log10(np.sqrt(np.square(np.array(rGyRegSeg)[RS_s])*1/sSize)), 
                                                  p0=[1., 0.], 
                                                  maxfev=100000)
    poptRS_sl.append(poptRS_s[0])


fig = plt.figure(figsize=(8,6))
plt.scatter(RS_x, poptRS_sl)
#plt.plot(np.array(regMDistLen)*sSize, fitYregR, color='tab:red')
#plt.yscale('log')
plt.hlines(poptR[0], 0.1, 1000, linestyles='--', color='tab:red')
plt.hlines(poptRS1[0], 0.1, 1000, linestyles='--', color='tab:green')
plt.hlines(poptRS3[0], 0.1, 1000, linestyles='--', color='tab:orange')
#plt.yscale('log')
plt.xscale('log')
plt.xlim(0.5, 1000)
#plt.ylim(0.005, 1000)
#plt.title(r"Scaling Behavior of Regularized $R_{g}$ to Regularized $N$", fontsize=20)
plt.xlabel(r"Average Number of Regularized Points ($\lambda N_{avg}$)", fontsize=15)
plt.ylabel(r"Slope ($\nu$)", fontsize=15)
#plt.tight_layout()
if SAVE:
    plt.savefig('./images/regSegRG_slope_' + str(RN) + '.png', dpi=300, bbox_inches='tight')
plt.show()



#randIdx = np.sort(np.random.choice(np.arange(0, len(indRegMDistLen)), 10, p=indMorph_dist_p, replace=False))
#
#fig = plt.figure(figsize=(24, 16))
#ax = plt.axes(projection='3d')
#
##for i in range(1):
#for j in randIdx:
#    dInt = np.arange(0, indRegMDistLen[j]-nSize[0], dSize)
#    for k in range(len(dInt)-1):
#        listOfPoints = np.array(indRegMDist[j])[dInt[k]:dInt[k+nSize[0]]]
#        for f in range(len(listOfPoints)-1):
#            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
#            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2])
#
#plt.show()




cInfo = pd.read_excel(r'./CElegansNeuroML-SNAPSHOT_030213/CElegansNeuronTables.xls')
cOrigin = cInfo["Origin"].to_numpy()
cTarget = cInfo["Target"].to_numpy()
cType = cInfo["Type"].to_numpy()
cNum = cInfo["Number of Connections"].to_numpy()

cOriginCVal = list(Counter(cOrigin).values())
cOriginCKey = list(Counter(cOrigin).keys())
cTargetCVal = list(Counter(cTarget).values())
cTargetCKey = list(Counter(cTarget).keys())

cOriginNeuronCorr = []
cOriginNeuronCorrS = []
cOriginS = []
cOriginNeuronCorrI = []
cOriginI = []
cOriginNeuronCorrM = []
cOriginM = []

for i in range(len(cOriginCKey)):
    cOriginNeuronCorr.append(neuron_id.index(cOriginCKey[i]))
    if cOriginCKey[i] in np.array(neuron_id)[sensory].tolist():
        cOriginS.append(i)
        cOriginNeuronCorrS.append(neuron_id.index(cOriginCKey[i]))
    elif cOriginCKey[i] in np.array(neuron_id)[inter].tolist():
        cOriginI.append(i)
        cOriginNeuronCorrI.append(neuron_id.index(cOriginCKey[i]))
    elif cOriginCKey[i] in np.array(neuron_id)[motor].tolist():
        cOriginM.append(i)
        cOriginNeuronCorrM.append(neuron_id.index(cOriginCKey[i]))
    
cTargetNeuronCorr = []
cTargetNeuronCorrS = []
cTargetS = []
cTargetNeuronCorrI = []
cTargetI = []
cTargetNeuronCorrM = []
cTargetM = []

for i in range(len(cTargetCKey)):
    cTargetNeuronCorr.append(neuron_id.index(cTargetCKey[i]))
    if cOriginCKey[i] in np.array(neuron_id)[sensory].tolist():
        cTargetS.append(i)
        cTargetNeuronCorrS.append(neuron_id.index(cTargetCKey[i]))
    elif cOriginCKey[i] in np.array(neuron_id)[inter].tolist():
        cTargetI.append(i)
        cTargetNeuronCorrI.append(neuron_id.index(cTargetCKey[i]))
    elif cOriginCKey[i] in np.array(neuron_id)[motor].tolist():
        cTargetM.append(i)
        cTargetNeuronCorrM.append(neuron_id.index(cTargetCKey[i]))


#fig = plt.figure()
#ax = plt.gca()
#ax.scatter(cTargetCVal, np.array(branchNum)[cTargetNeuronCorr])
##ax.scatter(cTargetCVal, np.array(length_total)[cTargetNeuronCorr])
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.show()
#
#fig = plt.figure()
#ax = plt.gca()
#ax.scatter(np.array(cTargetCVal)[cTargetS], np.array(branchNum)[cTargetNeuronCorrS])
##ax.scatter(np.array(cTargetCVal)[cTargetNeuronCorrS], np.array(length_total)[sensory])
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.show()
#
#fig = plt.figure()
#ax = plt.gca()
#ax.scatter(np.array(cTargetCVal)[cTargetI], np.array(branchNum)[cTargetNeuronCorrI])
##ax.scatter(np.array(cTargetCVal)[cTargetNeuronCorrS], np.array(length_total)[sensory])
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.show()
#
#fig = plt.figure()
#ax = plt.gca()
#ax.scatter(np.array(cTargetCVal)[cTargetM], np.array(branchNum)[cTargetNeuronCorrM])
##ax.scatter(np.array(cTargetCVal)[cTargetNeuronCorrS], np.array(length_total)[sensory])
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.show()


def plotFromPoints(listOfPoints, showPoint=False):
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


def plotMorphAll(showPoint=False):
    fig = plt.figure(figsize=(24, 16))
    ax = plt.axes(projection='3d')
    ax.set_xlim(-300, 300)
    ax.set_ylim(-150, 150)
    ax.set_zlim(-300, 300)
    cmap = cm.get_cmap('viridis', len(morph_id))
    for f in range(len(morph_id)):
        tararr = np.array(morph_dist[f])
        somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
        for p in range(len(morph_parent[f])):
            if morph_parent[f][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_dist[f][morph_id[f].index(morph_parent[f][p])], morph_dist[f][p]))
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(f))
                if showPoint:
                    ax.scatter3D(morph_dist[f][p][0], morph_dist[f][p][1], morph_dist[f][p][2], color=cmap(f), marker='x')
        ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(f))
        


def plotMorphNeuron(idx, scale=False, cmass=False, showPoint=False):
    fig = plt.figure(figsize=(24, 16))
    ax = plt.axes(projection='3d')
    if scale:
        ax.set_xlim(-300, 300)
        ax.set_ylim(-150, 150)
        ax.set_zlim(-300, 300)
    cmap = cm.get_cmap('viridis', len(morph_id))
    
    if cmass:
        cMass = np.sum(morph_dist[idx], axis=0)/len(morph_dist[idx])
        ax.scatter3D(cMass[0], cMass[1], cMass[2])
    
    if type(idx) == list or type(idx) == np.ndarray:
        for i in idx:
            tararr = np.array(morph_dist[i])
            somaIdx = np.where(np.array(morph_parent[i]) < 0)[0]
            for p in range(len(morph_parent[i])):
                if morph_parent[i][p] < 0:
                    pass
                else:
                    morph_line = np.vstack((morph_dist[i][morph_id[i].index(morph_parent[i][p])], morph_dist[i][p]))
                    ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i))
                    if showPoint:
                        ax.scatter3D(morph_dist[i][p][0], morph_dist[i][p][1], morph_dist[i][p][2], color=cmap(i), marker='x')
            ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(i))
            
    else:
        tararr = np.array(morph_dist[idx])
        somaIdx = np.where(np.array(morph_parent[idx]) < 0)[0]
        for p in range(len(morph_parent[idx])):
            if morph_parent[idx][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_dist[idx][morph_id[idx].index(morph_parent[idx][p])], morph_dist[idx][p]))
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(idx))
                if showPoint:
                    ax.scatter3D(morph_dist[idx][p][0], morph_dist[idx][p][1], morph_dist[idx][p][2], color=cmap(idx), marker='x')
        ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(idx))
        
        
def plotMorphProjection(idx, project='z', scale=False):
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
            plt.xlim(-300, 450)
            plt.ylim(-200, 200)
    cmap = cm.get_cmap('viridis', len(morph_id))
    if type(idx) == list or type(idx) == np.ndarray:
        for i in idx:
            tararr = np.array(morph_dist[i])
            somaIdx = np.where(np.array(morph_parent[i]) < 0)[0]
            for p in range(len(morph_parent[i])):
                if morph_parent[i][p] < 0:
                    pass
                else:
                    morph_line = np.vstack((morph_dist[i][morph_id[i].index(morph_parent[i][p])], morph_dist[i][p]))
                    if project == 'z':
                        plt.plot(morph_line[:,0], morph_line[:,1], color=cmap(i))
                    elif project == 'y':
                        plt.plot(morph_line[:,0], morph_line[:,2], color=cmap(i))
                    elif project == 'x':
                        plt.plot(morph_line[:,1], morph_line[:,2], color=cmap(i))
            if project == 'z':
                plt.scatter(tararr[somaIdx,0], tararr[somaIdx,1], color=cmap(i))
            elif project == 'y':
                plt.scatter(tararr[somaIdx,0], tararr[somaIdx,2], color=cmap(i))
            elif project == 'x':
                plt.scatter(tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(i))
    else:
        tararr = np.array(morph_dist[idx])
        somaIdx = np.where(np.array(morph_parent[idx]) < 0)[0]
        for p in range(len(morph_parent[idx])):
            if morph_parent[idx][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_dist[idx][morph_id[idx].index(morph_parent[idx][p])], morph_dist[idx][p]))
                if project == 'z':
                    plt.plot(morph_line[:,0], morph_line[:,1], color=cmap(idx))
                elif project == 'y':
                    plt.plot(morph_line[:,0], morph_line[:,2], color=cmap(idx))
                elif project == 'x':
                    plt.plot(morph_line[:,1], morph_line[:,2], color=cmap(idx))
        if project == 'z':
            plt.scatter(tararr[somaIdx,0], tararr[somaIdx,1], color=cmap(idx))
        elif project == 'y':
            plt.scatter(tararr[somaIdx,0], tararr[somaIdx,2], color=cmap(idx))
        elif project == 'x':
            plt.scatter(tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(idx))
    plt.show()


def _layer_pos(nodeList):
    pos = {}
    maxdist = 0
    for i in range(len(nodeList)):
        if len(nodeList[i]) > maxdist:
            maxdist = len(nodeList[i])
    
    for i in range(len(nodeList)):
        dist = maxdist/len(nodeList[i])
        print(dist)
        for j in range(len(nodeList[i])):
            if not nodeList[i][j] in pos.keys():
                pos[nodeList[i][j]] = (maxdist - len(nodeList[i]) + dist*j, 0 - i/5)
        
    return pos
    

def plotConnectionNetwork(name, hier=1, prog='twopi'):
    from networkx.drawing.nx_pydot import graphviz_layout
    
    namec = copy.deepcopy(name)
    if type(namec) == list:
        for i in range(len(namec)):
            if type(namec[i]) == int:
                namec[i] = neuron_id[namec[i]]
            if sum(cOrigin == namec[i]) == 0:
                raise(Exception("Unknown neuron id"))
    else:
        if type(namec) == int:
            namec = neuron_id[namec]
        if sum(cOrigin == namec) == 0:
            raise(Exception("Unknown neuron id"))
    
    color = []
    branch = []
    
    cmap = cm.get_cmap('Set1')
    
    nodeList = _trackConnection(namec, hier)
    
    nodeListFlat = np.unique([item for sublist in nodeList for item in sublist])
    G = nx.DiGraph()
    
    G.add_nodes_from(nodeListFlat)
    
    for i in range(len(nodeListFlat)):
        if nodeListFlat[i] == namec:
            color.append(cmap(0))
        elif nodeListFlat[i] in np.array(neuron_id)[sensory]:
            color.append(cmap(1))
        elif nodeListFlat[i] in np.array(neuron_id)[inter]:
            color.append(cmap(2))
        elif nodeListFlat[i] in np.array(neuron_id)[motor]:
            color.append(cmap(3))
        elif nodeListFlat[i] in np.array(neuron_id)[polymodal]:
            color.append(cmap(4))
        elif nodeListFlat[i] in np.array(neuron_id)[other]:
            color.append(cmap(5))
    
    for h in range(hier):
        for n in range(len(nodeList[h])):
            for i in range(sum(cOrigin == nodeList[h][n])):
                cTarind = np.where(cOrigin == nodeList[h][n])[0][i]
                G.add_edges_from([(cOrigin[cTarind], cTarget[cTarind])])

#    pos = nx.kamada_kawai_layout(G)
    pos = graphviz_layout(G, prog=prog)
#    pos = _layer_pos(nodeList)

    fig = plt.figure(figsize=(22, 14))
    nx.draw(G, pos, node_color=color, with_labels=True, node_size=1000)
    target_p = mpatches.Patch(color=cmap(0), label='Target Neuron')
    target_s = mpatches.Patch(color=cmap(1), label='Sensory Neuron')
    target_i = mpatches.Patch(color=cmap(2), label='Interneuron')
    target_m = mpatches.Patch(color=cmap(3), label='Motor Neuron')
    target_y = mpatches.Patch(color=cmap(4), label='Polymodal Neuron')
    target_o = mpatches.Patch(color=cmap(5), label='Other Neuron')
    plt.legend(handles=[target_p, target_s, target_i, target_m, target_y, target_o], fontsize=15)
    
    return G, nodeList


def plotMorphBranch(name, hier=1):
    namec = copy.deepcopy(name)
    if type(namec) == list:
        for i in range(len(namec)):
            if type(namec[i]) == int:
                namec[i] = neuron_id[namec[i]]
            if sum(cOrigin == namec[i]) == 0:
                raise(Exception("Unknown neuron id"))
    else:
        if type(namec) == int:
            namec = neuron_id[namec]
        if sum(cOrigin == namec) == 0:
            raise(Exception("Unknown neuron id"))
    
    color = []
    branch = []
    
    
    nodeList = _trackConnection(namec, hier)
    
    nodeListFlat = np.unique([item for sublist in nodeList for item in sublist])
    
    
    fig = plt.figure(figsize=(24, 16))
    ax = plt.axes(projection='3d')
    ax.set_xlim(-300, 300)
    ax.set_ylim(-150, 150)
    ax.set_zlim(-300, 300)
    cmap = cm.get_cmap('viridis', len(morph_id))
    
    for idx in range(len(nodeListFlat)):
        i = neuron_id.index(nodeListFlat[idx])
        tararr = np.array(morph_dist[i])
        somaIdx = np.where(np.array(morph_parent[i]) < 0)[0]
        for p in range(len(morph_parent[i])):
            if morph_parent[i][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_dist[i][morph_id[i].index(morph_parent[i][p])], morph_dist[i][p]))
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i))
        ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(i))
    
    plt.show()
    
    return nodeList


def plotScatterBranch(name, hier=1):
    namec = copy.deepcopy(name)
    if type(namec) == list:
        for i in range(len(namec)):
            if type(namec[i]) == int:
                namec[i] = neuron_id[namec[i]]
            if sum(cOrigin == namec[i]) == 0:
                raise(Exception("Unknown neuron id"))
    else:
        if type(namec) == int:
            namec = neuron_id[namec]
        if sum(cOrigin == namec) == 0:
            raise(Exception("Unknown neuron id"))
    
        nodeList = _trackConnection(namec, hier)
        nodeListFlat = np.unique([item for sublist in nodeList for item in sublist])
        selIdx = np.searchsorted(neuron_id,nodeListFlat)
        _length_branch_flat = [item for sublist in list(np.array(length_branch)[selIdx]) for item in sublist]
        
        fig = plt.figure(figsize=(12, 8))
        hist1 = plt.hist(_length_branch_flat, 
                         bins=int((np.max(_length_branch_flat) - np.min(_length_branch_flat))/10),
                         density=True)
        plt.title("Histogram of Segment Length", fontsize=20)
        plt.ylabel("Normalized Density", fontsize=15)
        plt.xlabel("Segment Length", fontsize=15)
        plt.tight_layout()
        plt.show()
        
        hist1centers = 0.5*(hist1[1][1:] + hist1[1][:-1])
    
        def objFuncPLS(p, *args):
            y = p[0]*np.power(args[0], p[1])
            
            return np.linalg.norm(y - args[1])
        
        res = scipy.optimize.differential_evolution(objFuncPLS, 
                                                    [(-10, 10), (-10, 10)], 
                                                    args=(hist1centers, hist1[0]))
        fitX = np.linspace(1, 10000, 1000)
        fitY1 = res.x[0]*np.power(fitX, res.x[1])
        
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(hist1centers, hist1[0])
        plt.title("Log-Log Plot of Segment Length", fontsize=20)
        plt.ylabel("Normalized Density", fontsize=15)
        plt.xlabel("Segment Length", fontsize=15)
        plt.yscale('log')
        plt.xscale('log')
    #    plt.xlim(1, 10000)
    #    plt.ylim(0.00001, 0.1)
        plt.plot(fitX, fitY1, 'r')
        plt.tight_layout()
        plt.show()
    

def _trackConnection(name, hier=1):
    if type(name) == list:
        nodeList = []
        for l in range(hier+1):
            nodeList.append([])
            
        for i in range(len(name)):
            if type(name[i]) == int:
                name[i] = neuron_id[name[i]]
            if sum(cOrigin == name[i]) == 0:
                raise(Exception("Unknown neuron id: ", name[i]))
            
            nodeList[0].append(name[i])
            
            for h in range(hier):
                nodeListTemp = []
                for n in range(len(nodeList[h])):
                    nodeListTemp.append(cTarget[np.where(cOrigin == nodeList[h][n])[0]])
                nodeListTemp = [item for sublist in nodeListTemp for item in sublist]
                nodeList[h+1].append(np.unique(nodeListTemp).tolist())
        
        for i in range(hier):
            nodeList[i+1] = np.unique([item for sublist in nodeList[i+1] for item in sublist]).tolist()
        
    else:
        if type(name) == int:
            name = neuron_id[name]
        if sum(cOrigin == name) == 0:
            raise(Exception("Unknown neuron id"))
    
        nodeList = []
        
        nodeList.append([name])
        for h in range(hier):
            nodeListTemp = []
            for n in range(len(nodeList[h])):
                nodeListTemp.append(cTarget[np.where(cOrigin == nodeList[h][n])[0]])
            nodeListTemp = [item for sublist in nodeListTemp for item in sublist]
            nodeList.append(np.unique(nodeListTemp).tolist())
    
    return nodeList

