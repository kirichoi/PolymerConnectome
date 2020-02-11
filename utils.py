# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:03:34 2020

@author: user
"""

import os
import numpy as np
import scipy.optimize


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

def sortDetailedPhysLoc(morph_dist):
    dphysLoc = np.empty(len(morph_dist))
    
    for i in range(len(morph_dist)):
        if (np.array(morph_dist[i])[:,1] <= -200).all():
            dphysLoc[i] = 0
        elif ((np.array(morph_dist[i])[:,1] <= -100).all() and (np.array(morph_dist[i])[:,1] > -200).all()):
            dphysLoc[i] = 1
        elif ((np.array(morph_dist[i])[:,1] >= 100).all() and (np.array(morph_dist[i])[:,1] < 200).all()):
            dphysLoc[i] = 3
        elif (np.array(morph_dist[i])[:,1] >= 200).all():
            dphysLoc[i] = 4
        else:
            dphysLoc[i] = 2
            
    return dphysLoc

def sortDetailedSomaPhysLoc(morph_dist, somaP):
    dsphysLoc = np.empty(len(morph_dist))
    
    for i in range(len(morph_dist)):
        if (np.array(morph_dist[i])[somaP[i]][1] <= -250):
            dsphysLoc[i] = 0
        elif ((np.array(morph_dist[i])[somaP[i]][1] <= -100) and (np.array(morph_dist[i])[somaP[i]][1] > -250)):
            dsphysLoc[i] = 1
        elif ((np.array(morph_dist[i])[somaP[i]][1] >= 100) and (np.array(morph_dist[i])[somaP[i]][1] < 250)):
            dsphysLoc[i] = 3
        elif (np.array(morph_dist[i])[somaP[i]][1] >= 250):
            dsphysLoc[i] = 4
        else:
            dsphysLoc[i] = 2
            
    return dsphysLoc

def exportOutput(Parameter, OutputData):
    
    outputdir = Parameter.outputdir
    RN = Parameter.RN
    
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
            
    outputtxt = open(os.path.join(outputdir, 'settings.txt'), 'w')
    outputtxt.writelines('------------------------- SETTINGS -----------------------\n')
    outputtxt.writelines('RUN COMPLETE. HERE ARE SOME METRIC YOU MIGHT BE INTERESTED\n')
    outputtxt.writelines('sSize: ' + str(Parameter.sSize) + '\n')
    outputtxt.writelines('nSize: ' + str(Parameter.nSize) + '\n')
    outputtxt.writelines('dSize: ' + str(Parameter.dSize) + '\n')
    outputtxt.writelines('SEED: ' + str(Parameter.SEED) + '\n')
    outputtxt.close()
    
    np.savetxt(outputdir + '/rGyRegSegs_' + str(RN) + '.csv', 
               OutputData.rGyRegSegs, delimiter=",")
    np.savetxt(outputdir + '/regSegOrdNs_' + str(RN) + '.csv', 
               OutputData.regSegOrdNs, delimiter=",")
    np.savetxt(outputdir + '/randTrks_' + str(RN) + '.csv',
               OutputData.randTrks, delimiter=",")
    np.savetxt(outputdir + '/rGyRegSegi_' + str(RN) + '.csv', 
               OutputData.rGyRegSegi, delimiter=",")
    np.savetxt(outputdir + '/regSegOrdNi_' + str(RN) + '.csv', 
               OutputData.regSegOrdNi, delimiter=",")
    np.savetxt(outputdir + '/randTrki_' + str(RN) + '.csv',
               OutputData.randTrki, delimiter=",")
    np.savetxt(outputdir + '/rGyRegSegm_' + str(RN) + '.csv', 
               OutputData.rGyRegSegm, delimiter=",")
    np.savetxt(outputdir + '/regSegOrdNm_' + str(RN) + '.csv',
               OutputData.regSegOrdNm, delimiter=",")
    np.savetxt(outputdir + '/randTrkm_' + str(RN) + '.csv',
               OutputData.randTrkm, delimiter=",")


def exportMorph(Parameter, time, MorphData, BranchData, LengthData):
    
    outputdir = Parameter.outputdir
    RN = Parameter.RN
    
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
            
    outputtxt = open(os.path.join(outputdir, 'settings.txt'), 'w')
    outputtxt.writelines('------------------------- SETTINGS -----------------------\n')
    outputtxt.writelines('RUN COMPLETE. HERE ARE SOME METRIC YOU MIGHT BE INTERESTED\n')
    outputtxt.writelines('sSize: ' + str(Parameter.sSize) + '\n')
    outputtxt.writelines('nSize: ' + str(Parameter.nSize) + '\n')
    outputtxt.writelines('dSize: ' + str(Parameter.dSize) + '\n')
    outputtxt.writelines('SEED: ' + str(Parameter.SEED) + '\n')
    outputtxt.writelines('Time: ' + str(time) + ' s\n')
    outputtxt.close()
    
#    with open(outputdir + '/neuron_id_' + str(RN) + '.txt', "w") as f:
#        f.write(repr(MorphData.neuron_id))
#    with open(outputdir + '/morph_id_' + str(RN) + '.txt', "w") as f:
#        f.write(repr(MorphData.morph_id))
#    with open(outputdir + '/morph_parent_' + str(RN) + '.txt', "w") as f:
#        f.write(repr(MorphData.morph_parent))
#    with open(outputdir + '/morph_dist_' + str(RN) + '.txt', "w") as f:
#        f.write(repr(MorphData.morph_dist))
#    with open(outputdir + '/endP_len_' + str(RN) + '.txt', "w") as f:
#        f.write(repr(MorphData.endP_len))    
    
#    with open(outputdir + '/branchNum_' + str(RN) + '.txt', "w") as f:
#        f.write(repr(BranchData.branchNum.tolist()))
#    with open(outputdir + '/branchP_' + str(RN) + '.txt', "w") as f:
#        f.write(repr(BranchData.branchP))
    with open(outputdir + '/branchTrk_' + str(RN) + '.txt', "w") as f:
        f.write(repr(BranchData.branchTrk))
    with open(outputdir + '/branch_dist_' + str(RN) + '.txt', "w") as f:
        f.write(repr(BranchData.branch_dist))
    with open(outputdir + '/indBranchTrk_' + str(RN) + '.txt', "w") as f:
        f.write(repr(BranchData.indBranchTrk))
    with open(outputdir + '/indMorph_dist_' + str(RN) + '.txt', "w") as f:
        f.write(repr(BranchData.indMorph_dist))
    with open(outputdir + '/indMorph_dist_p_us_' + str(RN) + '.txt', "w") as f:
        f.write(repr(BranchData.indMorph_dist_p_us.tolist()))
    
    with open(outputdir + '/length_branch_' + str(RN) + '.txt', "w") as f:
        f.write(repr(LengthData.length_branch))
    
    
def importData(Parameter):
    
    inputdir = Parameter.outputdir
    RN = Parameter.RN
    
    rGyRegSegs = np.genfromtxt(inputdir + '/rGyRegSegs_' + str(RN) + '.csv', 
                               delimiter=',')
    regSegOrdNs = np.genfromtxt(inputdir + '/regSegOrdNs_' + str(RN) + '.csv', 
                                dtype=int, delimiter=',')
    randTrks = np.genfromtxt(inputdir + '/randTrks_' + str(RN) + '.csv', 
                             dtype=int, delimiter=',')
    rGyRegSegi = np.genfromtxt(inputdir + '/rGyRegSegi_' + str(RN) + '.csv', 
                               delimiter=',')
    regSegOrdNi = np.genfromtxt(inputdir + '/regSegOrdNi_' + str(RN) + '.csv', 
                                dtype=int, delimiter=',')
    randTrki = np.genfromtxt(inputdir + '/randTrki_' + str(RN) + '.csv', 
                             dtype=int, delimiter=',')
    rGyRegSegm = np.genfromtxt(inputdir + '/rGyRegSegm_' + str(RN) + '.csv', 
                               delimiter=',')
    regSegOrdNm = np.genfromtxt(inputdir + '/regSegOrdNm_' + str(RN) + '.csv',
                                dtype=int, delimiter=',')
    randTrkm = np.genfromtxt(inputdir + '/randTrkm_' + str(RN) + '.csv', 
                             dtype=int, delimiter=',')
    
    return (rGyRegSegs, regSegOrdNs, randTrks, rGyRegSegi, regSegOrdNi, 
            randTrki, rGyRegSegm, regSegOrdNm, randTrkm)
    

def importMorph(Parameter):
    
    inputdir = Parameter.outputdir
    RN = Parameter.RN
    
#    with open(inputdir + '/neuron_id_' + str(RN) + '.txt', "r") as f:
#        neuron_id = eval(f.read())
#    with open(inputdir + '/morph_id_' + str(RN) + '.txt', "r") as f:
#        morph_id = eval(f.read())
#    with open(inputdir + '/morph_parent_' + str(RN) + '.txt', "r") as f:
#        morph_parent = eval(f.read())
#    with open(inputdir + '/morph_dist_' + str(RN) + '.txt', "r") as f:
#        morph_dist = eval(f.read())
#    with open(inputdir + '/endP_len_' + str(RN) + '.txt', "r") as f:
#        endP_len = eval(f.read())
#    
#    with open(inputdir + '/branchNum_' + str(RN) + '.txt', "r") as f:
#        branchNum = np.array(eval(f.read()))
#    with open(inputdir + '/branchP_' + str(RN) + '.txt', "r") as f:
#        branchP = eval(f.read())
    with open(inputdir + '/branchTrk_' + str(RN) + '.txt', "r") as f:
        branchTrk = eval(f.read())
    with open(inputdir + '/branch_dist_' + str(RN) + '.txt', "r") as f:
        branch_dist = eval(f.read())
    with open(inputdir + '/indBranchTrk_' + str(RN) + '.txt', "r") as f:
        indBranchTrk = eval(f.read())
    with open(inputdir + '/indMorph_dist_' + str(RN) + '.txt', "r") as f:
        indMorph_dist = eval(f.read())
    with open(inputdir + '/indMorph_dist_p_us_' + str(RN) + '.txt', "r") as f:
        indMorph_dist_p_us = np.array(eval(f.read()))
    
    with open(inputdir + '/length_branch_' + str(RN) + '.txt', "r") as f:
        length_branch = eval(f.read())
    
    return (branchTrk, branch_dist, indBranchTrk, indMorph_dist, indMorph_dist_p_us, length_branch)

    
    
def segmentMorph(Parameter, BranchData):
    regMDist = []
    
    for i in range(len(BranchData.branch_dist)):
        regMDist_temp1 = []
        for k in range(len(BranchData.branch_dist[i])):
            regMDist_temp2 = []
            for j in range(len(BranchData.branch_dist[i][k])-1):
                dist = np.linalg.norm(np.subtract(np.array(BranchData.branch_dist[i][k])[j+1][:3],
                                                  np.array(BranchData.branch_dist[i][k])[j][:3]))
                l1 = np.linspace(0,1,max(1, int(dist/Parameter.sSize)))
                nArr = np.array(BranchData.branch_dist[i][k])[j][:3]+(
                        np.subtract(np.array(BranchData.branch_dist[i][k])[j+1][:3],
                        np.array(BranchData.branch_dist[i][k])[j][:3]))*l1[:,None]
                regMDist_temp2.append(nArr.tolist())
            regMDist_temp1.append([item for sublist in regMDist_temp2 for item in sublist])
        regMDist_temp_flatten = [item for sublist in regMDist_temp1 for item in sublist]
        _, Uidx = np.unique(np.array(regMDist_temp_flatten), return_index=True, axis=0)
        uniqueUSorted = np.array(regMDist_temp_flatten)[np.sort(Uidx)].tolist()
        regMDist.append(uniqueUSorted)
    
    regMDistLen = np.array([len(arr) for arr in regMDist])
    
    return regMDist, regMDistLen
   
def indSegmentMorph(Parameter, BranchData):
    indRegMDist = []
    
    for i in range(len(BranchData.indMorph_dist_flat)):
        indRegMDist_temp = []
        for j in range(len(BranchData.indMorph_dist_flat[i])-1):
            dist = np.linalg.norm(np.subtract(np.array(BranchData.indMorph_dist_flat[i])[j+1][:3],
                                              np.array(BranchData.indMorph_dist_flat[i])[j][:3]))
            l1 = np.linspace(0,1,max(1, int(dist/Parameter.sSize)))
            nArr = np.array(BranchData.indMorph_dist_flat[i])[j][:3]+(
                    np.subtract(np.array(BranchData.indMorph_dist_flat[i])[j+1][:3],
                    np.array(BranchData.indMorph_dist_flat[i])[j][:3]))*l1[:,None]
            indRegMDist_temp.append(nArr.tolist())
        indRegMDist_temp_flatten = [item for sublist in indRegMDist_temp for item in sublist]
        _, Uidx = np.unique(np.array(indRegMDist_temp_flatten), return_index=True, axis=0)
        uniqueUSorted = np.array(indRegMDist_temp_flatten)[np.sort(Uidx)].tolist()
        indRegMDist.append(uniqueUSorted)
    
    indRegMDistLen = np.array([len(arr) for arr in indRegMDist])
    
    return indRegMDist, indRegMDistLen


def radiusOfGyration(MorphData):
    cML = np.empty((len(MorphData.morph_dist), 3))
    rGy = np.empty(len(MorphData.morph_dist))
    for i in range(len(MorphData.morph_dist)):
        cML[i] = np.sum(np.array(MorphData.morph_dist[i]), axis=0)[:3]/len(np.array(MorphData.morph_dist[i]))
        rList = scipy.spatial.distance.cdist(np.array(MorphData.morph_dist[i])[:,:3], 
                                             np.array([cML[i]])).flatten()
        rGy[i] = np.sqrt(np.sum(np.square(rList))/len(rList))
    
    return (rGy, cML)

def endPointRadiusOfGyration(MorphData, BranchData):
    cMLEP = np.empty((len(MorphData.morph_dist), 3))
    rGyEP = np.empty(len(MorphData.morph_dist))
    for i in range(len(MorphData.morph_dist)):
        distInd = np.where(np.isin(np.unique(np.hstack([MorphData.endP[i],
                                                        MorphData.somaP[i], 
                                                        BranchData.branchP[i]])), 
    MorphData.morph_id[i]))[0]
        MorphData.morph_dist_len_EP[i] = len(distInd)
        cMLEP[i] = np.sum(np.array(MorphData.morph_dist[i])[distInd], 
                            axis=0)[:3]/len(np.array(MorphData.morph_dist[i])[distInd])
        rList_EP = scipy.spatial.distance.cdist(np.array(MorphData.morph_dist[i])[distInd,:3], 
                                                np.array([cMLEP[i]])).flatten()
        rGyEP[i] = np.sqrt(np.sum(np.square(rList_EP))/len(rList_EP))
    
    return (rGyEP, cMLEP)

def regularRadiusOfGyration(regMDist, regMDistLen):
    
    cMLReg = np.empty((len(regMDist), 3))
    rGyReg = np.empty(len(regMDist))
    for i in range(len(regMDist)):
        cMLReg[i] = np.sum(np.array(regMDist[i]), axis=0)/regMDistLen[i]
        rList_reg = scipy.spatial.distance.cdist(np.array(regMDist[i]),
                                                 np.array([cMLReg[i]])).flatten()
        rGyReg[i] = np.sqrt(np.sum(np.square(rList_reg))/regMDistLen[i])
    
    return (rGyReg, cMLReg)

def regularSegmentRadiusOfGyration(Parameter, BranchData, indRegMDist, indRegMDistLen, numSample=10000, stochastic=True, p=None):

    nSize = np.array(Parameter.nSize)+1
    cMLRegSeg = np.empty((len(nSize)*numSample, 3))
    rGyRegSeg = np.empty(len(nSize)*numSample)
    regSegOrdN = np.empty(len(nSize)*numSample)
    randTrk = np.empty((len(nSize)*numSample, 3), dtype=int)
    idxTrk = 0
    
    if p == None:
        indMorph_dist_p = BranchData.indMorph_dist_p_us/np.sum(BranchData.indMorph_dist_p_us)
    else:
        indMorph_dist_p = BranchData.indMorph_dist_p_us[p]/np.sum(BranchData.indMorph_dist_p_us[p])
#    indMorph_dist_p = np.ones(len(indRegMDist))/len(indRegMDist)
    
    if stochastic:
        for i in range(len(nSize)):
            for j in range(numSample):
                randIdx1 = np.random.choice(np.arange(0, len(indRegMDist)), 
                                            1, 
                                            p=indMorph_dist_p)[0]
                while len(indRegMDist[randIdx1]) <= nSize[i]:
                    randIdx1 = np.random.choice(np.arange(0, len(indRegMDist)),
                                                1, 
                                                p=indMorph_dist_p)[0]
                idxTrk += 1
                randIdx2 = np.random.choice(np.arange(0, len(indRegMDist[randIdx1])-nSize[i]), 1)[0]
                
                randTrk[i*numSample+j] = (randIdx1, randIdx2, randIdx2+nSize[i])
                
                regSegOrdN[i*numSample+j] = nSize[i]-1
                cMLRegSeg[i*numSample+j] = np.sum(np.array(indRegMDist[randIdx1])[randIdx2:randIdx2+nSize[i]], axis=0)/nSize[i]
                rList_reg_seg = scipy.spatial.distance.cdist(np.array(indRegMDist[randIdx1])[randIdx2:randIdx2+nSize[i]], 
                                                             np.array([cMLRegSeg[i*numSample+j]])).flatten()
                rGyRegSeg[i*numSample+j] = np.sqrt(np.sum(np.square(rList_reg_seg))/nSize[i])
    else:
        for i in range(len(nSize)):
            randIdx = np.sort(np.random.choice(np.arange(0, len(indRegMDistLen)), 
                                               100, 
                                               p=indMorph_dist_p, 
                                               replace=False))
            for j in randIdx:
                dInt = np.arange(0, indRegMDistLen[j]-nSize[i], Parameter.dSize)
                for k in range(len(dInt)-1):
                    regSegOrdN[i*numSample + j] = nSize[i]-1
                    cMLRegSeg[i*numSample+j] = np.sum(np.array(indRegMDist[i])[dInt[k]:dInt[k]+nSize[i]], axis=0)/nSize[i]
                    rList_reg_seg = scipy.spatial.distance.cdist(np.array(indRegMDist[i])[dInt[k]:dInt[k]+nSize[i]], 
                                                                 np.array([cMLRegSeg[i*numSample+j]])).flatten()
                    rGyRegSeg[i*numSample+j] = np.sqrt(np.sum(np.square(rList_reg_seg))/nSize[i])
    
    return (rGyRegSeg, cMLRegSeg, regSegOrdN, randTrk)
    


def circle_points(r, n):
    t = np.linspace(0, 2*np.pi, n+1)
    x = r * np.cos(t)
    y = r * np.sin(t)    
    
    return x, y








