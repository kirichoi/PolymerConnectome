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
import utils

class Parameter:

    PATH = r'./CElegansNeuroML-SNAPSHOT_030213/CElegans/generatedNeuroML2'
    
    RUN = False
    SAVE = False
    PLOT = True 
    numSample = 1
    RN = '5'
    
    sSize = 0.1
    nSize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 75, 100, 250]
    dSize = 10
    
    SEED = 1234
    
    outputdir = './output/RN_' + str(RN)

fp = [f for f in os.listdir(Parameter.PATH) if os.path.isfile(os.path.join(Parameter.PATH, f))]
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
fp = [os.path.join(Parameter.PATH, f) for f in fp]

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
        self.sensory = []
        self.inter = []
        self.motor = []
        self.polymodal = []
        self.other = []
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
        
    
    def plotConnectionNetwork(self, name, hier=1, prog='twopi', gapjunction=False):
        from networkx.drawing.nx_pydot import graphviz_layout
        
        namec = copy.deepcopy(name)
        if type(namec) == list:
            for i in range(len(namec)):
                if type(namec[i]) == int:
                    namec[i] = self.neuron_id[namec[i]]
                if sum(cOrigin == namec[i]) == 0:
                    raise(Exception("Unknown neuron id"))
        else:
            if type(namec) == int:
                namec = self.neuron_id[namec]
            if sum(cOrigin == namec) == 0:
                raise(Exception("Unknown neuron id"))
        
        color = []
        branch = []
        
        cmap = cm.get_cmap('Set1')
        
        nodeList = self._trackConnection(namec, hier)
        
        nodeListFlat = np.unique([item for sublist in nodeList for item in sublist])
        G = nx.DiGraph()
        
        G.add_nodes_from(nodeListFlat)
        
        for i in range(len(nodeListFlat)):
            if nodeListFlat[i] == namec:
                color.append(cmap(0))
            elif nodeListFlat[i] in np.array(self.neuron_id)[self.sensory]:
                color.append(cmap(1))
            elif nodeListFlat[i] in np.array(self.neuron_id)[self.inter]:
                color.append(cmap(2))
            elif nodeListFlat[i] in np.array(self.neuron_id)[self.motor]:
                color.append(cmap(3))
            elif nodeListFlat[i] in np.array(self.neuron_id)[self.polymodal]:
                color.append(cmap(4))
            elif nodeListFlat[i] in np.array(self.neuron_id)[self.other]:
                color.append(cmap(5))
        
        for h in range(hier):
            for n in range(len(nodeList[h])):
                for i in range(sum(cOrigin == nodeList[h][n])):
                    cTarind = np.where(cOrigin == nodeList[h][n])[0][i]
                    G.add_edges_from([(cOrigin[cTarind], cTarget[cTarind])])
                    if cType[cTarind] == 'GapJunction' and gapjunction:
                        G.add_edges_from([(cTarget[cTarind], cOrigin[cTarind])])
    
        if prog == 'spatial':
            pos = self._layer_pos(self.neuron_id)
        elif prog == 'kamada':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = graphviz_layout(G, prog=prog)
    
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
    
    
    def plotConnectedNeurons(self, name, hier=1):
        namec = copy.deepcopy(name)
        if type(namec) == list:
            for i in range(len(namec)):
                if type(namec[i]) == int:
                    namec[i] = self.neuron_id[namec[i]]
                if sum(cOrigin == namec[i]) == 0:
                    raise(Exception("Unknown neuron id"))
        else:
            if type(namec) == int:
                namec = self.neuron_id[namec]
            if sum(cOrigin == namec) == 0:
                raise(Exception("Unknown neuron id"))
        
        nodeList = self._trackConnection(namec, hier)
        
        nodeListFlat = np.unique([item for sublist in nodeList for item in sublist])
        
        
        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection='3d')
        ax.set_xlim(-300, 300)
        ax.set_ylim(-150, 150)
        ax.set_zlim(-300, 300)
        cmap = cm.get_cmap('viridis', len(self.morph_id))
        
        for idx in range(len(nodeListFlat)):
            i = self.neuron_id.index(nodeListFlat[idx])
            tararr = np.array(self.morph_dist[i])
            somaIdx = np.where(np.array(self.morph_parent[i]) < 0)[0]
            for p in range(len(self.morph_parent[i])):
                if self.morph_parent[i][p] < 0:
                    pass
                else:
                    morph_line = np.vstack((self.morph_dist[i][self.morph_id[i].index(self.morph_parent[i][p])], self.morph_dist[i][p]))
                    ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i))
            ax.scatter3D(tararr[somaIdx,0], tararr[somaIdx,1], tararr[somaIdx,2], color=cmap(i))
        
        plt.show()
        
        return nodeList
    
    
    def plotFullConnectionNetwork(self, prog='fdp', gapjunction=False, node_size=300, with_labels=True):
        from networkx.drawing.nx_pydot import graphviz_layout
        
        color = []
        edge_color = []
        branch = []
        
        cmap = cm.get_cmap('Set1')
        
        G = nx.DiGraph()
        
        G.add_nodes_from(self.neuron_id)
        
        for i in range(len(self.neuron_id)):
            if i in self.sensory:
                color.append(cmap(1))
            elif i in self.inter:
                color.append(cmap(2))
            elif i in self.motor:
                color.append(cmap(3))
            elif i in self.polymodal:
                color.append(cmap(4))
            elif i in self.other:
                color.append(cmap(5))
        
        for h in range(len(cOrigin)):
            G.add_edges_from([(cOrigin[h], cTarget[h])])
            if cType[h] == 'GapJunction' and gapjunction:
                edge_color.append('tab:red')
            else:
                edge_color.append('k')
    
        if prog == 'spatial':
            pos = self._spat_pos(self.neuron_id)
        elif prog == 'kamada':
            pos = nx.kamada_kawai_layout(G)
        elif prog == 'phys':
            sdphysLoc = utils.sortDetailedSomaPhysLoc(self.morph_dist, self.somaP)
            sec1 = np.array(self.neuron_id)[np.where(sdphysLoc == 0)]
            sec2 = np.array(self.neuron_id)[np.where(sdphysLoc == 1)]
            sec3 = np.array(self.neuron_id)[np.where(sdphysLoc == 2)]
            sec4 = np.array(self.neuron_id)[np.where(sdphysLoc == 3)]
            sec5 = np.array(self.neuron_id)[np.where(sdphysLoc == 4)]
            
            pos = {}
            
            for i in range(len(sec1)):
                pos[sec1[i]] = np.array([-2, len(sec1)/2. - i])
            for i in range(len(sec2)):
                pos[sec2[i]] = np.array([-1, len(sec2)/2. - i])
            for i in range(len(sec3)):
                pos[sec3[i]] = np.array([0, len(sec3)/2. - i])
            for i in range(len(sec4)):
                pos[sec4[i]] = np.array([1, len(sec4)/2. - i])
            for i in range(len(sec5)):
                pos[sec5[i]] = np.array([2, len(sec5)/2. - i])
        elif prog == 'cat':
            pos = {}
            
            for i in range(len(self.sensory)):
                pos[self.neuron_id[self.sensory[i]]] = np.array([-20, (len(self.sensory)+len(self.polymodal)+len(self.other)+9) - 2*i])
            for i in range(len(self.inter)):
                pos[self.neuron_id[self.inter[i]]] = np.array([0, len(self.inter) - 2*i])
            for i in range(len(self.motor)):
                pos[self.neuron_id[self.motor[i]]] = np.array([20, len(self.motor) - 2*i])
            for i in range(len(self.polymodal)):
                pos[self.neuron_id[self.polymodal[i]]] = np.array([-20, (len(self.sensory)+len(self.polymodal)+len(self.other)+9) - 2*(i+len(self.sensory)+3)])
            for i in range(len(self.other)):
                pos[self.neuron_id[self.other[i]]] = np.array([-20, (len(self.sensory)+len(self.polymodal)+len(self.other)+9) - 2*(i+len(self.sensory)+len(self.polymodal)+6)])
        elif prog == 'catcirc':
            x, y = utils.circle_points(10, 302)
            cc = 0
            
            pos = {}
            for i in range(len(self.sensory)):
                pos[self.neuron_id[self.sensory[i]]] = np.array([x[cc], y[cc]])
                cc += 1
            for i in range(len(self.inter)):
                pos[self.neuron_id[self.inter[i]]] = np.array([x[cc], y[cc]])
                cc += 1
            for i in range(len(self.motor)):
                pos[self.neuron_id[self.motor[i]]] = np.array([x[cc], y[cc]])
                cc += 1
            for i in range(len(self.polymodal)):
                pos[self.neuron_id[self.polymodal[i]]] = np.array([x[cc], y[cc]])
                cc += 1
            for i in range(len(self.other)):
                pos[self.neuron_id[self.other[i]]] = np.array([x[cc], y[cc]])
                cc += 1
        elif prog == 'physcat':
            sdphysLoc = utils.sortDetailedSomaPhysLoc(self.morph_dist, self.somaP)
            sec1 = np.array(self.neuron_id)[np.where(sdphysLoc == 0)]
            sec2 = np.array(self.neuron_id)[np.where(sdphysLoc == 1)]
            sec3 = np.array(self.neuron_id)[np.where(sdphysLoc == 2)]
            sec4 = np.array(self.neuron_id)[np.where(sdphysLoc == 3)]
            sec5 = np.array(self.neuron_id)[np.where(sdphysLoc == 4)]
            
            pos = {}
            
            for i in range(len(sec1)):
                pos[sec1[i]] = np.array([-2, len(sec1)/2. - i])
            for i in range(len(sec2)):
                pos[sec2[i]] = np.array([-1, len(sec2)/2. - i])
            for i in range(len(sec3)):
                pos[sec3[i]] = np.array([0, len(sec3)/2. - i])
            for i in range(len(sec4)):
                pos[sec4[i]] = np.array([1, len(sec4)/2. - i])
            for i in range(len(sec5)):
                pos[sec5[i]] = np.array([2, len(sec5)/2. - i])
                
            for i in range(len(self.sensory)):
                pos[self.neuron_id[self.sensory[i]]] = np.array([-2, (len(self.sensory)+len(self.polymodal)+6)/2. - i])
            for i in range(len(self.inter)):
                pos[self.neuron_id[self.inter[i]]] = np.array([-1, len(self.inter)/2. - i])
            for i in range(len(self.motor)):
                pos[self.neuron_id[self.motor[i]]] = np.array([2, len(self.motor)/2. - i])
            for i in range(len(self.polymodal)):
                pos[self.neuron_id[self.polymodal[i]]] = np.array([-2, (len(self.sensory)+len(self.polymodal)+6)/2. - (i+len(self.sensory)+3)])
            for i in range(len(self.other)):
                pos[self.neuron_id[self.other[i]]] = np.array([0, len(self.other)/2. - i])
        else:
            pos = graphviz_layout(G, prog=prog)
    
        fig = plt.figure(figsize=(30, 30))
        nx.draw(G, pos, node_color=color, with_labels=with_labels, node_size=node_size, edge_color=edge_color)
        target_s = mpatches.Patch(color=cmap(1), label='Sensory Neuron')
        target_i = mpatches.Patch(color=cmap(2), label='Interneuron')
        target_m = mpatches.Patch(color=cmap(3), label='Motor Neuron')
        target_y = mpatches.Patch(color=cmap(4), label='Polymodal Neuron')
        target_o = mpatches.Patch(color=cmap(5), label='Other Neuron')
        plt.legend(handles=[target_s, target_i, target_m, target_y, target_o], fontsize=15)
        
        return G
    
    
    def plotCombinedConnectionNetwork(self, with_labels=True):
        from matplotlib.patches import FancyArrowPatch, Circle
        
        color = []
        
        sNodes = 1
        iNodes = 1
        mNodes = 1
        pNodes = 1
        oNodes = 1
        siEdge = 0
        smEdge = 0
        spEdge = 0
        soEdge = 0
        isEdge = 0
        imEdge = 0
        ipEdge = 0
        ioEdge = 0
        msEdge = 0
        miEdge = 0
        mpEdge = 0
        moEdge = 0
        psEdge = 0
        piEdge = 0
        pmEdge = 0
        poEdge = 0
        osEdge = 0
        oiEdge = 0
        omEdge = 0
        opEdge = 0
        
        cmap = cm.get_cmap('Set1')
        labels = ['Sensory', 'Inter', 'Motor', 'Polymodal', 'Other']
        
        fig = plt.figure(figsize=(20, 20))
        ax = plt.gca()
        
        for i in range(5):
            color.append(cmap(i+1))
        
        for h in range(len(cOrigin)):
            cOidx = np.where(np.array(MorphData.neuron_id) == cOrigin[h])[0]
            cTidx = np.where(np.array(MorphData.neuron_id) == cTarget[h])[0]
                
            if cOidx in self.sensory:
                if cTidx in self.sensory:
                    sNodes += 10
                    if cType[h] == 'GapJunction':
                        sNodes += 1
                elif cTidx in self.inter:
                    siEdge += 1
                    if cType[h] == 'GapJunction':
                        isEdge += 1
                elif cTidx in self.motor:
                    smEdge += 1
                    if cType[h] == 'GapJunction':
                        msEdge += 1
                elif cTidx in self.polymodal:
                    spEdge += 1
                    if cType[h] == 'GapJunction':
                       psEdge += 1 
                elif cTidx in self.other:
                    soEdge += 1
                    if cType[h] == 'GapJunction':
                        osEdge += 1
            elif cOidx in self.inter:
                if cTidx in self.sensory:
                    isEdge += 1
                    if cType[h] == 'GapJunction':
                        siEdge += 1
                elif cTidx in self.inter:
                    iNodes += 10
                    if cType[h] == 'GapJunction':
                        iNodes += 1
                elif cTidx in self.motor:
                    imEdge += 1
                    if cType[h] == 'GapJunction':
                        miEdge += 1
                elif cTidx in self.polymodal:
                    ipEdge += 1
                    if cType[h] == 'GapJunction':
                        piEdge += 1
                elif cTidx in self.other:
                    ioEdge += 1
                    if cType[h] == 'GapJunction':
                        oiEdge += 1
            elif cOidx in self.motor:
                if cTidx in self.sensory:
                    msEdge += 1
                    if cType[h] == 'GapJunction':
                        smEdge += 1
                elif cTidx in self.inter:
                    miEdge += 1
                    if cType[h] == 'GapJunction':
                        imEdge += 1
                elif cTidx in self.motor:
                    mNodes += 10
                    if cType[h] == 'GapJunction':
                        mNodes += 1
                elif cTidx in self.polymodal:
                    mpEdge += 1
                    if cType[h] == 'GapJunction':
                        poEdge += 1
                elif cTidx in self.other:
                    moEdge += 1
                    if cType[h] == 'GapJunction':
                        omEdge += 1
            elif cOidx in self.polymodal:
                if cTidx in self.sensory:
                    psEdge += 1
                    if cType[h] == 'GapJunction':
                        spEdge += 1
                elif cTidx in self.inter:
                    piEdge += 1
                    if cType[h] == 'GapJunction':
                        ipEdge += 1
                elif cTidx in self.motor:
                    pmEdge += 1
                    if cType[h] == 'GapJunction':
                        mpEdge += 1
                elif cTidx in self.polymodal:
                    pNodes += 10
                    if cType[h] == 'GapJunction':
                        pNodes += 1
                elif cTidx in self.other:
                    poEdge += 1
                    if cType[h] == 'GapJunction':
                        opEdge += 1
            elif cOidx in self.other:
                if cTidx in self.sensory:
                    osEdge += 1
                    if cType[h] == 'GapJunction':
                        soEdge += 1
                elif cTidx in self.inter:
                    oiEdge += 1
                    if cType[h] == 'GapJunction':
                        ioEdge += 1
                elif cTidx in self.motor:
                    omEdge += 1
                    if cType[h] == 'GapJunction':
                        moEdge += 1
                elif cTidx in self.polymodal:
                    opEdge += 1
                    if cType[h] == 'GapJunction':
                        poEdge += 1
                elif cTidx in self.other:
                    oNodes += 1
                    if cType[h] == 'GapJunction':
                        oNodes += 10

        x, y = utils.circle_points(100, 5)
        
        pos = {}
        for i in range(5):
            pos[labels[i]] = np.array([x[i], y[i]])
        
        rad = 0.2
        
        nl = [sNodes, iNodes, mNodes, pNodes, oNodes]
        el = [[sNodes, siEdge, smEdge, spEdge, soEdge], 
              [isEdge, iNodes, imEdge, ipEdge, ioEdge],
              [msEdge, miEdge, mNodes, mpEdge, moEdge], 
              [psEdge, piEdge, pmEdge, pNodes, poEdge],
              [osEdge, oiEdge, omEdge, opEdge, oNodes]]
        
        npl = []
        for i in range(5):
            npl.append(Circle((x[i], y[i]), radius=np.log(nl[i]), edgecolor='k', facecolor=color[i], lw=3))
            
        for i in range(5):
            for j in range(5):
                if i == j:
                    pass
                else:
                    c = FancyArrowPatch((x[i], y[i]), (x[j], y[j]), patchA=npl[i], patchB=npl[j], 
                                        shrinkA=0, shrinkB=20, color=color[i], connectionstyle='arc3,rad=%s'%rad, 
                                        mutation_scale=el[i][j]/2, lw=np.log(el[i][j]))
                    ax.add_patch(c)
        
        for i in range(5):
            ax.add_patch(npl[i])
            if with_labels:
                plt.text(x[i], y[i], labels[i], 
                         fontsize=15, horizontalalignment='center', 
                         verticalalignment='center', color='k')
        
        ax.autoscale()
        plt.axis('off')
        plt.tight_layout()
        target_s = mpatches.Patch(color=cmap(1), label='Sensory Neuron')
        target_i = mpatches.Patch(color=cmap(2), label='Interneuron')
        target_m = mpatches.Patch(color=cmap(3), label='Motor Neuron')
        target_y = mpatches.Patch(color=cmap(4), label='Polymodal Neuron')
        target_o = mpatches.Patch(color=cmap(5), label='Other Neuron')
        plt.legend(handles=[target_s, target_i, target_m, target_y, target_o], fontsize=15)
        plt.show()
        
        return el
    
    
    def _trackConnection(self, name, hier=1):
        if type(name) == list:
            nodeList = []
            for l in range(hier+1):
                nodeList.append([])
                
            for i in range(len(name)):
                if type(name[i]) == int:
                    name[i] = self.neuron_id[name[i]]
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
                name = self.neuron_id[name]
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
    morph_neu_id = []
    morph_neu_parent = []
    morph_neu_prox = []
    morph_neu_dist = []
    doc = loaders.NeuroMLLoader.load(fp[f])
    MorphData.neuron_id.append(doc.cells[0].id)
    MorphData.neuron_type.append(doc.cells[0].notes.strip())
    if doc.cells[0].notes.strip() == "SensoryNeuron":
        MorphData.sensory.append(f)
    elif doc.cells[0].notes.strip() == "Interneuron":
        MorphData.inter.append(f)
    elif doc.cells[0].notes.strip() == "Motor Neuron":
        MorphData.motor.append(f)
    elif doc.cells[0].notes.strip() == "PolymodalNeuron":
        MorphData.polymodal.append(f)
    else:
        MorphData.other.append(f)
    sgmts = doc.cells[0].morphology
    for s in range(sgmts.num_segments):
        sgmt = doc.cells[0].morphology.segments[s]
        morph_neu_id.append(sgmt.id)
        if sgmt.parent != None:
            morph_neu_parent.append(sgmt.parent.segments)
        else:
            morph_neu_parent.append(-1)
            MorphData.somaP.append(s)
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
    
    MorphData.morph_id.append(morph_neu_id)
    MorphData.morph_parent.append(morph_neu_parent)
    MorphData.morph_prox.append(morph_neu_prox)
    MorphData.morph_dist.append(morph_neu_dist)
    ctr = Counter(morph_neu_parent)
    ctrVal = list(ctr.values())
    ctrKey = list(ctr.keys())
    BranchData.branchNum[f] = sum(i > 1 for i in ctrVal)
    branchInd = np.array(ctrKey)[np.where(np.array(ctrVal) > 1)[0]]
    
    neu_branchTrk = []
    neu_indBranchTrk = []
    
    list_end = np.setdiff1d(MorphData.morph_id[f], MorphData.morph_parent[f])
    
    BranchData.branchP.append(branchInd)
    MorphData.endP.append(list_end)
    bPoint = np.append(branchInd, list_end)
    
    for bp in range(len(bPoint)):
        neu_branchTrk_temp = []
        neu_branchTrk_temp.append(bPoint[bp])
        parentTrck = bPoint[bp]
        while (parentTrck not in branchInd or bPoint[bp] in branchInd) and (parentTrck != -1):
            parentTrck = MorphData.morph_parent[f][MorphData.morph_id[f].index(parentTrck)]
            if parentTrck != -1:
                neu_branchTrk_temp.append(parentTrck)
        if len(neu_branchTrk_temp) > 1:
            neu_branchTrk.append(neu_branchTrk_temp)
    BranchData.branchTrk.append(neu_branchTrk)
    
    for ep in range(len(list_end)):
        neu_indBranchTrk_temp = []
        neu_indBranchTrk_temp.append(list_end[ep])
        parentTrck = list_end[ep]
        while parentTrck != 0:
            parentTrck = MorphData.morph_parent[f][MorphData.morph_id[f].index(parentTrck)]
            neu_indBranchTrk_temp.append(parentTrck)
        if len(neu_indBranchTrk_temp) > 1:
            neu_indBranchTrk_temp.reverse()
            neu_indBranchTrk.append(neu_indBranchTrk_temp)
    BranchData.indBranchTrk.append(neu_indBranchTrk)


for b in range(len(BranchData.branchTrk)):
    branch_dist_temp1 = []
    length_branch_temp = []
    for sb in range(len(BranchData.branchTrk[b])):
        dist = 0
        branch_dist_temp2 = []
        for sbp in range(len(BranchData.branchTrk[b][sb])):
            branch_dist_temp2.append(np.array(MorphData.morph_dist[b])[np.where(MorphData.morph_id[b] 
                                  == np.array(BranchData.branchTrk[b][sb][sbp]))[0]].flatten().tolist())
            bid = MorphData.morph_id[b].index(BranchData.branchTrk[b][sb][sbp])
            if MorphData.morph_parent[b][bid] != -1:
                pid = MorphData.morph_id[b].index(MorphData.morph_parent[b][bid])
                rhs = MorphData.morph_dist[b][pid][:3]
                lhs = MorphData.morph_dist[b][bid][:3]
                
                dist += np.linalg.norm(np.subtract(rhs, lhs))
        branch_dist_temp2.reverse()
        branch_dist_temp1.append(branch_dist_temp2)
        length_branch_temp.append(dist)
    BranchData.branch_dist.append(branch_dist_temp1)
    LengthData.length_branch.append(length_branch_temp)

#branch_dist_flat = [item for sublist in branch_dist for item in sublist]

LengthData.length_branch_flat = [item for sublist in LengthData.length_branch for item in sublist]
LengthData.length_branch_sensory = []
LengthData.length_branch_inter = []
LengthData.length_branch_motor = []
LengthData.length_branch_polymodal = []
LengthData.length_branch_other = []
LengthData.length_average = np.empty(len(fp))

for lb in range(len(LengthData.length_branch)):
    LengthData.length_total[lb] = np.sum(LengthData.length_branch[lb])
    LengthData.length_average[lb] = np.average(LengthData.length_branch[lb])
    if lb in MorphData.sensory:
        LengthData.length_branch_sensory.append(LengthData.length_branch[lb])
    elif lb in MorphData.inter:
        LengthData.length_branch_inter.append(LengthData.length_branch[lb])
    elif lb in MorphData.motor:
        LengthData.length_branch_motor.append(LengthData.length_branch[lb])
    elif lb in MorphData.polymodal:
        LengthData.length_branch_polymodal.append(LengthData.length_branch[lb])
    elif lb in MorphData.other:
        LengthData.length_branch_other.append(LengthData.length_branch[lb])


LengthData.length_branch_sensory_flat = [item for sublist in LengthData.length_branch_sensory for item in sublist]
LengthData.length_branch_inter_flat = [item for sublist in LengthData.length_branch_inter for item in sublist]
LengthData.length_branch_motor_flat = [item for sublist in LengthData.length_branch_motor for item in sublist]
LengthData.length_branch_polymodal_flat = [item for sublist in LengthData.length_branch_polymodal for item in sublist]
LengthData.length_branch_other_flat = [item for sublist in LengthData.length_branch_other for item in sublist]

MorphData.morph_dist_len = np.array([len(arr) for arr in MorphData.morph_dist])
MorphData.morph_dist_len_EP = np.empty((len(MorphData.morph_dist_len)))
MorphData.endP_len = [len(arr) for arr in MorphData.endP]

indMorph_dist_p_us = []
indMorph_dist_id = []
indMorph_dist_id_s = []
indMorph_dist_id_i = []
indMorph_dist_id_m = []

for i in range(len(BranchData.indBranchTrk)):
    indMorph_dist_temp1 = []
    for j in range(len(BranchData.indBranchTrk[i])):
        indMorph_dist_temp2 = []
        indMorph_dist_p_us.append(1/len(BranchData.indBranchTrk[i]))
        for k in range(len(BranchData.indBranchTrk[i][j])):
            indMorph_dist_temp2.append(np.array(MorphData.morph_dist[i])[np.where(MorphData.morph_id[i] 
                                    == np.array(BranchData.indBranchTrk[i][j][k]))[0]].flatten().tolist())
    
        indMorph_dist_id.append(i)
        if i in MorphData.sensory:
            indMorph_dist_id_s.append(len(indMorph_dist_id)-1)
        elif i in MorphData.inter:
            indMorph_dist_id_i.append(len(indMorph_dist_id)-1)
        elif i in MorphData.motor:
            indMorph_dist_id_m.append(len(indMorph_dist_id)-1)
            
        indMorph_dist_temp1.append(indMorph_dist_temp2)
    BranchData.indMorph_dist.append(indMorph_dist_temp1)

BranchData.indMorph_dist_p_us = np.array(indMorph_dist_p_us)
BranchData.indMorph_dist_flat = [item for sublist in BranchData.indMorph_dist for item in sublist]

MorphData.physLoc = utils.sortPhysLoc(MorphData.morph_dist)

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
    cOriginNeuronCorr.append(MorphData.neuron_id.index(cOriginCKey[i]))
    if cOriginCKey[i] in np.array(MorphData.neuron_id)[MorphData.sensory].tolist():
        cOriginS.append(i)
        cOriginNeuronCorrS.append(MorphData.neuron_id.index(cOriginCKey[i]))
    elif cOriginCKey[i] in np.array(MorphData.neuron_id)[MorphData.inter].tolist():
        cOriginI.append(i)
        cOriginNeuronCorrI.append(MorphData.neuron_id.index(cOriginCKey[i]))
    elif cOriginCKey[i] in np.array(MorphData.neuron_id)[MorphData.motor].tolist():
        cOriginM.append(i)
        cOriginNeuronCorrM.append(MorphData.neuron_id.index(cOriginCKey[i]))
    
cTargetNeuronCorr = []
cTargetNeuronCorrS = []
cTargetS = []
cTargetNeuronCorrI = []
cTargetI = []
cTargetNeuronCorrM = []
cTargetM = []

for i in range(len(cTargetCKey)):
    cTargetNeuronCorr.append(MorphData.neuron_id.index(cTargetCKey[i]))
    if cOriginCKey[i] in np.array(MorphData.neuron_id)[MorphData.sensory].tolist():
        cTargetS.append(i)
        cTargetNeuronCorrS.append(MorphData.neuron_id.index(cTargetCKey[i]))
    elif cOriginCKey[i] in np.array(MorphData.neuron_id)[MorphData.inter].tolist():
        cTargetI.append(i)
        cTargetNeuronCorrI.append(MorphData.neuron_id.index(cTargetCKey[i]))
    elif cOriginCKey[i] in np.array(MorphData.neuron_id)[MorphData.motor].tolist():
        cTargetM.append(i)
        cTargetNeuronCorrM.append(MorphData.neuron_id.index(cTargetCKey[i]))


t1 = time.time()

print('checkpoint 1: ' + str(t1-t0))

np.random.seed(Parameter.SEED)

(MorphData.regMDist, MorphData.regMDistLen) = utils.segmentMorph(Parameter, BranchData)
(MorphData.indRegMDist, MorphData.indRegMDistLen) = utils.indSegmentMorph(Parameter, BranchData)

t2 = time.time()

print('checkpoint 2: ' + str(t2-t1))

(rGy, cML) = utils.radiusOfGyration(MorphData)

(rGyEP, cMLEP) = utils.endPointRadiusOfGyration(MorphData, BranchData)

t3 = time.time()

print('checkpoint 3: ' + str(t3-t2))

(rGyReg, cMLReg) = utils.regularRadiusOfGyration(MorphData.regMDist, MorphData.regMDistLen)

t4 = time.time()

print('checkpoint 4: ' + str(t4-t3))

if Parameter.RUN:
    (OutputData.rGyRegSegs, 
     OutputData.cMLRegSegs, 
     OutputData.regSegOrdNs, 
     OutputData.randTrks) = utils.regularSegmentRadiusOfGyration(Parameter,
                        BranchData,
                        np.array(MorphData.indRegMDist)[indMorph_dist_id_s], 
                        MorphData.indRegMDistLen[indMorph_dist_id_s], 
                        numSample=Parameter.numSample,
                        stochastic=True,
                        p=indMorph_dist_id_s)
    (OutputData.rGyRegSegi, 
     OutputData.cMLRegSegi, 
     OutputData.regSegOrdNi, 
     OutputData.randTrki) = utils.regularSegmentRadiusOfGyration(Parameter,
                        BranchData,
                        np.array(MorphData.indRegMDist)[indMorph_dist_id_i], 
                        MorphData.indRegMDistLen[indMorph_dist_id_i], 
                        numSample=Parameter.numSample,
                        stochastic=True,
                        p=indMorph_dist_id_i)
    (OutputData.rGyRegSegm, 
     OutputData.cMLRegSegm, 
     OutputData.regSegOrdNm, 
     OutputData.randTrkm) = utils.regularSegmentRadiusOfGyration(Parameter,
                        BranchData,
                        np.array(MorphData.indRegMDist)[indMorph_dist_id_m], 
                        MorphData.indRegMDistLen[indMorph_dist_id_m], 
                        numSample=Parameter.numSample,
                        stochastic=True,
                        p=indMorph_dist_id_m)
    
    if Parameter.SAVE:
        utils.exportOutput(Parameter, OutputData)
        
else:
    (OutputData.rGyRegSegs, OutputData.regSegOrdNs, OutputData.randTrks, 
     OutputData.rGyRegSegi, OutputData.regSegOrdNi, OutputData.randTrki, 
     OutputData.rGyRegSegm, OutputData.regSegOrdNm, OutputData.randTrkm) = utils.importData(Parameter)

OutputData.rGyRegSeg = np.concatenate((OutputData.rGyRegSegs, 
                                       OutputData.rGyRegSegi,
                                       OutputData.rGyRegSegm))
OutputData.regSegOrdN = np.concatenate((OutputData.regSegOrdNs,
                                        OutputData.regSegOrdNi, 
                                        OutputData.regSegOrdNm))

t5 = time.time()

print('checkpoint 5: ' + str(t5-t4))



if Parameter.PLOT:

    fig, ax = plt.subplots(1, 2, figsize=(20,6))
    hist1 = ax[0].hist(LengthData.length_total, 
              bins=int((np.max(LengthData.length_total) - np.min(LengthData.length_total))/10),
              density=True)
    ax[0].set_title("Distribution of Total Length", fontsize=20)
    ax[0].set_ylabel("Normalized Density", fontsize=15)
    ax[0].set_xlabel("Total Length", fontsize=15)
    ax[0].set_xlim(0, 1000)
    
    hist2 = ax[1].hist(LengthData.length_total[MorphData.sensory], 
                     bins=int((np.max(LengthData.length_total[MorphData.sensory]) - 
                               np.min(LengthData.length_total[MorphData.sensory]))/10), 
                     density=True, 
                     alpha=0.5)
    hist3 = ax[1].hist(LengthData.length_total[MorphData.inter], 
                     bins=int((np.max(LengthData.length_total[MorphData.inter]) - 
                               np.min(LengthData.length_total[MorphData.inter]))/10),
                     density=True, 
                     alpha=0.5)
    hist4 = ax[1].hist(LengthData.length_total[MorphData.motor],
                     bins=int((np.max(LengthData.length_total[MorphData.motor]) - 
                               np.min(LengthData.length_total[MorphData.motor]))/10), 
                     density=True,
                     alpha=0.5)
    ax[1].set_title("Distribution of Total Length by Type", fontsize=20)
    ax[1].set_ylabel("Normalized Density", fontsize=15)
    ax[1].set_xlabel("Total Length", fontsize=15)
    ax[1].legend(['Sensory', 'Inter', 'Motor'], fontsize=15)
    ax[1].set_xlim(0, 1000)
    plt.tight_layout()
    plt.show()
    
    
    
    # Segment Length Histogram
    
    fig, ax = plt.subplots(1, 2, figsize=(20,6))
    hist1 = ax[0].hist(LengthData.length_branch_flat, 
              bins=int((np.max(LengthData.length_branch_flat) - np.min(LengthData.length_branch_flat))/10),
              density=True)
    ax[0].set_title("Distribution of Segment Length", fontsize=20)
    ax[0].set_ylabel("Normalized Density", fontsize=15)
    ax[0].set_xlabel("Segment Length", fontsize=15)
    
    hist2 = ax[1].hist(LengthData.length_branch_sensory_flat, 
                     bins=int((np.max(LengthData.length_branch_sensory_flat) - 
                               np.min(LengthData.length_branch_sensory_flat))/10), 
                     density=True, 
                     alpha=0.5)
    hist3 = ax[1].hist(LengthData.length_branch_inter_flat, 
                     bins=int((np.max(LengthData.length_branch_inter_flat) - 
                               np.min(LengthData.length_branch_inter_flat))/10),
                     density=True, 
                     alpha=0.5)
    hist4 = ax[1].hist(LengthData.length_branch_motor_flat,
                     bins=int((np.max(LengthData.length_branch_motor_flat) - 
                               np.min(LengthData.length_branch_motor_flat))/10), 
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
    
    def objFuncL(xdata, a):
        y = a*xdata
        
        return y
    
    def objFuncGL(xdata, a, b):
        y = a*xdata + b
        
        return y
    
    
    
    #==============================================================================
    
    popt1, pcov1 = scipy.optimize.curve_fit(objFuncGL, np.log10(hist1centers[np.nonzero(hist1[0])]), 
                                            np.log10(hist1[0][np.nonzero(hist1[0])]), p0=[0.1, -0.1], maxfev=10000)
    
#    popt1, pcov1 = scipy.optimize.curve_fit(objFuncP, hist1centers, hist1[0], p0=[0.1, -0.1], maxfev=10000)
    
    
    fitX = np.linspace(1, 10000, 1000)
    fitY1 = objFuncPpow(fitX, popt1[0], popt1[1])
    
    popt2, pcov2 = scipy.optimize.curve_fit(objFuncGL, np.log10(hist2centers[np.nonzero(hist2[0])]), 
                                            np.log10(hist2[0][np.nonzero(hist2[0])]), p0=[0.1, -0.1], maxfev=10000)
    popt3, pcov3 = scipy.optimize.curve_fit(objFuncGL, np.log10(hist3centers[np.nonzero(hist3[0])]), 
                                            np.log10(hist3[0][np.nonzero(hist3[0])]), p0=[0.1, -0.1], maxfev=10000)
    popt4, pcov4 = scipy.optimize.curve_fit(objFuncGL, np.log10(hist4centers[np.nonzero(hist4[0])]), 
                                            np.log10(hist4[0][np.nonzero(hist4[0])]), p0=[0.1, -0.1], maxfev=10000)
    
#    popt2, pcov2 = scipy.optimize.curve_fit(objFuncP, hist2centers, hist2[0], p0=[0.1, -0.1], maxfev=10000)
#    popt3, pcov3 = scipy.optimize.curve_fit(objFuncP, hist3centers, hist3[0], p0=[0.1, -0.1], maxfev=10000)
#    popt4, pcov4 = scipy.optimize.curve_fit(objFuncP, hist4centers, hist4[0], p0=[0.1, -0.1], maxfev=10000)
    
    fitY2 = objFuncPpow(fitX, popt2[0], popt2[1])
    fitY3 = objFuncPpow(fitX, popt3[0], popt3[1])
    fitY4 = objFuncPpow(fitX, popt4[0], popt4[1])
    
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
    
    
    #==============================================================================
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
    
    #==============================================================================
    # Average Segment Length Histogram
    
    fig, ax = plt.subplots(1, 2, figsize=(20,6))
    hist9 = ax[0].hist(LengthData.length_average,
              bins=int((np.max(LengthData.length_average) - np.min(LengthData.length_average))/10),
              density=True)
    ax[0].set_title("Distribution of Average Segment Length", fontsize=20)
    ax[0].set_ylabel("Normalized Density", fontsize=15)
    ax[0].set_xlabel("Segment Length", fontsize=15)
    
    hist5 = ax[1].hist(LengthData.length_average[MorphData.sensory], 
                     bins=int((np.max(LengthData.length_average[MorphData.sensory]) - 
                               np.min(LengthData.length_average[MorphData.sensory]))/10),
                     density=True,
                     alpha=0.5)
    hist6 = ax[1].hist(LengthData.length_average[MorphData.inter], 
                     bins=int((np.max(LengthData.length_average[MorphData.inter]) -
                               np.min(LengthData.length_average[MorphData.inter]))/10),
                     density=True,
                     alpha=0.5)
    hist7 = ax[1].hist(LengthData.length_average[MorphData.motor],
                     bins=int((np.max(LengthData.length_average[MorphData.motor]) -
                               np.min(LengthData.length_average[MorphData.motor]))/10),
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
    
    #==============================================================================
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
    
    #==============================================================================
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
    
    
    #==============================================================================
    # BranchNum vs Total Segment Length vs Average Segment Length by Type
    
    poptL = []
    
    fig, ax = plt.subplots(4, 3, figsize=(18,24))
    ax[0][0].scatter(LengthData.length_total[MorphData.sensory], BranchData.branchNum[MorphData.sensory])
    ax[0][0].set_title("Sensory Neuron", fontsize=20)
    ax[0][0].set_xlabel("Total Length", fontsize=15)
    ax[0][0].set_ylabel("Number of Branches", fontsize=15)
    ax[0][0].set_xlim(-50, 1000)
    ax[0][0].set_ylim(-1, 8)
    
    ax[0][1].scatter(LengthData.length_total[MorphData.inter], BranchData.branchNum[MorphData.inter])
    ax[0][1].set_title("Interneuron", fontsize=20)
    ax[0][1].set_xlabel("Total Length", fontsize=15)
    ax[0][1].set_xlim(-50, 1000)
    ax[0][1].set_ylim(-1, 8)
    
    ax[0][2].scatter(LengthData.length_total[MorphData.motor], BranchData.branchNum[MorphData.motor])
    ax[0][2].set_title("Motor Neuron", fontsize=20)
    ax[0][2].set_xlabel("Total Length", fontsize=15)
    ax[0][2].set_xlim(-50, 1000)
    ax[0][2].set_ylim(-1, 8)
    
    ax[1][0].scatter(LengthData.length_average[MorphData.sensory], BranchData.branchNum[MorphData.sensory])
    ax[1][0].set_xlabel("Average Segment Length", fontsize=15)
    ax[1][0].set_ylabel("Number of Branches", fontsize=15)
    ax[1][0].set_xlim(-50, 1000)
    ax[1][0].set_ylim(-1, 8)
    
    ax[1][1].scatter(LengthData.length_average[MorphData.inter], BranchData.branchNum[MorphData.inter])
    ax[1][1].set_xlabel("Average Segment Length", fontsize=15)
    ax[1][1].set_xlim(-50, 1000)
    ax[1][1].set_ylim(-1, 8)
    
    ax[1][2].scatter(LengthData.length_average[MorphData.motor], BranchData.branchNum[MorphData.motor])
    ax[1][2].set_xlabel("Average Segment Length", fontsize=15)
    ax[1][2].set_xlim(-50, 1000)
    ax[1][2].set_ylim(-1, 8)
    
    for i in range(len(np.unique(BranchData.branchNum[MorphData.sensory]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.sensory] ==
                            np.unique(BranchData.branchNum[MorphData.sensory])[i])[0]
        ax[2][0].scatter(LengthData.length_average[MorphData.sensory][scttrInd], 
                         LengthData.length_total[MorphData.sensory][scttrInd])
        fitX = np.linspace(0, 1000, 1000)
    ax[2][0].set_xlabel("Average Segment Length", fontsize=15)
    ax[2][0].set_ylabel("Total Length", fontsize=15)
    ax[2][0].legend(np.unique(BranchData.branchNum[MorphData.sensory])[:-1], fontsize=15)
    for i in range(len(np.unique(BranchData.branchNum[MorphData.sensory]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.sensory] == 
                            np.unique(BranchData.branchNum[MorphData.sensory])[i])[0]
        if np.unique(BranchData.branchNum[MorphData.sensory])[i] == 0:
            fitY = objFuncL(fitX, 1)
            ax[2][0].plot(fitX, fitY)
        elif (np.unique(BranchData.branchNum[MorphData.sensory])[i] == 1 or 
              np.unique(BranchData.branchNum[MorphData.sensory])[i] == 2):
            popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                    LengthData.length_average[MorphData.sensory][scttrInd], 
                                                    LengthData.length_total[MorphData.sensory][scttrInd],
                                                    p0=[1.],
                                                    maxfev=10000)
            fitY = objFuncL(fitX, popt[0])
            ax[2][0].plot(fitX, fitY)
            poptL.append(popt[0])
    ax[2][0].set_xlim(-50, 1000)
    ax[2][0].set_ylim(0, 1000)
    
    for i in range(len(np.unique(BranchData.branchNum[MorphData.inter]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.inter] == 
                            np.unique(BranchData.branchNum[MorphData.inter])[i])[0]
        ax[2][1].scatter(LengthData.length_average[MorphData.inter][scttrInd], 
                         LengthData.length_total[MorphData.inter][scttrInd])
    ax[2][1].set_xlabel("Average Segment Length", fontsize=15)
    ax[2][1].legend(np.unique(BranchData.branchNum[MorphData.inter]), fontsize=15)
    for i in range(len(np.unique(BranchData.branchNum[MorphData.inter]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.inter] == 
                            np.unique(BranchData.branchNum[MorphData.inter])[i])[0]
        if np.unique(BranchData.branchNum[MorphData.inter])[i] == 0:
            fitY = objFuncL(fitX, 1)
            ax[2][1].plot(fitX, fitY)
        elif (np.unique(BranchData.branchNum[MorphData.inter])[i] == 1 or 
              np.unique(BranchData.branchNum[MorphData.inter])[i] == 2):
            popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                    LengthData.length_average[MorphData.inter][scttrInd], 
                                                    LengthData.length_total[MorphData.inter][scttrInd],
                                                    p0=[1.],
                                                    maxfev=10000)
            fitY = objFuncL(fitX, popt[0])
            ax[2][1].plot(fitX, fitY)
            poptL.append(popt[0])
    ax[2][1].set_xlim(-50, 1000)
    ax[2][1].set_ylim(0, 1000)
    
    for i in range(len(np.unique(BranchData.branchNum[MorphData.motor]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.motor] == 
                            np.unique(BranchData.branchNum[MorphData.motor])[i])[0]
        ax[2][2].scatter(LengthData.length_average[MorphData.motor][scttrInd], 
                         LengthData.length_total[MorphData.motor][scttrInd])
    ax[2][2].set_xlabel("Average Segment Length", fontsize=15)
    ax[2][2].legend(np.unique(BranchData.branchNum[MorphData.motor]), fontsize=15)
    for i in range(len(np.unique(BranchData.branchNum[MorphData.motor]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.motor] == 
                            np.unique(BranchData.branchNum[MorphData.motor])[i])[0]
        if np.unique(BranchData.branchNum[MorphData.motor])[i] == 0:
            fitY = objFuncL(fitX, 1)
            ax[2][2].plot(fitX, fitY)
        elif (np.unique(BranchData.branchNum[MorphData.motor])[i] == 1 or 
              np.unique(BranchData.branchNum[MorphData.motor])[i] == 2):
            popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                    LengthData.length_average[MorphData.motor][scttrInd], 
                                                    LengthData.length_total[MorphData.motor][scttrInd],
                                                    p0=[1.],
                                                    maxfev=10000)
            fitY = objFuncL(fitX, popt[0])
            ax[2][2].plot(fitX, fitY)
            poptL.append(popt[0])
    ax[2][2].set_xlim(-50, 1000)
    ax[2][2].set_ylim(0, 1000)
    
    
    
    
    length_branch_len = np.array([len(arr) for arr in LengthData.length_branch])
    repeated_length_total = np.repeat(LengthData.length_total, length_branch_len)
    
    for i in range(len(np.unique(BranchData.branchNum[MorphData.sensory]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.sensory] == 
                            np.unique(BranchData.branchNum[MorphData.sensory])[i])[0]
        length_branch_len_sensory = [len(arr) for arr in np.array(LengthData.length_branch)[MorphData.sensory][scttrInd]]
        repeated_length_total_sensory = np.repeat(LengthData.length_total[MorphData.sensory][scttrInd], 
                                                  length_branch_len[MorphData.sensory][scttrInd])
        ax[3][0].scatter([item for sublist in np.array(LengthData.length_branch)[MorphData.sensory][scttrInd].tolist() for item in sublist], 
                         repeated_length_total_sensory)
    ax[3][0].set_xlabel("Segment Length", fontsize=15)
    ax[3][0].set_ylabel("Total Length", fontsize=15)
    ax[3][0].legend(np.unique(BranchData.branchNum[MorphData.sensory])[:-1], fontsize=15)
    for i in range(len(np.unique(BranchData.branchNum[MorphData.sensory]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.sensory] == 
                            np.unique(BranchData.branchNum[MorphData.sensory])[i])[0]
        length_branch_len_sensory = [len(arr) for arr in np.array(LengthData.length_branch)[MorphData.sensory][scttrInd]]
        repeated_length_total_sensory = np.repeat(LengthData.length_total[MorphData.sensory][scttrInd], 
                                                  length_branch_len[MorphData.sensory][scttrInd])
        if np.unique(BranchData.branchNum[MorphData.sensory])[i] == 0:
            fitY = objFuncL(fitX, 1)
            ax[3][0].plot(fitX, fitY)
        elif (np.unique(BranchData.branchNum[MorphData.sensory])[i] == 1 or 
            np.unique(BranchData.branchNum[MorphData.sensory])[i] == 2):
            popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                  [item for sublist in 
                                                   np.array(LengthData.length_branch)[MorphData.sensory][scttrInd].tolist() for item in sublist], 
                                                  repeated_length_total_sensory,
                                                  p0=[1.],
                                                  maxfev=10000)
            fitY = objFuncL(fitX, popt[0])
            ax[3][0].plot(fitX, fitY)
            poptL.append(popt[0])
    ax[3][0].set_xlim(-50, 1000)
    ax[3][0].set_ylim(0, 1000)
    
    for i in range(len(np.unique(BranchData.branchNum[MorphData.inter]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.inter] == 
                            np.unique(BranchData.branchNum[MorphData.inter])[i])[0]
        length_branch_len_inter = [len(arr) for arr in np.array(LengthData.length_branch)[MorphData.inter][scttrInd]]
        repeated_length_total_inter = np.repeat(LengthData.length_total[MorphData.inter][scttrInd], 
                                                length_branch_len[MorphData.inter][scttrInd])
        ax[3][1].scatter([item for sublist in np.array(LengthData.length_branch)[MorphData.inter][scttrInd].tolist() for item in sublist], 
                         repeated_length_total_inter)
    ax[3][1].set_xlabel("Segment Length", fontsize=15)
    ax[3][1].legend(np.unique(BranchData.branchNum[MorphData.inter]), fontsize=15)
    for i in range(len(np.unique(BranchData.branchNum[MorphData.inter]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.inter] == np.unique(BranchData.branchNum[MorphData.inter])[i])[0]
        length_branch_len_inter = [len(arr) for arr in np.array(LengthData.length_branch)[MorphData.inter][scttrInd]]
        repeated_length_total_inter = np.repeat(LengthData.length_total[MorphData.inter][scttrInd], 
                                                length_branch_len[MorphData.inter][scttrInd])
        if np.unique(BranchData.branchNum[MorphData.inter])[i] == 0:
            fitY = objFuncL(fitX, 1)
            ax[3][1].plot(fitX, fitY)
        elif (np.unique(BranchData.branchNum[MorphData.inter])[i] == 1 or 
              np.unique(BranchData.branchNum[MorphData.inter])[i] == 2):
            popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                  [item for sublist in 
                                                   np.array(LengthData.length_branch)[MorphData.inter][scttrInd].tolist() for item in sublist], 
                                                  repeated_length_total_inter,
                                                  p0=[1.],
                                                  maxfev=10000)
            fitY = objFuncL(fitX, popt[0])
            ax[3][1].plot(fitX, fitY)
            poptL.append(popt[0])
    ax[3][1].set_xlim(-50, 1000)
    ax[3][1].set_ylim(0, 1000)
    
    for i in range(len(np.unique(BranchData.branchNum[MorphData.motor]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.motor] == 
                            np.unique(BranchData.branchNum[MorphData.motor])[i])[0]
        length_branch_len_motor = [len(arr) for arr in np.array(LengthData.length_branch)[MorphData.motor][scttrInd]]
        repeated_length_total_motor = np.repeat(LengthData.length_total[MorphData.motor][scttrInd], 
                                                length_branch_len[MorphData.motor][scttrInd])
        ax[3][2].scatter([item for sublist in np.array(LengthData.length_branch)[MorphData.motor][scttrInd].tolist() for item in sublist], 
                         repeated_length_total_motor)
    ax[3][2].set_xlabel("Segment Length", fontsize=15)
    ax[3][2].legend(np.unique(BranchData.branchNum[MorphData.motor]), fontsize=15)
    for i in range(len(np.unique(BranchData.branchNum[MorphData.motor]))):
        scttrInd = np.where(BranchData.branchNum[MorphData.motor] == 
                            np.unique(BranchData.branchNum[MorphData.motor])[i])[0]
        length_branch_len_motor = [len(arr) for arr in np.array(LengthData.length_branch)[MorphData.motor][scttrInd]]
        repeated_length_total_motor = np.repeat(LengthData.length_total[MorphData.motor][scttrInd], 
                                                length_branch_len[MorphData.motor][scttrInd])
        if np.unique(BranchData.branchNum[MorphData.motor])[i] == 0:
            fitY = objFuncL(fitX, 1)
            ax[3][2].plot(fitX, fitY)
        elif (np.unique(BranchData.branchNum[MorphData.motor])[i] == 1 or 
              np.unique(BranchData.branchNum[MorphData.motor])[i] == 2):
            popt, pcov = scipy.optimize.curve_fit(objFuncL, 
                                                  [item for sublist in 
                                                   np.array(LengthData.length_branch)[MorphData.motor][scttrInd].tolist() for item in sublist], 
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
    
    
    #==============================================================================
    
    
    branchEndPDict = {'branch': BranchData.branchNum, 'endP': MorphData.endP_len}
    branchEndPDF = pd.DataFrame(data=branchEndPDict)
    fig = plt.figure(figsize=(8,6))
    seaborn.swarmplot(x='branch', y='endP', data=branchEndPDF.loc[branchEndPDF['branch'] < 197])
    plt.title("Distribution of Number of Endpoints\n for Given Number of Branches", fontsize=20)
    plt.xlabel("Number of Branches", fontsize=15)
    plt.ylabel("Number of Endpoints", fontsize=15)
    #plt.xlim(-1, 10)
    #plt.ylim(-1, 10)
    plt.tight_layout()
    plt.show()
    
    
    #==============================================================================
    
    
    fig = plt.figure(figsize=(8,6))
    seaborn.kdeplot(np.delete(BranchData.branchNum[MorphData.sensory], 
                              np.where(BranchData.branchNum[MorphData.sensory] == 197)[0]), 
                    bw=.6, 
                    label="Sensory")
    seaborn.kdeplot(BranchData.branchNum[MorphData.inter], bw=.6, label="Inter")
    seaborn.kdeplot(BranchData.branchNum[MorphData.motor], bw=.6, label="Motor")
    plt.xlim(-2, 8)
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
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(MorphData.morph_dist_len_EP, rGyEP)
    plt.yscale('log')
    plt.xscale('log')
    #plt.xlim(1, 10000)
    #plt.ylim(0.005, 1000)
    plt.title("Scaling Behavior of $R_{g}$ to $N_{EP}$", fontsize=20)
    plt.xlabel("Number of Nodes", fontsize=15)
    plt.ylabel("Radius of Gyration", fontsize=15)
    plt.tight_layout()
    plt.show()
    
    
    #==============================================================================
    
    #reg_len_scale = np.average(np.divide(regMDistLen, morph_dist_len))
    poptR, pcovR = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(MorphData.regMDistLen*Parameter.sSize), 
                                            np.log10(np.sqrt(np.square(rGyReg)*1/Parameter.sSize)), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR = objFuncPpow(MorphData.regMDistLen*Parameter.sSize, poptR[0], poptR[1])
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(MorphData.regMDistLen*Parameter.sSize, np.sqrt(np.square(rGyReg)*1/Parameter.sSize))
    plt.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR, color='tab:red')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(10, 10000)
    plt.ylim(7, 4000)
    plt.title(r"Scaling Behavior of $R_{g}$ to Length", fontsize=20)
    plt.xlabel(r"Length ($\lambda N$)", fontsize=15)
    plt.ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
    plt.tight_layout()
    plt.show()
    
    
    #==============================================================================
    
    s0 = np.array(MorphData.sensory)[np.where(MorphData.physLoc[MorphData.sensory] == 0)[0]]
    s1 = np.array(MorphData.sensory)[np.where(MorphData.physLoc[MorphData.sensory] == 1)[0]]
    s1 = np.delete(s1, [7,8])
    s2 = np.array(MorphData.sensory)[np.where(MorphData.physLoc[MorphData.sensory] == 2)[0]]
    
    sidx1 = np.where(MorphData.regMDistLen[MorphData.sensory]*Parameter.sSize < 176)[0]
    sidx2 = np.where((MorphData.regMDistLen[MorphData.sensory]*Parameter.sSize > 176) &
                     (MorphData.regMDistLen[MorphData.sensory]*Parameter.sSize < 1e3))[0]
    
    poptR_sidx0, pcovR_sidx0 = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(MorphData.regMDistLen[s0]*Parameter.sSize), 
                                            np.log10(np.sqrt(np.square(rGyReg)[s0]*1/Parameter.sSize)), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR_sidx0 = objFuncPpow(MorphData.regMDistLen*Parameter.sSize, poptR_sidx0[0], poptR_sidx0[1])
    
    
    poptR_sidx1, pcovR_sidx1 = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(MorphData.regMDistLen[s1]*Parameter.sSize), 
                                            np.log10(np.sqrt(np.square(rGyReg)[s1]*1/Parameter.sSize)), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR_sidx1 = objFuncPpow(MorphData.regMDistLen*Parameter.sSize, poptR_sidx1[0], poptR_sidx1[1])
    
    poptR_sidx2, pcovR_sidx2 = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(MorphData.regMDistLen[s2]*Parameter.sSize), 
                                            np.log10(np.sqrt(np.square(rGyReg)[s2]*1/Parameter.sSize)), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR_sidx2 = objFuncPpow(MorphData.regMDistLen*Parameter.sSize, poptR_sidx2[0], poptR_sidx2[1])
    
    
    fig = plt.figure(figsize=(8,6))
    #plt.scatter(regMDistLen[MorphData.sensory]*Parameter.sSize, np.sqrt(np.square(rGyReg)[MorphData.sensory]*1/Parameter.sSize))
    plt.scatter(MorphData.regMDistLen[s0]*Parameter.sSize, np.sqrt(np.square(rGyReg)[s0]*1/Parameter.sSize))
    plt.scatter(MorphData.regMDistLen[s1]*Parameter.sSize, np.sqrt(np.square(rGyReg)[s1]*1/Parameter.sSize))
    plt.scatter(MorphData.regMDistLen[s2]*Parameter.sSize, np.sqrt(np.square(rGyReg)[s2]*1/Parameter.sSize))
    plt.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR_sidx0, color='tab:blue')
    plt.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR_sidx1, color='tab:orange')
    plt.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR_sidx2, color='tab:green')
    plt.legend(['Head', 'Body', 'Tail'], fontsize=15)
    plt.vlines(56, 0.1, 1e4, linestyles='dashed')
    plt.vlines(176, 0.1, 1e4, linestyles='dashed')
    plt.vlines(1000, 0.1, 1e4, linestyles='dashed')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(10, 10000)
    plt.ylim(7, 4000)
    plt.title(r"$R_{g}$ to Length for Sensory Neurons", fontsize=20)
    plt.xlabel(r"Length ($\lambda N$)", fontsize=15)
    plt.ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
    plt.tight_layout()
    plt.show()
    
    
    
    i0 = np.array(MorphData.inter)[np.where(MorphData.physLoc[MorphData.inter] == 0)[0]]
    i1 = np.array(MorphData.inter)[np.where(MorphData.physLoc[MorphData.inter] == 1)[0]]
    i2 = np.array(MorphData.inter)[np.where(MorphData.physLoc[MorphData.inter] == 2)[0]]
    
    iidx1 = np.where(MorphData.regMDistLen[MorphData.inter]*Parameter.sSize < 176)[0]
    iidx2 = np.where((MorphData.regMDistLen[MorphData.inter]*Parameter.sSize > 176) &
                     (MorphData.regMDistLen[MorphData.inter]*Parameter.sSize < 1e3))[0]
    
    poptR_iidx0, pcovR_iidx0 = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(MorphData.regMDistLen[i0]*Parameter.sSize), 
                                            np.log10(np.sqrt(np.square(rGyReg)[i0]*1/Parameter.sSize)), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR_iidx0 = objFuncPpow(MorphData.regMDistLen*Parameter.sSize, poptR_iidx0[0], poptR_iidx0[1])
    
    
    poptR_iidx1, pcovR_iidx1 = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(MorphData.regMDistLen[i1]*Parameter.sSize), 
                                            np.log10(np.sqrt(np.square(rGyReg)[i1]*1/Parameter.sSize)), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR_iidx1 = objFuncPpow(MorphData.regMDistLen*Parameter.sSize, poptR_iidx1[0], poptR_iidx1[1])
    
    poptR_iidx2, pcovR_iidx2 = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(MorphData.regMDistLen[i2]*Parameter.sSize), 
                                            np.log10(np.sqrt(np.square(rGyReg)[i2]*1/Parameter.sSize)), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR_iidx2 = objFuncPpow(MorphData.regMDistLen*Parameter.sSize, poptR_iidx2[0], poptR_iidx2[1])
    
    
    fig = plt.figure(figsize=(8,6))
    #plt.scatter(regMDistLen[MorphData.inter]*Parameter.sSize, np.sqrt(np.square(rGyReg)[MorphData.inter]*1/Parameter.sSize))
    plt.scatter(MorphData.regMDistLen[i0]*Parameter.sSize, 
                np.sqrt(np.square(rGyReg)[i0]*1/Parameter.sSize))
    plt.scatter(MorphData.regMDistLen[i1]*Parameter.sSize, 
                np.sqrt(np.square(rGyReg)[i1]*1/Parameter.sSize))
    plt.scatter(MorphData.regMDistLen[i2]*Parameter.sSize, 
                np.sqrt(np.square(rGyReg)[i2]*1/Parameter.sSize))
    plt.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR_iidx0, color='tab:blue')
    plt.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR_iidx1, color='tab:orange')
    plt.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR_iidx2, color='tab:green')
    plt.legend(['Head', 'Body', 'Tail'], fontsize=15)
    plt.vlines(56, 0.1, 1e4, linestyles='dashed')
    plt.vlines(176, 0.1, 1e4, linestyles='dashed')
    plt.vlines(1000, 0.1, 1e4, linestyles='dashed')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(10, 10000)
    plt.ylim(7, 4000)
    plt.title(r"$R_{g}$ to Length for Interneurons", fontsize=20)
    plt.xlabel(r"Length ($\lambda N$)", fontsize=15)
    plt.ylabel(r"Radius of Gyration ($R^{l}_{g}$)", fontsize=15)
    plt.tight_layout()
    plt.show()
    
    
    
    m0 = np.array(MorphData.motor)[np.where(MorphData.physLoc[MorphData.motor] == 0)[0]]
    m1 = np.array(MorphData.motor)[np.where(MorphData.physLoc[MorphData.motor] == 1)[0]]
    m2 = np.array(MorphData.motor)[np.where(MorphData.physLoc[MorphData.motor] == 2)[0]]
    
    midx1 = np.where(MorphData.regMDistLen[MorphData.motor]*Parameter.sSize < 176)[0]
    midx2 = np.where((MorphData.regMDistLen[MorphData.motor]*Parameter.sSize > 176) &
                     (MorphData.regMDistLen[MorphData.motor]*Parameter.sSize < 1e3))[0]
    
    poptR_midx0, pcovR_midx0 = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(MorphData.regMDistLen[m0]*Parameter.sSize), 
                                            np.log10(np.sqrt(np.square(rGyReg)[m0]*1/Parameter.sSize)), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR_midx0 = objFuncPpow(MorphData.regMDistLen*Parameter.sSize, poptR_midx0[0], poptR_midx0[1])
    
    
    poptR_midx1, pcovR_midx1 = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(MorphData.regMDistLen[m1]*Parameter.sSize), 
                                            np.log10(np.sqrt(np.square(rGyReg)[m1]*1/Parameter.sSize)), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR_midx1 = objFuncPpow(MorphData.regMDistLen*Parameter.sSize, poptR_midx1[0], poptR_midx1[1])
    
    poptR_midx2, pcovR_midx2 = scipy.optimize.curve_fit(objFuncGL, 
                                            np.log10(MorphData.regMDistLen[m2]*Parameter.sSize), 
                                            np.log10(np.sqrt(np.square(rGyReg)[m2]*1/Parameter.sSize)), 
                                            p0=[1., 0.], 
                                            maxfev=100000)
    fitYregR_midx2 = objFuncPpow(MorphData.regMDistLen*Parameter.sSize, poptR_midx2[0], poptR_midx2[1])
    
    
    fig = plt.figure(figsize=(8,6))
    #plt.scatter(regMDistLen[MorphData.motor]*Parameter.sSize, np.sqrt(np.square(rGyReg)[MorphData.motor]*1/Parameter.sSize))
    plt.scatter(MorphData.regMDistLen[m0]*Parameter.sSize, np.sqrt(np.square(rGyReg)[m0]*1/Parameter.sSize))
    plt.scatter(MorphData.regMDistLen[m1]*Parameter.sSize, np.sqrt(np.square(rGyReg)[m1]*1/Parameter.sSize))
    plt.scatter(MorphData.regMDistLen[m2]*Parameter.sSize, np.sqrt(np.square(rGyReg)[m2]*1/Parameter.sSize))
    plt.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR_midx0, color='tab:blue')
    plt.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR_midx1, color='tab:orange')
    plt.plot(MorphData.regMDistLen*Parameter.sSize, fitYregR_midx2, color='tab:green')
    plt.legend(['Head', 'Body', 'Tail'], fontsize=15)
    plt.vlines(56, 0.1, 1e4, linestyles='dashed')
    plt.vlines(176, 0.1, 1e4, linestyles='dashed')
    plt.vlines(1000, 0.1, 1e4, linestyles='dashed')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(10, 10000)
    plt.ylim(7, 4000)
    plt.title(r"$R_{g}$ to Length for Motor Neurons", fontsize=20)
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




