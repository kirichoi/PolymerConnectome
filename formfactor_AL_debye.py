# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:59:00 2020

@author: user
"""

import numpy as np
from scipy import spatial
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import itertools
import ctypes

def formfactor(args):
    AL_dist_flat_glo_r = np.frombuffer(AL_dist_flat_glo.get_obj())
    AL_dist_flat_glo_s = AL_dist_flat_glo_r.reshape((n_glo.value,m_glo.value))
    qrvec = np.logspace(-2,3,100)[args[0]]*spatial.distance.cdist(np.array([AL_dist_flat_glo_s[args[1]]]), AL_dist_flat_glo_s)[0]
    qrvec = qrvec[np.nonzero(qrvec)]
    ffq = np.sum(np.sin(qrvec)/qrvec)
    return ffq

def parallelinit(AL_dist_flat_glo_, n_glo_, m_glo_):
    global AL_dist_flat_glo, n_glo, m_glo
    AL_dist_flat_glo = AL_dist_flat_glo_
    n_glo = n_glo_
    m_glo = m_glo_

if __name__ == '__main__': 
    AL_dist_flat = np.load(r'./AL_dist_flat.npy')
    
    n = np.shape(AL_dist_flat)[0]
    m = np.shape(AL_dist_flat)[1]
    q_range = np.logspace(-2,3,100)
    
    # q_range_glo = mp.Array(ctypes.c_double, q_range)
    AL_dist_flat_glo =  mp.Array(ctypes.c_double, AL_dist_flat.flatten())
    n_glo = mp.Value(ctypes.c_int, n)
    m_glo = mp.Value(ctypes.c_int, m)
    
    paramlist = list(itertools.product(range(100), range(n)))
    
    pool = mp.Pool(20, initializer=parallelinit, initargs=(AL_dist_flat_glo, n_glo, m_glo))
    
    t1 = time.time()
    
    results = pool.map(formfactor, paramlist)
    pool.close()
    
    t2 = time.time()
    
    print(t2-t1)
    
    np.save(r'./AL_results_debye.npy', results)
    
    Pq = np.divide(np.sum(np.divide(np.array(results).reshape(100, n), n), axis=1), n)
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(q_range, Pq, lw=3, color='tab:orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$q$', fontsize=15)
    plt.ylabel('$P(q)$', fontsize=15)
    plt.tight_layout()
    plt.savefig(r'./AL_form_factor_debye_log.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
