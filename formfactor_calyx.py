# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:59:00 2020

@author: user
"""

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import itertools
import ctypes

def formfactor(args):
    # with calyx_dist_flat_glo.get_lock:
    calyx_dist_flat_glo_r = np.frombuffer(calyx_dist_flat_glo.get_obj())
    calyx_dist_flat_glo_s = calyx_dist_flat_glo_r.reshape((n_glo.value,m_glo.value))
    ffq = np.sum(np.cos(np.dot(np.logspace(-2,3,100)[args[0]]*np.array([1,0,0]), 
                               np.subtract(calyx_dist_flat_glo_s[args[1]], calyx_dist_flat_glo_s[1+args[1]:]).T)))
    return ffq

def parallelinit(calyx_dist_flat_glo_, n_glo_, m_glo_):
    global calyx_dist_flat_glo, n_glo, m_glo
    calyx_dist_flat_glo = calyx_dist_flat_glo_
    n_glo = n_glo_
    m_glo = m_glo_

if __name__ == '__main__': 
    calyx_dist_flat = np.load(r'./calyx_dist_flat.npy')
    
    n = np.shape(calyx_dist_flat)[0]
    m = np.shape(calyx_dist_flat)[1]
    q_range = np.logspace(-2,3,100)
    # r_x = np.array([1, 0, 0])
    
    # q_range_glo = mp.Array(ctypes.c_double, q_range)
    calyx_dist_flat_glo =  mp.Array(ctypes.c_double, calyx_dist_flat.flatten())
    n_glo = mp.Value(ctypes.c_int, n)
    m_glo = mp.Value(ctypes.c_int, m)
    # r_x_glo = mp.Array(ctypes.c_double, r_x)
    
    paramlist = list(itertools.product(range(100), range(n)))
    
    pool = mp.Pool(20, initializer=parallelinit, initargs=(calyx_dist_flat_glo, n_glo, m_glo))
    
    t1 = time.time()
    
    results = pool.map(formfactor, paramlist)
    pool.close()
    
    t2 = time.time()
    
    print(t2-t1)
    
    np.save(r'./calyx_results.npy', results)
    
    Pq = 2*np.divide(np.sum(np.array(results).reshape(100, n), axis=1), n)
    
    # fig = plt.figure(figsize=(8,6))
    # plt.plot(q_range, Pq, lw=3, color='tab:orange')
    # plt.xscale('log')
    # plt.xlabel('$q$', fontsize=15)
    # plt.ylabel('$P(q)$', fontsize=15)
    # plt.tight_layout()
    # plt.savefig(r'./calyx_form_factor.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(q_range, Pq, lw=3, color='tab:orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$q$', fontsize=15)
    plt.ylabel('$P(q)$', fontsize=15)
    plt.tight_layout()
    plt.savefig(r'./calyx_form_factor_log.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
