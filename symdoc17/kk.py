import argparse
import sys
import typing

from symdoc import Markdown, doit, symfunc, gsym

cmdline_parser = argparse.ArgumentParser()
Markdown.add_parser_option(cmdline_parser)

M = Markdown('kk', title='kamada-kawai')
markdown, cmdline = M.markdown, M.cmdline


######################################################################
# 大域的な環境設定

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.optimize import minimize
from functools import partial
import time

######################################################################
# ユーティリティ関数

def convertToMatrix(nodeList,adj_list):
    '隣接リスト -> 隣接行列'
    n = len(nodeList)
    adj_matrix = np.zeros([n,n])

    for i in range(n):
        edges_of_i = len(adj_list[i])
        for j in range(edges_of_i):
            for k in range(n):
                if(adj_list[i][j] == nodeList[k]):
                    adj_matrix[i][k] = 1

    return adj_matrix

def warshall_floyd(adj_matrix:np.ndarray):
    n = adj_matrix.shape[0]
    # generate distance_matrix
    distance_matrix = -n*adj_matrix + n+1

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if(distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j] ):
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]

    return distance_matrix

def findBiggestPotentialIndex(jacobian,pos):
    '''
    jacobian:関数リスト、リストのm番目にはEをm番目の座標で偏微分したときの関数が入っているとする
    pos:点の座標群
    '''
    nodes = pos.shape[0]
    max_idx = 0
    delta_max = 0
    for m in range(nodes):
        mth_jac = jacobian[m]
        delta = la.norm(mth_jac(pos))
        if(delta_max < delta):
            delta_max = delta
            max_idx = m

    return max_idx, delta_max

def draw_2Dgraph(adj_matrix,pos):
    n,dim = pos.shape
    for i in range(n-1):
        for j in range(i,n):
            if(adj_matrix[i,j] == 1):
                plt.plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]], 'k-')

    plt.plot(pos.T[0],pos.T[1],'bo')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

#################################
def kamada_kawai(Pos,SpCons,Len,eps):
    nodes,dim = Pos.shape
    const = (SpCons,Len)
    P = sp.IndexedBase('P')
    K = sp.IndexedBase('K')
    L = sp.IndexedBase('L')
    X = sp.IndexedBase('X')

    i,j,d = [sp.Idx(*spec) for spec in [('i',nodes),('j',nodes),('d',dim)]]
    i_range,j_range,d_range = [(idx,idx.lower,idx.upper) for idx in [i,j,d]]

    #potential functionの用意
    dist = sp.sqrt(sp.Sum((P[i,d]-P[j,d])**2,d_range))
    E = sp.Sum(1/2* K[i,j] * (dist-L[i,j])**2,i_range,j_range)
    E_eq = E.doit()

    #list of equations,functions
    def generate_eqfc_list(nodes,dim,E,*const):
        '関数Eのjacobian,Hessianの関数式と関数を生成'

        E_jac_eq, E_jac_fc, E_hes_eq, E_hes_fc = [],[],[],[]

        for m in range(nodes):
            variables = [P[m,d] for d in range(dim)]
            E_jac_eq_m = sp.Matrix([E]).jacobian(variables)
            E_jac_eq.append(E_jac_eq_m)

            E_jac_fc_m = partial(sp.lambdify((K,L,P),E_jac_eq_m,dummify = False),*const)
            E_jac_fc.append(E_jac_fc_m)

            E_hes_eq_m = sp.hessian(E,variables)
            E_hes_eq.append(E_hes_eq_m)

            E_hes_fc_m = partial(sp.lambdify((K,L,P),E_hes_eq_m,dummify = False),*const)
            E_hes_fc.append(E_hes_fc_m)

        return E_jac_eq,E_jac_fc,E_hes_eq,E_hes_fc

    E_jac_eq, E_jac_fc, E_hes_eq, E_hes_fc = generate_eqfc_list(nodes,dim,E_eq,*const)

    print('Fitting start')

    ##loop部分
    delta = sp.oo
    loops=1
    while(delta>eps):
        m, delta = findBiggestPotentialIndex(E_jac_fc,Pos)

        x0 = Pos[m].copy()
        for i in range(dim):
            Pos[m,i] = X[i]

        def generate_x_function(equation):
            return sp.lambdify(X,equation.replace(K,SpCons).replace(L,Len).replace(P,Pos).doit())

        F,G,H = [generate_x_function(f) for f in [E_eq,E_jac_eq[m],E_hes_eq[m]]]

        res = minimize(F, x0.tolist(), jac=lambda x: G(x).flatten(), hess=H, method='trust-ncg')

        Pos[m] = res.x
        print(loops,'th: delta=',delta,'Particle',m,':',x0,'->',Pos[m])
        loops+=1

    print('Fitting Succeeded')

    return Pos

def kk_ver2(Pos,SpCons,Len,eps):
    t_start = time.time()
    nodes,dim = Pos.shape
    const = (SpCons,Len)
    P = sp.IndexedBase('P')
    K = sp.IndexedBase('K')
    L = sp.IndexedBase('L')
    X = sp.IndexedBase('X')

    i,j,d = [sp.Idx(*spec) for spec in [('i',nodes),('j',nodes),('d',dim)]]
    i_range,j_range,d_range = [(idx,idx.lower,idx.upper) for idx in [i,j,d]]

    #potential functionの用意
    print('reserving Potential function')
    start = time.time()
    dist = sp.sqrt(sp.Sum((P[i,d]-P[j,d])**2,d_range)).doit()
    E = sp.Sum(K[i,j] * (dist-L[i,j])**2,i_range,j_range)
    end = time.time()
    print('[time:',end-start,'s]')

    #jacobian,Hessianの用意
    print('reserving jacobian and hessian')
    start = time.time()
    varP = [P[i,d] for i in range(nodes) for d in range(dim)]
    E_jac = sp.Matrix([E]).jacobian(varP)
    E_hes = sp.hessian(E,varP)
    end = time.time()
    print('[time:',end-start,'s]')


    print('generating derivative equation')
    varX = [X[i] for i in range(nodes*dim)]
    start = time.time()
    PX = np.array([X[i*dim+j] for i in range(nodes) for j in range(dim)]).reshape(nodes,dim)

    E_X = E.replace(K,SpCons).replace(L,Len).replace(P,PX).doit()
    E_jac_X = sp.Matrix([E_X]).jacobian(varX)
    E_hes_X = sp.hessian(E_X,varX)
    end = time.time()
    print('[time:',end-start,'s]')

    print('generating derivative function')
    start = time.time()
    F,G,H = [sp.lambdify(X,f) for f in [E_X,E_jac_X,E_hes_X]]
    end = time.time()
    print('[time:',end-start,'s]')

    print('fitting')
    start = time.time()
    res = minimize(F, Pos, jac=lambda x: np.array([G(x)]).flatten(), hess = H, method='trust-ncg')
    end = time.time()
    print('[time:',end-start,'s]')

    print('[time:',time.time()-t_start,'s]')
    return res.x.reshape(nodes,dim)


#####################################

def initial_setting(nodeList,consList,L0=10,K0=10,eps=0.01,dim=2):
    n = len(nodeList)
    adj = convertToMatrix(nodeList,consList)
    d = warshall_floyd(adj)

    P = np.array(np.random.rand(n*dim).reshape(n,dim),dtype=object)
    L = L0 * d * (-np.eye(n)+np.ones([n,n]))
    K = K0 * d**(-2) * (-np.eye(n)+np.ones([n,n]))

    P = kk_ver2(P,K,L,eps)

    if(dim==2):
        draw_2Dgraph(adj,P)
    else:
        print(P)


#####################################
triangle = ([0,1,2,3],[[1,2],[0,2],[0,2,3],[2]])
double_triangle = ([0,1,2,3,4,5],[[2,3],[4,5],[0,3,5],[0,2],[1,5],[1,2,4]])
cube = ([0,1,2,3,4,5,6,7],[[1,3,4],[0,2,5],[1,3,6],[0,2,7],[0,5,7],[1,4,6],[2,5,7],[3,4,6]])
octahedron = ([0,1,2,3,4,5],[[1,2,3,4],[0,2,4,5],[0,1,3,5],[0,2,4,5],[0,1,3,5],[1,2,3,4]])
octagon_x = ([0,1,2,3,4,5,6,7],[[1,4,7],[0,2],[1,3,6],[2,4],[0,3,5],[4,6],[2,5,7],[0,6]])

@doit
def test():
    initial_setting(*cube)
