import argparse
import sys
import typing

from symdoc import Markdown, doit, symfunc, gsym

cmdline_parser = argparse.ArgumentParser()
Markdown.add_parser_option(cmdline_parser)

M = Markdown('kk', title='Network Optimisation')
markdown, cmdline = M.markdown, M.cmdline


######################################################################
# 大域的な環境設定

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.optimize import minimize
from functools import partial
from time import time

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

def draw_2Dgraph(adj_matrix,pos,name='sample'):
    figname = 'pics/' + name + '.png'
    n,dim = pos.shape
    for i in range(n-1):
        for j in range(i,n):
            if(adj_matrix[i,j] == 1):
                plt.plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]], 'k-')

    plt.plot(pos.T[0],pos.T[1],'bo')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(figname)
    plt.show()


#################################
def kk_ver1(Pos,SpCons,Len,eps=0.001):
    '関数リスト'
    t_start = time()
    print('start')
    nodes,dim = Pos.shape
    const = (SpCons,Len)
    P = sp.IndexedBase('P')
    K = sp.IndexedBase('K')
    L = sp.IndexedBase('L')
    i,j,d = [sp.Idx(*spec) for spec in [('i',nodes),('j',nodes),('d',dim)]]
    i_range,j_range,d_range = [(idx,idx.lower,idx.upper) for idx in [i,j,d]]

    #potential functionの用意
    dist = sp.sqrt(sp.Sum((P[i,d]-P[j,d])**2,d_range))
    Potential = 1/2* K[i,j] * (dist-L[i,j])**2
    E = sp.Sum(Potential,i_range,j_range).doit()

    #list of equations,functions
    E_jac, E_hess = [],[]
    for m in range(nodes):
        variables = [P[m,d] for d in range(dim)]
        mth_jac, mth_hess = [partial(sp.lambdify((K,L,P),f,dummify = False),*const) for f in [sp.Matrix([E]).jacobian(variables),sp.hessian(E,variables)]]
        E_jac.append(mth_jac)
        E_hess.append(mth_hess)
    print('generate function:',time()-t_start,'s')

    print('Fitting start')

    ##最適化部分
    delta_max = sp.oo
    while(delta_max>eps):
        max_idx, delta_max = 0, 0
        for m in range(nodes):
            mth_jac = E_jac[m]
            delta = la.norm(mth_jac(Pos))
            if(delta_max < delta):
                delta_max = delta
                max_idx = m

        jac = E_jac[max_idx]
        hess = E_hess[max_idx]
        delta_x = la.solve(hess(Pos),jac(Pos).flatten())
        Pos[max_idx] -= delta_x

    print('Fitting Succeeded')
    print('Finish:',time()-t_start,'s')
    return Pos

def kk_ver2(Pos,SpCons,Len):
    '2n変数の一括最適化'
    t_start = time()
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
    dist = sp.sqrt(sp.Sum((P[i,d]-P[j,d])**2,d_range)).doit()
    E = sp.Sum(K[i,j] * (dist-L[i,j])**2,i_range,j_range)

    #jacobian,Hessianの用意
    print('reserving jacobian and hessian')
    varP = [P[i,d] for i in range(nodes) for d in range(dim)]
    E_jac = sp.Matrix([E]).jacobian(varP)
    E_hes = sp.hessian(E,varP)

    print('generating derivative equation')
    varX = [X[i] for i in range(nodes*dim)]
    PX = np.array([X[i*dim+j] for i in range(nodes) for j in range(dim)]).reshape(nodes,dim)

    E_X = E.replace(K,SpCons).replace(L,Len).replace(P,PX).doit()
    E_jac_X = sp.Matrix([E_X]).jacobian(varX)
    E_hes_X = sp.hessian(E_X,varX)

    print('generating derivative function')
    F,G,H = [sp.lambdify(X,f) for f in [E_X,E_jac_X,E_hes_X]]

    print('fitting')
    res = minimize(F, Pos, jac=lambda x: np.array([G(x)]).flatten(), hess = H, method='trust-ncg')

    print('[time:',time()-t_start,'s]')
    return res.x.reshape(nodes,dim)

#####################################
def setting(name,nodeList,consList,L0=10,K0=10,eps=0.01,dim=2):
    n = len(nodeList)
    adj = convertToMatrix(nodeList,consList)
    D = warshall_floyd(adj)

    if(dim==2):
        P = np.array(np.zeros((n,dim)),dtype=object)
        for m in range(n):
            P[m,0] = np.cos(2*np.pi/n*m)
            P[m,1] = np.sin(2*np.pi/n*m)
    else:
        P = np.array(L0*np.random.rand(n*dim).reshape(n,dim),dtype=object)

    L = L0 * D * (-np.eye(n)+np.ones([n,n]))
    K = K0 * D**(-2) * (-np.eye(n)+np.ones([n,n]))

    P = kk_ver1(P,K,L)
    #P = kk_ver2(P,K,L)

    if(dim==2):
        draw_2Dgraph(adj,P,name)
    else:
        print(P)


    D_,K_,L_,P_ = [sp.Matrix(M) for M in [D,K,L,P]]
    markdown(
r'''
##グラフ{name}の描画

頂点を{nodeList}、隣接する点のリストを{consList}とすると、最短経路行列$D$,ばね定数行列$K$,自然長行列$L$は次のようになる。
$$D = {D_},K = {K_},L = {L_}$$
このとき、ランダム生成した座標からkamada-kawai法により頂点位置の最適化を行なうと、頂点座標は次のようになる。
$$P = {P_}$$

プロットした画像は次のようになる。

![img](pics/{name}.png)

''',**locals())

#####################################

def test():
    triangle = ([0,1,2,3],[[1,2],[0,2],[0,2,3],[2]])
    tetrahedron = ([0,1,2,3],[[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
    double_triangle = ([0,1,2,3,4,5],[[2,3],[4,5],[0,3,5],[0,2],[1,5],[1,2,4]])
    cube = ([0,1,2,3,4,5,6,7],[[1,3,4],[0,2,5],[1,3,6],[0,2,7],[0,5,7],[1,4,6],[2,5,7],[3,4,6]])
    octahedron = ([0,1,2,3,4,5],[[1,2,3,4],[0,2,4,5],[0,1,3,5],[0,2,4,5],[0,1,3,5],[1,2,3,4]])
    octagon_x = ([0,1,2,3,4,5,6,7],[[1,4,7],[0,2],[1,3,6],[2,4],[0,3,5],[4,6],[2,5,7],[0,6]])
    initial_setting(*cube)

@doit
def _kamada_kawai_intro_():
    nodes,dim =3,2
    markdown(
r'''
#力学モデルを用いたグラフ描画

グラフを描画する際、頂点の配置をどのようにするかということは視覚的な理解に大きな影響を及ぼす。
本記事では、グラフの頂点と辺に仮想的な力を割り当て、力学的エネルギーの低い安定状態を探して
グラフのレイアウトを求める*kamada-kawai*法を用いる。
''')

    P = sp.IndexedBase('P')
    K = sp.IndexedBase('k')
    L = sp.IndexedBase('l')

    i,j,d = [sp.Idx(*spec) for spec in [('i',nodes),('j',nodes),('d',dim)]]
    i_range,j_range,d_range = [(idx,idx.lower,idx.upper) for idx in [i,j,d]]

    #potential functionの用意
    dist = sp.sqrt(sp.Sum((P[i,d]-P[j,d])**2,d_range)).doit()
    Potential = K[i,j]/2 * (dist-L[i,j])**2
    E = sp.Sum(Potential,i_range,j_range)/2

    P_id,k_ij,l_ij,d_ij = ['P_{i,d}','k_{i,j}','l_{i,j}','d_{i,j}']

    markdown(
r'''
##力学的エネルギーの定義

まず、グラフの全頂点がばねで結ばれていると仮定すると、系全体の力学的エネルギーは次で表される。
$$E = {E}$$
ただし、${P_id}$はi番目の頂点の座標の第$j$成分、${k_ij},{l_ij}$はそれぞれ頂点$i$と頂点$j$の間のばね定数、自然長とする。

${k_ij},{l_ij}$は、${d_ij}$を頂点$i$と頂点$j$を結ぶ最短経路の長さとして
$${k_ij} = K / {d_ij}^2 (i \neq j) \ \  0(i = j)$$
$${l_ij} = L \times {d_ij}$$
で定める。($K,L$は定数)
${d_ij}$は*Warshall-Floyd*のアルゴリズムにより求めることができる。
''',**locals())

    nd = nodes*dim
    P_m = sp.var('P_m')
    _E = sp.Function('E')
    dEdPm = sp.diff(_E(P_m),P_m)
    E = E.doit()
    var1 = [P[1,d] for d in range(dim)]
    E_jac_1 = sp.simplify(sp.Matrix([E]).jacobian(var1))
    E_hess_1 = sp.hessian(E,var1) #simplifyするのにすごい時間がかかる
    markdown(
r'''
##エネルギーの最小化

例として、頂点数が{nodes},次元が{dim}であるときを考える。
力学的エネルギーが最小になる点では、$gradE = \vec 0$が成り立つ。
すなわち、${nd}$本の非線型連立方程式を解けばよいのだが、これを解析的に解くのは難しい。
そこで、Newton-Raphson法を用いて近似解を求める。

##近似解の導出
まず、$\|{dEdPm}\||$が最大となる添字$i$を探す($P_i$はベクトルであることに注意)。
このとき、$P_i$の各成分を変数としたときのEのヤコビアンを$J_i$、ヘッシアンを$H_i$として
$$H_m \Delta P_m = -J_m$$
により変位$\Delta P_i$を求め、$P_i = P_i + \Delta P_i$により座標を更新する。
例えば、$i=1$だとすると、$\Delta P_1を求める式は$
$${E_hess_1}\Delta P_1 = -{E_jac_1.T} $$
となる。
以上を繰り返し、$max_i \|{dEdPm}\||$が十分小さくなったら更新を終了して
その時の座標を力学的エネルギー$E$が最小となる解とする

''',**locals())


setting('double_triangle',[0,1,2,3,4,5],[[2,3],[4,5],[0,3,5],[0,2],[1,5],[1,2,4]])
setting('cube',[0,1,2,3,4,5,6,7],[[1,3,4],[0,2,5],[1,3,6],[0,2,7],[0,5,7],[1,4,6],[2,5,7],[3,4,6]])
