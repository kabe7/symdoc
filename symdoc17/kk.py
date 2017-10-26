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

def warshall_floyd(adj_matrix):
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
                plt.plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]],'k-')

    plt.plot(pos.T[0],pos.T[1],'go')
    plt.title(name)
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(figname)
    plt.clf()
    #plt.show()

def _xpkl(m,P,K,L,n):
    'kk_ver3の補助関数'
    x,p = [P[m], np.r_[P[0:m],P[m+1:n]]]
    k,l = [np.r_[M[m,0:m], M[m,m+1:n]] for M in [K,L]]
    return x,p,k,l

#################################
def kk_ver1(Pos,SpCons,Len,eps=0.001):
    '関数リストを用いた部分最適化'
    t_start = time()
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

    #list of functions
    E_jac, E_hess = [],[]
    for m in range(nodes):
        variables = [P[m,d] for d in range(dim)]
        mth_jac, mth_hess = [partial(sp.lambdify((K,L,P),f,dummify = False),*const) for f in [sp.Matrix([E]).jacobian(variables),sp.hessian(E,variables)]]
        E_jac.append(mth_jac)
        E_hess.append(mth_hess)
        print('generating...',int(m/nodes*100),'%',"\r",end="")


    print('derivative functions are generated:',time()-t_start,'s')

    ##Optimisation
    delta_max = sp.oo
    loops = 0
    while(delta_max>eps):
        max_idx, delta_max = 0, 0
        for m in range(nodes):
            mth_jac = E_jac[m]
            delta = la.norm(mth_jac(Pos))
            if(delta_max < delta):
                delta_max = delta
                max_idx = m

        print(loops,'th:',max_idx,' delta=',delta_max)
        loops += 1
        jac = E_jac[max_idx]
        hess = E_hess[max_idx]
        while(la.norm(jac(Pos))>eps):
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

def kk_ver3(Pos,SpCons,Len,eps=0.001):
    '頂点の匿名性から微分関数を1つにまとめたもの'
    t_start = time()
    nodes,dim = Pos.shape
    ni = nodes-1
    X = sp.IndexedBase('X') # 動かす頂点
    P = sp.IndexedBase('P') # 動かさない頂点
    Ki = sp.IndexedBase('Ki') # 動かす頂点に関するばね定数
    Li = sp.IndexedBase('Li') # 動かす頂点に関する自然長

    j,d = [sp.Idx(*spec) for spec in [('j',ni),('d',dim)]]
    j_range,d_range = [(idx,idx.lower,idx.upper) for idx in [j,d]]

    #potential functionの用意
    print('reserving Potential function')
    dist = sp.sqrt(sp.Sum((X[d]-P[j,d])**2,d_range)).doit()
    Ei = sp.Sum(Ki[j] * (dist-Li[j])**2,j_range)

    #jacobian,Hessianの用意
    print('reserving jacobian and hessian')
    varX = [X[d] for d in range(dim)]
    Ei_jac, Ei_hess = [sp.lambdify((X,P,Ki,Li),sp.simplify(f),dummify = False) for f in [sp.Matrix([Ei]).jacobian(varX), sp.hessian(Ei,varX)]]

    print('fitting')

    ##Optimisation
    delta_max = sp.oo
    xpkl = partial(_xpkl,P=Pos,K=SpCons,L=Len,n=nodes)
    while(delta_max>eps):
        # 最も改善すべき頂点の選択
        norms = np.array(list(map(lambda m: la.norm(Ei_jac(*xpkl(m))),range(nodes))))
        max_idx, delta_max = norms.argmax(), norms.max()

        # Newton法で最適化
        xm,pm,km,lm = xpkl(max_idx)
        while(la.norm(Ei_jac(xm,pm,km,lm))>eps):
            delta_x = la.solve(Ei_hess(xm,pm,km,lm),Ei_jac(xm,pm,km,lm).flatten())
            xm -= delta_x

    print('Finish:',time()-t_start,'s')
    return Pos

#####################################
def kamada_kawai(name,nodeList,consList,L0=10,K0=10,eps=0.01,dim=2):
    n = len(nodeList)
    adj = convertToMatrix(nodeList,consList)
    D = warshall_floyd(adj)

    if(dim==2):
        a = np.arange(n)
        P = L0*np.array([np.cos(2*np.pi/n*a),np.sin(2*np.pi/n*a)]).T
    else:
        P = np.array(L0*np.random.rand(n*dim).reshape(n,dim))

    K = K0 * D**(-2) * (-np.eye(n)+np.ones([n,n]))
    D = D * (-np.eye(n)+np.ones([n,n]))
    L = L0 * D
    if(dim==2):
        draw_2Dgraph(adj,P,name+'_before')

    print('Start optimisation:',name)
    #P = kk_ver1(P,K,L)
    #P = kk_ver2(P,K,L)
    P = kk_ver3(P,K,L)

    if(dim==2):
        draw_2Dgraph(adj,P,name+'_after')
    else:
        print(P)


    D_,K_,L_,P_ = [sp.Matrix(M) for M in [D,K,L,P]]
    markdown(
r'''
##グラフ{name}の描画

頂点を{nodeList}、隣接する点のリストを{consList}とすると、最短経路行列$D$,ばね定数行列$K$,自然長行列$L$は次のようになる。
$$D = {D_},K = {K_},L = {L_}$$
このとき、円周上に並べた頂点からkamada-kawai法により頂点座標の最適化を行なうと、次のようになる。
$$P = {P_}$$

プロットした画像は次のようになる。

![最適化前](pics/{name}_before.png)
![最適化後](pics/{name}_after.png)

''',**locals())

#####################################
def testData():
    triangle = ([0,1,2,3],[[1,2],[0,2],[0,2,3],[2]])
    tetrahedron = ([0,1,2,3],[[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
    double_triangle = ([0,1,2,3,4,5],[[2,3],[4,5],[0,3,5],[0,2],[1,5],[1,2,4]])
    four_triangle = ([0,1,2,3,4,5,6,7,8,9,10,11,12],[[1,4,7,10],[0,2,3],[1,3],[1,2],[0,5,6],[4,6],[4,5],[0,8,9],[7,9],[7,8],[0,11,12],[10,12],[10,11]])
    cube = ([0,1,2,3,4,5,6,7],[[1,3,4],[0,2,5],[1,3,6],[0,2,7],[0,5,7],[1,4,6],[2,5,7],[3,4,6]])
    octahedron = ([0,1,2,3,4,5],[[1,2,3,4],[0,2,4,5],[0,1,3,5],[0,2,4,5],[0,1,3,5],[1,2,3,4]])
    octagon_x = ([0,1,2,3,4,5,6,7],[[1,4,7],[0,2],[1,3,6],[2,4],[0,3,5],[4,6],[2,5,7],[0,6]])
    initial_setting(*cube)

@doit
def _kamada_kawai_intro_(dim=2):
    markdown(
r'''
#力学モデルを用いたグラフ描画

グラフを描画する際、頂点の配置をどのようにするかということは視覚的な理解に大きな影響を及ぼす。
本記事では、グラフの頂点と辺に仮想的な力を割り当て、力学的エネルギーの低い安定状態を探して
グラフのレイアウトを求める**kamada-kawai**法を用いる。
''')

    P = sp.IndexedBase('P')
    K = sp.IndexedBase('k')
    L = sp.IndexedBase('l')

    n = sp.Symbol('n', integer=True)

    i,j,d = [sp.Idx(*spec) for spec in [('i',n),('j',n),('d',dim)]]
    i_range,j_range,d_range = [(idx,idx.lower,idx.upper) for idx in [i,j,d]]

    #potential functionの用意
    dist = sp.sqrt(sp.Sum((P[i,d]-P[j,d])**2,d_range)).doit()
    Potential = K[i,j] * (dist-L[i,j])**2/2
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
で定める($K,L$は定数)。

${d_ij}$は**Warshall-Floyd**のアルゴリズムにより求めることができる。
''',**locals())

    nd = n*dim
    _E = sp.Function('E')

    x = sp.IndexedBase('x')
    m = sp.Symbol('m',integer=True)
    i0 = sp.Idx('i',(1,n))
    i0_range = (i0,i0.lower,i0.upper)

    var0 = [x[d] for d in range(dim)]
    var0_m = sp.Matrix([var0])
    var0_Pm = sp.Matrix([[P[m,d] for d in range(dim)]])

    dist0 = sp.sqrt(sp.Sum((P[i0,d]-x[d])**2,d_range)).doit()
    E0 = sp.Sum(K[i0,0] * (dist0-L[i0,0])**2/2,i0_range)
    E0_jac = sp.simplify(sp.Matrix([E0]).jacobian(var0))
    E0_hess = sp.simplify(sp.hessian(E0,var0))

    delta_x = sp.IndexedBase("\Delta x")
    delta_x_vec = sp.Matrix([[delta_x[d] for d in range(dim)]])
    norm = sp.sqrt(sp.Sum(sp.diff(_E(P[i,d]),P[i,d])**2,d_range).doit())

    markdown(
r'''
##エネルギーの最小化

例として、頂点数が{n},次元が{dim}であるときを考える。
力学的エネルギーが最小になる点では、$gradE = \vec 0$が成り立つ。すなわち、変数が${nd}$個ある${nd}$本の非線型連立方程式を解けばよいのだが、これを解析的に解くのは難しい。
そこで、次のような方法で近似解を求める。

1:まず、特定の頂点1つに着目し、他の頂点の位置を固定する。

2:そして、Newton-Raphson法により選んだ頂点の座標について力学的エネルギー$E$の最小化を行う。

3:着目する頂点を変えて1,2を繰り返す。

4:$\|gradE\|$が十分小さくなったら終了、その時の座標の値を解とする。

以下で、その具体的な方法について述べる。

##近似解の導出

選んだ頂点をmとし、その座標を$P_m = {var0_m}$とする。つまり${var0_Pm} = {var0_m}$である。

このときNewton-Raphson法による反復式は、変数を${var0_m}$としたときのEの1次導関数を$J_m$、2次導関数を$H_m$として
$$H_m {delta_x_vec.T} = -J_m$$
により表される。
これは{dim}元連立1次方程式となり容易に解けて変位$\Delta P_i = {delta_x_vec}$が求められるので、$P_i = P_i + \Delta P_i$により座標を更新する。
以上を繰り返し、変位が十分小さくなったら操作を終了する。

例えば、$m=0$だとすると、反復式は
$${E0_hess}{delta_x_vec.T} = -{E0_jac.T} $$
となる。

選んだ頂点の最適化が終わったら、別な頂点を選んで上記の最適化を繰り返す。
$max_i {norm}$が十分小さくなったら更新を終了してその時の座標を力学的エネルギー$E$が最小となる解とする。
''',**locals())


kamada_kawai('double_triangle',[0,1,2,3,4,5],[[2,3],[4,5],[0,3,5],[0,2],[1,5],[1,2,4]])
kamada_kawai('four_triangle',[0,1,2,3,4,5,6,7,8,9,10,11,12],[[1,4,7,10],[0,2,3],[1,3],[1,2],[0,5,6],[4,6],[4,5],[0,8,9],[7,9],[7,8],[0,11,12],[10,12],[10,11]])
