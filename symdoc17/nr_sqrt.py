import argparse
import sys
import typing


from symdoc import Markdown, doit, symfunc, gsym

cmdline_parser = argparse.ArgumentParser()
Markdown.add_parser_option(cmdline_parser)

M = Markdown('nr_sqrt', title='Calculate Sqrt by Newton-Raphson Method')
markdown, cmdline = M.markdown, M.cmdline


######################################################################
# 大域的な環境設定

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

sp.var('x a n')

######################################################################
# ユーティリティ関数

def NewtonRaphson(関数, x, *args):
    x_n,x_n1 = sp.var('x_n x_{n+1}')
    F = sp.Function('F')
    NR = x - F(x) / sp.diff(F(x),x)

    itr_equation = sp.simplify(NR.subs(F(x),関数).doit())
    itr_func = sp.lambdify((x,*args), itr_equation)

    func = sp.lambdify((x, *args), 関数)

    def newton_raphson(*args, x=1, loops=5):
        x0 = x
        for i in range(loops):
            x = itr_func(x,*args)

        x_loops = 'x_{'+str(loops)+'}'

        markdown(
r'''
$a={args[0]}$として、初期値$x_0 = {x0}$から反復を{loops}回行うと近似解として次を得る。
$$\alpha \approx {x_loops} = {x}$$
''',**locals())

        return x

    itr_eq_xn = itr_equation.subs(x,x_n)

    markdown(
r'''
## ${関数}=0$の近似解の導出

Newton-Raphson法を基に、${関数}=0$の解に収束する反復列を作ると次のようになる。

$${x_n1} = {itr_eq_xn}$$

''',**locals())

    return newton_raphson

######################################################################

@doit
def __newton_raphson__():
    F = sp.Function('F')
    x_n,x_n1 = sp.var('x_n x_{n+1}')
    NR = x - F(x) / sp.diff(F(x),x)
    NR_xn = NR.subs(x,x_n)

    markdown(
r'''
#Newton-Raphson法による近似解の計算

*Newton-Raphson法*によれば、適当な初期値$x_0$から
$${x_n1} = {NR_xn}$$
により反復列を作ると、$F(x)=0$の解 $\alpha$ に収束する数列となる。
これを利用して、解$\alpha$を近似的に求めることができる。
''',**locals()
    )

mysqrt = NewtonRaphson(x**2-a,x,a)
mysqrt(2)
mysqrt(3,x=100,loops=10)

mycbrt = NewtonRaphson(x**3-a,x,a)
mycbrt(4)
mycbrt(10,x=1,loops=10)

myequaion= NewtonRaphson(x/sp.tan(x)-1/a,x,a)
myequaion(1,x=5,loops=100)

######################################################################
