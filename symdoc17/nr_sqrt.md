---
title: Calculate Sqrt by Newton-Raphson Method
layout: page
---



#Newton-Raphson法による近似解の計算

*Newton-Raphson法*によれば、適当な初期値$x_0$から
$$x_{n+1} = x_{n} - \frac{F{\left (x_{n} \right )}}{\frac{d}{d x_{n}} F{\left (x_{n} \right )}}$$
により反復列を作ると、$F(x)=0$の解 $\alpha$ に収束する数列となる。
これを利用して、解$\alpha$を近似的に求めることができる。



## $- a + x^{2}=0$の近似解の導出

Newton-Raphson法を基に、$- a + x^{2}=0$の解に収束する反復列を作ると次のようになる。

$$x_{n+1} = \frac{a + x_{n}^{2}}{2 x_{n}}$$




$a=2$として、初期値$x_0 = 1$から反復を5回行うと近似解として次を得る。
$$\alpha \approx x_{5} = 1.414213562373095$$



$a=3$として、初期値$x_0 = 100$から反復を10回行うと近似解として次を得る。
$$\alpha \approx x_{10} = 1.7320508075688787$$



## $- a + x^{3}=0$の近似解の導出

Newton-Raphson法を基に、$- a + x^{3}=0$の解に収束する反復列を作ると次のようになる。

$$x_{n+1} = \frac{a + 2 x_{n}^{3}}{3 x_{n}^{2}}$$




$a=4$として、初期値$x_0 = 1$から反復を5回行うと近似解として次を得る。
$$\alpha \approx x_{5} = 1.587401052015271$$



$a=10$として、初期値$x_0 = 1$から反復を10回行うと近似解として次を得る。
$$\alpha \approx x_{10} = 2.154434690031884$$



## $\frac{x}{\tan{\left (x \right )}} - \frac{1}{a}=0$の近似解の導出

Newton-Raphson法を基に、$\frac{x}{\tan{\left (x \right )}} - \frac{1}{a}=0$の解に収束する反復列を作ると次のようになる。

$$x_{n+1} = \frac{2 a x_{n}^{2} + \cos{\left (2 x_{n} \right )} - 1}{a \left(2 x_{n} - \sin{\left (2 x_{n} \right )}\right)}$$




$a=1$として、初期値$x_0 = 5$から反復を100回行うと近似解として次を得る。
$$\alpha \approx x_{100} = 4.493409457909064$$


