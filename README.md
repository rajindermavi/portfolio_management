
# Deep Financial Portfolio Management

The financial portfolio management problem is to choose an investment strategy given a selection of financial products. In this repository we implement a deep portfolio management algorithm and compare it to several benchmarking strategies. 

There are three branches associated to this repository:

* **main** - Exposition of the basic logic of the training of the reinforcement learning agent.

* **interactive_implementation** - Implements an interactive streamlit app, the app allows users to select a date range and set of stocks. [Give the interactive app a try.](https://mavi-portfolio-management.herokuapp.com/)

* **pm_with_nlp** - (In progress) The goal of this branch is to incorporate news data to the deep portfolio management algorithm.

## Details about the trading agents

In training and validation simulations each agent is initialized with 1M in a cash account. On each trading day agents are provided with trading data (max, min, opening, closing prices) of d securities from the last 64 days and must select an investment strategy for the following day. The values of the securities at are given by a vector
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=Z^{(t)}=\begin{pmatrix}Z^{(t)}_0\\Z^{(t)}_1\\\vdots\\Z^{(t)}_d\end{pmatrix}.">
</p>
The first d values are risky securities and the final value represents a risk free investment, a cash account earning 1% interest per year.
 The portfolio at each time step may be expressed as a vector
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=X^{(t)}=\begin{pmatrix}X^{(t)}_0\\X^{(t)}_1\\\vdots\\X^{(t)}_d\end{pmatrix}.">
</p>
Each value <img src="https://render.githubusercontent.com/render/math?math=X_i^{(t)}"> is the cash value in the corresponding security. We constrain all holdings to be positive. The total value of the portfolio on time step t is
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=V^{(t)}=\sum_{i=0}^dX^{(t)}_i.">
</p> 
The output of each agent (the daily investment strategy) is expressed in terms of the portfolio weight vector 
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=w^{(t)}=\frac{1}{V^{(t)}}\begin{pmatrix}X^{(t)}_0\\X^{(t)}_1\\\vdots\\X^{(t)}_d\end{pmatrix}.">
</p> 
At the close of each trading day each agent refinances their portfolios yeilding an updated weight vector <img src="https://render.githubusercontent.com/render/math?math=\hat{w^{(t)}}."> A trading penalty is assessed proportional to the total volume of the trade
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=penalty=(10^{-4})V^{(t)}\|w^{(t)}-\hat{w^{(t)}}\|_1.">
</p> 
The closing value of the porfolio on day t is then <img src="https://render.githubusercontent.com/render/math?math=\hat{V(t)}=V^{(t)}-penalty."> 

We then allow trading on day t+1 to occur and subsequently recalculate the portfolio positions as follows. The evolution of the porfolio value is calculated as
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{V(t+1)}{\hat{V(t)}}=(Z^{(t+1)}{\oslash}Z^{(t)}){\cdot}w^{(t)}.">
</p> 

### Deep Portfolio Management

