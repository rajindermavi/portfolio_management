
# Deep Financial Portfolio Management

The financial portfolio management problem is to choose an investment strategy given a selection of financial products. In this repository we implement a deep portfolio management algorithm and compare it to several benchmarking strategies. 

There are three branches associated to this repository:

* **main** - Exposition of the basic logic of the training of the reinforcement learning agent.

* **interactive_implementation** - Implements an interactive streamlit app, the app allows users to select a date range and set of stocks.

* **pm_with_nlp** - (In progress) The goal of this branch is to incorporate news data to the deep portfolio management algorithm.

## Details about the trading agents

In training and validation simulations each agent is initialized with 1M in a cash account. On each trading day agents are provided with trading data (max,min,opening,closing prices) from the last 64 days and must select an investment strategy for the following day. The portfolio at each time step may be expressed as a vector
<img src="https://render.githubusercontent.com/render/math?math= X^{(t)} = \begin{pmatrix} X^{(t)}_0\\ X^{(t)}_1\\ \vdots \\X^{(t)}_d\end{pmatrix} ">
$$ X^{(t)} = \begin{pmatrix} X^{(t)}_0\\ X^{(t)}_1\\ \vdots \\X^{(t)}_d\end{pmatrix} $$
 where the first $d$ values are investments in stocks and the final value is held in an account earning a risk free return of $1\%$ per year. The total value of the portfolio on each time step is $$V^{(t)} = \sum_{i=0}^d X^{(t)}_i. $$ The output of each agent is expressed in terms of the portfolio weight vector $$ w^(t) = (w^{(t)}_0,...,w^{(t)}_d)^T = \frac{1}{V^{(t)}}(X^{(t)}_0,X^{(t)}_1,...,X^{(t)}_d)^T .$$  

### Deep Portfolio Management

