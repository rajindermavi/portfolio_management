import streamlit as st
import datetime 
import plotly.graph_objects as go
from model import AgentComparison


def plot_histories(agents):
    start = agents.meta_env.env._start_tick
    date_range=agents.meta_env.dates[start:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date_range,
                             y=agents.dpm_vals,
                             mode='lines',name='DeepPortfolio'))
    fig.add_trace(go.Scatter(x=date_range,
                             y=agents.uniform_vals,
                             mode='lines',name='Uniform'))
    fig.add_trace(go.Scatter(x=date_range,
                             y=agents.capm_vals,
                             mode='lines',name='CAPM'))
    fig.add_trace(go.Scatter(x=date_range,
                             y=agents.mvp_vals,
                             mode='lines',name='MVP'))
                             
    return fig,date_range

def main():
    st.title('Portfolio Management')
    x = st.slider("Select a calendar date range", datetime.date(2017,1,1), datetime.date.today(), 
                    (datetime.date(2017,1,1), datetime.date.today()))
    start=x[0].isoformat()
    end=x[1].isoformat()
    all_symbols = ["AAPL","ABMD","ABT","ALB","ALK",
                     "AMZN","ATVI",
                     "ADBE","AFL","AMD","BMY","BSX","CB","CVX","DIS",
                   "GE","GS","HAS","HD","HON","HPQ","HUM","IBM","INTC",
                   "JNJ","LOW","MMM","MSFT","NEE","NKE",
                   "NVDA","PEP","PFE","SBUX","WFC","WMT"]
    default_symbols = ["BSX","MMM","AMD","GE","WMT"]
    symbols = st.multiselect(
             'Select symbols for your portfolio.',
             all_symbols,
             default_symbols) 
 
    if len(symbols) == 0:
        st.write('Please select at least one symbol')
    else:
        with st.spinner('Simulating portfolios...'):
            agents = AgentComparison(symbols,start,end)
    
        fig,date_range = plot_histories(agents) 
        d_init = date_range.iloc[0].date().isoformat()
        d_final = date_range.iloc[-1].date().isoformat()
        
        st.write('''Each agent begins with 1M in cash and
                    rebalances its portfolio each day using
                    data from previous 64 trading days.''')
        st.write(f'First trading day: {d_init}. Final trading day: {d_final}')
        st.write(fig)

    st.subheader('About the agents:') 
    st.markdown('''In addition to stock investments, 
                we assume a risk free investment earning a $1\%$ return year.''')
    st.markdown('''**DeepPortfolio** is a deep reinforcement 
                learning portfolio management algorithm based on the paper 
                [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem]
                (https://arxiv.org/abs/1706.10059). 
                The network architecture consists of 3 convolutional layers with ELU and LeakyRelu activation layers.
                The agent was trained on data from 36 stocks from Jan 2000 to Dec 2016.''')
    st.markdown('''**CAPM -- Capital Asset Pricing Model** uses recent stock price data to estimate the expected return $R$ and covariance $\Sigma$ for the collection of stocks. 
                   The agent then selects the portfolio weight $w$ maximizing
                   the Sharpe ratio $(R - R_f)/(\sqrt{w^T\Sigma w})$ constrained to take long positions.''')
    st.markdown('''**MVP -- Minimum Variance Portfolio** also uses recent stock price data to estimate the covariance of the collection of stocks.
                   The agent then selects the portfolio minimizing the risk (variance) $w^T\Sigma w$ constrained to keep long positions (and no investments at the risk free rate.)''')
    st.markdown('''**Uniform** rebalances each day to place equal value in each stock of the portfolio.''')


if __name__ == "__main__":
    main()