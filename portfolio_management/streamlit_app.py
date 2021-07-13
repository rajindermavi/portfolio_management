import streamlit as st
import datetime 
import plotly.graph_objects as go
from model import MetaAgent, MetaEnv
from agents import agent_loss
from data import GetYFArchive

#global yf_archive
yf_archive = None

# Simulations are restricted to 2017 and on.
# We will build the archive for this timespan.
CAL_START = datetime.date(2019,1,1)
CAL_END = datetime.date.today()
# Initialize with suggested simulation date range
range_start = datetime.date(2020,1,1)
range_end = datetime.date(2020,12,31)
# List of symbols our bot is able to trade    
ALL_SYMBOLS = ["AAPL","ABMD","ABT","ALB","ALK","AMZN","ATVI",
                "ADBE","AFL","AMD","BMY","BSX","CB","CVX","DIS",
                "GE","GS","HAS","HD","HON","HPQ","HUM","IBM","INTC",
                "JNJ","LOW","MMM","MSFT","NEE","NKE",
                "NVDA","PEP","PFE","SBUX","WFC","WMT"]  
# Initialize with list of suggested symbols
select_symbols = ["AMD","BSX","GE","MMM","WMT"]

# get_yf_archive is always called with same arguments (CAL_START,CAL_END,ALL_SYMBOLS)
@st.cache()
def get_yf_archive(start,end,symbols):
    '''Returns pandas DF with columns 
    Date,Symbol,high,low,open,close,volume,adjclose.'''
    yf=GetYFArchive()
    return yf.get(start,end,symbols,'daily')

# reduce yf_data to needed data for an agent simulation and create environment
def get_env(range_start,range_end,select_symbols):
    date_mask = (yf_archive['Date'] >= range_start)&(yf_archive['Date'] <= range_end)  
    meta_env = MetaEnv(yf_archive,date_mask,select_symbols)
    return meta_env.env, meta_env.trading_dates

def get_meta_agent(n_stocks):
    return MetaAgent(n_stocks)
    
def plot_agent_histories(sim_agent_names,simulations,sim_trading_dates):
     
     
    sim_fig = go.Figure()
    for agent_name,sim,dates in list(zip(sim_agent_names,simulations,sim_trading_dates)):

        sim_fig.add_trace(go.Scatter(x=dates,y=sim,
                    mode='lines',name=agent_name))

    sim_fig.update_layout(title='Portfolio values over time')
    return sim_fig

def plot_final_weights(weights_list,sim_agent_names,select_symbols):

    select_symbols.append('Risk free')
    zipped = list(zip(weights_list,sim_agent_names))

    final_weights_fig = go.Figure()

    for weights,agent_name in zipped: 
        final_weights_fig.add_trace(go.Bar(x=select_symbols,y=weights[0],name=agent_name))
   
    final_weights_fig.update_layout(title='Final weights of the portfolios',
                                    barmode='group', 
                                    xaxis_tickangle=-45)

    return final_weights_fig


@st.cache()
def agent_simulation(agent_name,range_start,range_end,select_symbols):
    
    env,trading_dates = get_env(range_start,range_end,select_symbols)
    
    meta_agent = get_meta_agent(len(select_symbols))

    _loss = agent_loss(env,
                       meta_agent.agent_dic[agent_name],
                       len(select_symbols))
    return env.portfolio_value_hist,trading_dates[env._start_tick:],env._portfolio_weights



def main():
    '''Sets up main splash page'''
    # Top Information
    st.title('Portfolio Management')
    st.write('''Below four agents using different trading algorithms (DeepPortfolio, CAPM, MVP, Uniform) 
                will simulate trading activity based on the chosen collection of stocks and date range.''')
    st.write('''Each agent begins with 1M in cash and
                    rebalances its portfolio each day using
                    data from previous 64 trading days.''')
    
    # form box for interactive component
    display = st.form("display")
    
    # Bottom Information
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
                   The agent then selects the portfolio minimizing the risk (variance) $w^T\Sigma w$ constrained to keep long positions (and no investments at the risk free rate).''')
    st.markdown('''**Uniform** rebalances each day to place equal value in each stock of the portfolio.''')

    # Create interactive components
    global range_start
    global range_end
    cal_dates=display.slider("Select a calendar date range", CAL_START, CAL_END, 
                    (range_start, range_end))
    range_start=cal_dates[0]
    range_end=cal_dates[1]

    global select_symbols
    select_symbols = display.multiselect(
                 'Select symbols for your portfolio.',
                  ALL_SYMBOLS,
                  select_symbols) 

    sim_agent_names = display.multiselect(
                      'Select your agents.',
                      ['DeepPortfolio','CAPM','MVP','Uniform'],
                      ['DeepPortfolio','Uniform'])

    run=display.form_submit_button("Run")



    # Check if passed parameters are null
    check_n_symbols = len(select_symbols) == 0
    check_n_agents = len(sim_agent_names) == 0 
    check_date_range = range_start + datetime.timedelta(days=90) >= range_end

    if run:
        # If passed parameters are null, request new parameters.
        if  check_n_symbols or check_n_agents or check_date_range: 
            display.write('''Please select a calendar range of at least 90 days,
                          with at least one symbol, and at least one agent.''')
        else:
            simulations = [] 
            sims_trading_dates = []
            final_weights_list = []
            for agent_name in sim_agent_names:
                    
                with st.spinner('...simulating '+agent_name+'...'):
                    
                    simulation,trading_dates,weights=agent_simulation(agent_name,
                                        range_start.isoformat(),
                                        range_end.isoformat(),
                                        select_symbols)
                simulations.append(simulation)
                sims_trading_dates.append(trading_dates)
                final_weights_list.append(weights)

            d_init = trading_dates.iloc[0].date().isoformat()
            d_final = trading_dates.iloc[-1].date().isoformat()
        

        sim_fig = plot_agent_histories(sim_agent_names,simulations,sims_trading_dates)

        display.write(f'First trading day: {d_init}. Final trading day: {d_final}')
        display.write(sim_fig)

        final_weights_fig = plot_final_weights(final_weights_list,sim_agent_names,select_symbols)
        display.write(final_weights_fig)

if __name__ == "__main__":
    # Get data from Yahoo Finance
    with st.spinner('...fetching data...'):
        yf_archive=get_yf_archive(CAL_START.isoformat(),
                                  CAL_END.isoformat(),
                                  ALL_SYMBOLS)
    main()