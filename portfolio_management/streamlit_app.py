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
                             mode='lines',name='DPM'))
    fig.add_trace(go.Scatter(x=date_range,
                             y=agents.uniform_vals,
                             mode='lines',name='uniform'))
    fig.add_trace(go.Scatter(x=date_range,
                             y=agents.capm_vals,
                             mode='lines',name='CAPM'))
    fig.add_trace(go.Scatter(x=date_range,
                             y=agents.mvp_vals,
                             mode='lines',name='MVP'))
                             
    return fig,date_range

def main():
    st.title('Portfolio algorithms comparison.')
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
 
    agents = AgentComparison(symbols,start,end)
    
    fig,date_range = plot_histories(agents) 
    d_init = date_range.iloc[0].date().isoformat()
    d_final = date_range.iloc[-1].date().isoformat()
    #print(type(d_init))
    st.write('''Each agent uses data from previous 64 trading days
               to make trading decisions for the following day.
               Note there are approximately 253 trading days in a year.''')
    st.write(f'First trading day: {d_init}. Final trading day: {d_final}')
    st.write(fig)





if __name__ == "__main__":
    main()