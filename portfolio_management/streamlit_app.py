import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from model import AgentComparison

def plot_histories(agents):
    start = agents.meta_env.env._start_tick
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agents.meta_env.dates[start:],
                             y=agents.dpm_vals,
                             mode='lines',name='DPM'))
    fig.add_trace(go.Scatter(x=agents.meta_env.dates[start:],
                             y=agents.uniform_vals,
                             mode='lines',name='uniform'))
    fig.add_trace(go.Scatter(x=agents.meta_env.dates[start:],
                             y=agents.capm_vals,
                             mode='lines',name='CAPM'))
    fig.add_trace(go.Scatter(x=agents.meta_env.dates[start:],
                             y=agents.mvp_vals,
                             mode='lines',name='MVP'))
                             
    return fig

def main():
    st.title('Portfolio algorithms comparison.')
    agents = AgentComparison(["ADBE","AFL","AMD","BMY","BSX",'AAPL','MMM'])
    
    fig = plot_histories(agents)

    st.write(fig)





if __name__ == "__main__":
    main()