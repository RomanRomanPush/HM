import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pydeck as pdk
import pandas as pd
import numpy as np
import plotly.express as px
import numpy as np
import plotly
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json


# Для запуска в терминале введите в терминале -> streamlit run app.py


st.set_option('deprecation.showPyplotGlobalUse', False)

path = "data/housing.csv"

def load_data(path):
    df = pd.read_csv(path)
    return df

def plot_hist(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_rows = len(num_cols) // 2 
    if len(num_cols) % 2: 
        num_rows += 1 

    plt.figure(figsize=[15, num_rows*5])
    for index, col in enumerate(num_cols):
        plt.subplot(num_rows, 2, index+1)
        df[col].plot(kind='hist', title=col)
    plt.tight_layout()
    st.pyplot()

def show_scatterplot(df):
    df_filled = df.fillna(0.0)
    # Define tooltip
    tooltip = {
        "html": "<b>Median House Value:</b> {median_house_value} <br/> <b>Population:</b> {population}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    scatterplot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_filled,
        get_position=['longitude', 'latitude'],
        get_color='[200, 30, 0, 160]',
        get_radius="population",
        radius_scale=0.1,
        pickable=True,   # Enables tooltip
        auto_highlight=True,   # Highlight the point when the mouse is over it
        tooltip=tooltip    # Define what shows up in the tooltip
    )

    view_state = pdk.ViewState(latitude=37.7749295, longitude=-122.4194155, zoom=6, bearing=0, pitch=0)
    st.pydeck_chart(pdk.Deck(layers=[scatterplot_layer], initial_view_state=view_state))

def show_heatmap(df):
    df = df.fillna(0.0)
    st.subheader("Heatmap: Median House Value")
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        opacity=0.9,
        get_position=['longitude', 'latitude'],
        get_weight="median_house_value",
        threshold=0.3,
        aggregation='"MEAN"',
    )
    view_state = pdk.ViewState(latitude=37.7749295, longitude=-122.4194155, zoom=6, bearing=0, pitch=0)
    st.pydeck_chart(pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state))


def main():
    df = load_data(path)

    st.sidebar.title("California Housing Dataset Analysis")
    st.sidebar.markdown("This app allows you to explore the California Housing dataset.")
    
    st.title("California Housing Dataset")

    if st.sidebar.button("Show DataFrame"):
        st.dataframe(df)

    if st.sidebar.button("Show Histograms"):
        plot_hist(df)
    
    if st.sidebar.button("Show Scatterplot"):
        show_scatterplot(df)

    if st.sidebar.button("Show Heatmap"):
        show_heatmap(df)

if __name__ == "__main__":
    main()