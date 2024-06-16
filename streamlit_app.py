import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

title = st.text_input("Movie title", "Life of Brian")
st.write("The current movie title is", title)
