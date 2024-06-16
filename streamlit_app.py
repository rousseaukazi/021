import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

idea = st.text_input("What's your idea?")
st.write("Your idea is ", get_gpt4_response(idea))
