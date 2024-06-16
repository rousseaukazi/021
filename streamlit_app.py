import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI

client = OpenAI(
  organization='org-WOpUWQvB82kuQqU0GO5bX6Nu',
  project='proj_EXw0srD1epiNHMuCTdGF64Ft',
)

def get_gpt4_response(prompt):
  response = openai.Completion.create
  (
    model="text-davinci-004",  # Ensure you have access to the GPT-4 model
    prompt=prompt,
    max_tokens=150  # Adjust as needed
  ) 
  return response.choices[0].text.strip()

idea = st.text_input("What's your idea?")
st.write("Your idea is ", get_gpt4_response(idea))
