from openai import OpenAI
import streamlit as st
import os

client = OpenAI(
  organization='org-WOpUWQvB82kuQqU0GO5bX6Nu',
  project='proj_EXw0srD1epiNHMuCTdGF64Ft',
)


prompt = st.text_input("What's your idea?")
