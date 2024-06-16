import openai
import streamlit as st
import os

# Set the OpenAI API key from environment variables
openai.api_key = os.getenv("sk-proj-Mqh0dXdwdd20ucNt10NIT3BlbkFJ9v5xl2AIhj3PXAyupBSx")

# Function to get a response from GPT-4
def get_gpt4_response(prompt):
    response = openai.Completion.create(
        model="gpt-4",  # Ensure you have access to the GPT-4 model
        prompt=prompt,
        max_tokens=150  # Adjust as needed
    )
    return response.choices[0].text.strip()

# Streamlit app
st.title("ChatGPT Interaction with Streamlit")
st.write("Enter your prompt and get a response from GPT-4:")

prompt = st.text_input("What's your idea?")
if prompt:
    response = get_gpt4_response(prompt)
    st.write("Response from GPT-4:")
    st.write(response)
