import openai
import streamlit as st

# Set up your OpenAI API key
# Note: Directly setting the API key in code is not recommended for production
# Instead, consider using environment variables or a secure vault
openai.api_key = 'your-api-key-here'

# Function to get a response from GPT-4
def get_gpt4_response(prompt):
    response = openai.Completion.create(
        model="gpt-4",  # Use "gpt-4" or the specific model name you have access to
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
