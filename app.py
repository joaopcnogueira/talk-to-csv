import os
import openai
import pandas as pd
import streamlit as st

from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent


st.title("Talk to CSV")

user_openai_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password"
)

os.environ["OPENAI_API_KEY"] = user_openai_api_key

st.info("Make sure the name of the columns does not have spaces or special characters.")
uploaded_file = st.file_uploader("Upload your file here", type=".csv", accept_multiple_files=False)

if uploaded_file is None:
    st.info("Using example data. Upload a file above to use your own data!")
    uploaded_file = open("./data/titanic.csv", "r")
    tb_name = "Titanic"
    df = pd.read_csv(uploaded_file)
    with st.expander("Example data"):
        st.write(df)
else:
    st.success("Uploaded your file!")
    df = pd.read_csv(uploaded_file)
    tb_name = uploaded_file.name.split(".")[0].capitalize()
    with st.expander("Uploaded data"):
        st.write(df)

st.subheader("Ask a question about your data!")

with st.form("query_form"):
   user_input = st.text_input("Pergunta", value="Quantas pessoas sobreviveram?")

   submitted = st.form_submit_button("Submit")
   if submitted:
        try:
            openai.api_key = user_openai_api_key
            agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)       
            result = agent.run(user_input)
            st.write(result)
        except Exception as e:
            st.error("Please, provide your OpenAI API key above.")
