import os
import openai
import pandas as pd
import streamlit as st

from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent


st.title("Faça Consultas SQL em Português")
openai.api_key = os.environ["OPENAI_API_KEY"]

st.info("Certifique-se de que o nome das colunas não contenha espaços ou caracteres especiais.")
uploaded_file = st.file_uploader("Faça upload do arquivo csv", type=".csv", accept_multiple_files=False)

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

st.subheader("O que você deseja saber sobre os dados?")

with st.form("query_form"):
   user_input = st.text_input("Pergunta", value="Quantas pessoas sobreviveram?")

   submitted = st.form_submit_button("Submit")
   if submitted:
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)       
        result = agent.run(user_input)
        st.write(result)
