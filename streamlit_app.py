import os
import openai
import pandas as pd
import streamlit as st

import db_utils
import openai_utils

st.title("Faça Consultas SQL em Português")
openai.api_key = os.environ["OPENAI_API_KEY"]

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
        database = db_utils.dataframe_to_database(df, table_name=tb_name)
        fixed_sql_prompt = openai_utils.create_table_definition_prompt(df, table_name=tb_name)
        final_prompt = openai_utils.combine_prompts(fixed_sql_prompt, user_input)
        response = openai_utils.send_to_openai(final_prompt)
        proposed_query = response["choices"][0]["text"]
        proposed_query_postprocessed = db_utils.handle_response(response)
        result = db_utils.execute_query(database, proposed_query_postprocessed, return_pandas=True)
        st.write(result)


