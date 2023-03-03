# Dataset based on #https://www.kaggle.com/datasets/kyanyoga/sample-sales-data
# USAGE: python main.py sales_data_sample.csv
# IF you want to query another dataset, just replace the csv file in the data folder and run the script again.

import os
import sys
import logging

import pandas as pd
import openai

import db_utils
import openai_utils

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
openai.api_key = os.environ["OPENAI_API_KEY"]

if __name__ == "__main__":
    logging.info("Loading data...")
    data_filename = sys.argv[1]
    df = pd.read_csv("data/" + data_filename)
    logging.info(f"Data Format: {df.shape}")

    logging.info("Converting to database...")
    tb_name = data_filename.split(".")[0].capitalize()
    database = db_utils.dataframe_to_database(df, table_name=tb_name)
    
    fixed_sql_prompt = openai_utils.create_table_definition_prompt(df, table_name=tb_name)
    logging.info(f"Fixed SQL Prompt: {fixed_sql_prompt}")

    while True:
        logging.info("Waiting for user input...")
        user_input = openai_utils.user_query_input()
        final_prompt = openai_utils.combine_prompts(fixed_sql_prompt, user_input)
        logging.info(f"Final Prompt: {final_prompt}")

        logging.info("Sending to OpenAI...")
        response = openai_utils.send_to_openai(final_prompt)
        proposed_query = response["choices"][0]["text"]
        proposed_query_postprocessed = db_utils.handle_response(response)
        logging.info(f"Response obtained. Proposed sql query: {proposed_query_postprocessed}")
        result = db_utils.execute_query(database, proposed_query_postprocessed, return_pandas=True)
        #logging.info(f"Result: {result}")
        print(result)

        response = input("Press enter to continue or type 'stop' to exit: ")
        if response.lower() == "stop":
            break
