import streamlit as st
import pandas as pd
import sqlite3
import json
from openai import OpenAI
import os
import chardet
import io
import re
from collections import Counter

# OpenAI API setup
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Constants
DB_NAME = 'data.db'
FIXED_TABLE_NAME = "uploaded_data"
############################################### HELPER FUNCTIONS ########################################################
def detect_encoding(file_path):                           # Detects the encoding of a file
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']

def get_data_type(values):                                # Determines the data type of a column
    if all(isinstance(val, (int, float)) for val in values if pd.notna(val)):
        return "Number"
    elif all(isinstance(val, str) for val in values if pd.notna(val)):
        return "Text"
    elif all(pd.to_datetime(val, errors='coerce') is not pd.NaT for val in values if pd.notna(val)):
        return "Date"
    else:
        return "Mixed"

def analyze_csv(file_path, max_examples=3):                # Analyzes the CSV file and generates a description
    encoding = detect_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    
    prompt = "This CSV file contains the following columns:\n\n"
    
    for col in df.columns:
        values = df[col].dropna().tolist()
        data_type = get_data_type(values)
        
        unique_count = df[col].nunique()
        total_count = len(df)
        is_unique = unique_count == total_count
        
        examples = df[col].dropna().sample(min(max_examples, len(values))).tolist()
        
        prompt += f"Column: {col}\n"
        prompt += f"Data Type: {data_type}\n"
        prompt += f"Examples: {', '.join(map(str, examples))}\n"
        
        if is_unique:
            prompt += "Note: This column contains unique values for each row.\n"
        
        null_count = df[col].isnull().sum()
        if null_count > 0:
            prompt += f"Note: This column contains {null_count} NULL values.\n"
        
        if data_type == "Text":
            value_counts = Counter(values)
            most_common = value_counts.most_common(3)
            if len(most_common) < len(value_counts):
                prompt += f"Most common values: {', '.join(f'{val} ({count})' for val, count in most_common)}\n"
        
        prompt += "\n"
    
    return prompt

def reset_chat():
    st.session_state.messages = []

def display_sql_query(query):
    with st.expander("View SQL Query", expanded=False):
        st.code(query, language="sql")

def display_json_data(json_data):
    with st.expander("View JSON Data", expanded=False):
        st.json(json_data)

def df_to_sqlite(df, table_name, db_name=DB_NAME): #The purpose of this function is to save a Pandas DataFrame to an SQLite database with error handling
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"An error occurred while creating the table: {e}")
        return False

############################################## AI INTERACTION FUNCTIONS ######################################################

def generate_sql_query(user_input, prompt, chat_history):
    messages = [
        {"role": "system", "content": prompt},
    ]
    
    # Add chat history
    for message in chat_history:
        messages.append({"role": message["role"], "content": message["content"]})
    
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def execute_query_and_save_json(input_string, table_name, db_name=DB_NAME):
    try:
        sql_data = json.loads(input_string)
        sql_query = sql_data["SQL"]
    except json.JSONDecodeError:
        st.error("Failed to parse SQL query response.")
        return None

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        result = cursor.fetchone()
    except sqlite3.Error as e:
        st.error(f"An error occurred while executing the query: {e}")
        return None
    finally:
        conn.close()
    
    column_name = sql_query.split('AS ')[-1].split(' FROM')[0].strip() if ' AS ' in sql_query else 'result'
    result_dict = {column_name: result[0] if result else None}
    
    with open('query_result.json', 'w') as json_file:
        json.dump(result_dict, json_file, indent=2)
    
    return result_dict

def generate_response(json_data, prompt, chat_history):
    messages = [
        {"role": "system", "content": prompt},
    ]
    
    # Add chat history
    for message in chat_history:
        messages.append({"role": message["role"], "content": message["content"]})
    
    messages.append({"role": "user", "content": f"JSON data: {json_data}"})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

@st.cache_data
def load_data(uploaded_file):
    try:
        # Save the uploaded file temporarily
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Analyze the CSV
        csv_analysis = analyze_csv("temp.csv")
        
        # Read the CSV with the detected encoding
        encoding = detect_encoding("temp.csv")
        df = pd.read_csv("temp.csv", encoding=encoding)
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        return None, None, None

    if not df_to_sqlite(df, FIXED_TABLE_NAME):
        return None, None, None
    
    return df, FIXED_TABLE_NAME, csv_analysis

def main():
    st.set_page_config(layout="wide", page_title="DataChat", page_icon="📈")
    st.title("Data Chat Application")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    df = None
    csv_explanation = ""

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df, table_name, csv_analysis = load_data(uploaded_file)
        if df is not None:
            # Append the analysis to the explanation text area
            csv_explanation = st.text_area("Please enter an explanation for your CSV data:", 
                                           value=csv_analysis,
                                           height=300)
            if st.button("Submit Explanation"):
                st.success("Explanation submitted successfully!")
        else:
            st.warning("Failed to load the CSV file. Please try again.")
            return
    else:
        st.warning("Please upload a CSV file.")
        return

    if df is not None:
        st.sidebar.success("Data loaded successfully!")
        
        st.sidebar.subheader("Data Preview")
        st.sidebar.dataframe(df.head())

        st.header("Chat with your data")

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        prompt = st.chat_input("What would you like to know about the data?")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Generating response..."):
                sql_generation_prompt = f'''
                Table name: {table_name}
                Columns: {', '.join([col for col in df.columns])}
                {csv_analysis}

                User's explanation of the CSV:
                {csv_explanation}

                A user will now chat with you. Your task is to transform the user's request into an SQL query that retrieves exactly what they are asking for.

                Rules:
                1. Return only two JSON variables: "Explanation" and "SQL".
                2. No matter how complex the user question is, return only one SQL query.
                3. Always return the SQL query in a one-line format.
                4. Consider the chat history when generating the SQL query.

                Example output:
                {{
                "Explanation": "The user is asking about the number of users. To retrieve this, we need to count all rows in the table.",
                "SQL": "SELECT COUNT(*) AS User_Count FROM {table_name}"
                }}

                Your prompt ends here. Everything after this is the chat with the user. Remember to always return the accurate SQL query.
                '''
                sql_query_response = generate_sql_query(prompt, sql_generation_prompt, st.session_state.messages[:-1])
                
                try:
                    sql_data = json.loads(sql_query_response)
                    sql_query = sql_data["SQL"]
                    display_sql_query(sql_query)
                    
                    result_dict = execute_query_and_save_json(sql_query_response, table_name)

                    if result_dict:
                        display_json_data(result_dict)
                        
                        response_generation_prompt = f'''
                        Table name: {table_name}
                        Columns: {', '.join([col for col in df.columns])}
                        {csv_analysis}

                        User's explanation of the CSV:
                        {csv_explanation}

                        Now you will receive a JSON containing the SQL output that answers the user's inquiry. Your task is to use the SQL's output to answer the user's inquiry in plain English. Consider the chat history when generating your response.
                        '''
                        response = generate_response(json.dumps(result_dict), response_generation_prompt, st.session_state.messages)

                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        with st.chat_message("assistant"):
                            st.markdown(response)
                    else:
                        st.error("Failed to execute the SQL query. Please try rephrasing your question.")
                except json.JSONDecodeError:
                    st.error("Failed to generate a valid SQL query. Please try rephrasing your question.")
           
        st.sidebar.button("Reset Chat", on_click=reset_chat)

if __name__ == "__main__":
    main()
