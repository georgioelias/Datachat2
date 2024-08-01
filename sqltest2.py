import streamlit as st
import pandas as pd
import sqlite3
import json
from openai import OpenAI
import os
import chardet
import io
import re

# OpenAI API setup
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Constants
DB_NAME = 'data.db'
FIXED_TABLE_NAME = "uploaded_data"

def reset_chat():
    st.session_state.messages = []

def display_sql_query(query):
    with st.expander("View SQL Query", expanded=False):
        st.code(query, language="sql")

def get_table_schema(table_name, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    schema = [{"name": column[1], "type": column[2]} for column in columns]
    conn.close()
    return schema

def sanitize_table_name(name):
    sanitized = re.sub(r'\W+', '_', name)
    if sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized.lower()

def df_to_sqlite(df, table_name, db_name=DB_NAME):
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"An error occurred while creating the table: {e}")
        return False

def table_exists(table_name, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def execute_sql_to_json(query, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    try:
        df = pd.read_sql_query(query, conn)
        return df.to_json(orient='records')
    except sqlite3.Error as e:
        st.error(f"An error occurred while executing the query: {e}")
        return None
    finally:
        conn.close()

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
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        try:
            df = pd.read_csv(io.StringIO(raw_data.decode(encoding)))
        except UnicodeDecodeError:
            encodings_to_try = ['iso-8859-1', 'cp1252']
            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(io.StringIO(raw_data.decode(enc)))
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error(f"Unable to read the CSV file. Please check the file encoding.")
                return None, None

    if not df_to_sqlite(df, FIXED_TABLE_NAME):
        return None, None
    
    return df, FIXED_TABLE_NAME

def main():
    st.set_page_config(layout="wide", page_title="DataChat", page_icon="📈")
    st.title("Data Chat Application")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    df = None
    csv_explanation = ""

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df, table_name = load_data(uploaded_file)
        if df is not None:
            csv_explanation = st.text_area("Please enter an explanation for your CSV data:", 
                                           placeholder="Enter a detailed explanation of your CSV file structure here...")
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
                        response_generation_prompt = f'''
                        Table name: {table_name}
                        Columns: {', '.join([col for col in df.columns])}
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
