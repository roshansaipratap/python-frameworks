import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import os
import truststore
truststore.inject_into_ssl()


st.title("Agent-CSV")
st.write("An agent Upload a CSV file and ask questions about the data.")

uploaded_files = st.file_uploader("Upload a CSV file", type="csv", accept_multiple_files=True)

if uploaded_files is not None:
    dfs = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        dfs.append(df)
        st.write(f"### Data Preview for {uploaded_file.name}")
        st.dataframe(df.head())

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    system_prompt = """
    You are a smart data assistant capable of reading multiple CSV files.
    - You have access to 4 different datasets: SaaS Docs, Credit Card Terms, Hospital Policy, and Ecommerce FAQs.
    - When asked a question, determine which DataFrame is most relevant.
    - Do NOT answer from general knowledge.
    - Answer in plain English.
    """

    agent = create_pandas_dataframe_agent(
        llm, dfs, verbose=True, allow_dangerous_code=True, prefix=system_prompt, agent_type="tool-calling")

    st.write("### Ask a question about the data")
    user_question = st.text_input("Enter a question:")
    if user_question:
        response = agent.run(user_question)
        st.write("### Response")
        st.write(response)
    else:
        st.write("Please enter a question to get a response.")

    