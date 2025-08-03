# HarshAI – Streamlit app to generate personalized offer letters using HR PDFs and employee CSV

import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load employee list
EMPLOYEE_CSV_PATH = "Employee_List.csv"
PDF_FILES = [
    "HR Offer Letter.pdf",
    "HR Travel Policy.pdf",
    "HR Leave Policy.pdf"
]


def load_employee_data():
    df = pd.read_csv(EMPLOYEE_CSV_PATH)
    df.columns = df.columns.str.strip()
    return df.set_index("Employee Name")


def get_pdf_text(filepaths):
    text = ""
    for path in filepaths:
        with open(path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)


def build_vector_store():
    raw_text = get_pdf_text(PDF_FILES)
    chunks = get_text_chunks(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    You are HarshAI, an HR Officer. Use the given context to generate a complete offer letter for the employee below.
    If Band-specific compensation is missing, interpolate using nearby Bands or use policy rules from Travel & Leave PDFs.

    Context:
    {context}

    Question:
    Generate a professional offer letter for {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)


def query_offer_letter(employee_name, employee_df):
    if employee_name not in employee_df.index:
        return f"❌ Employee '{employee_name}' not found in the organization."

    row = employee_df.loc[employee_name]
    band = row.get("Band", "Unknown")
    dept = row.get("Department", "Unknown")
    question = f"Generate a full offer letter for employee {employee_name}, who is in Band {band} and Department {dept}."

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]


def main():
    st.set_page_config(page_title="HarshAI Offer Letter Generator", page_icon="📄")
    st.title("📄 HarshAI: Personalized Offer Letter Generator")

    if not os.path.exists("faiss_index"):
        with st.spinner("Processing HR documents and building vector DB..."):
            build_vector_store()
            st.success("Vector store ready.")

    employee_df = load_employee_data()

    st.markdown("---")
    st.subheader("Enter Employee Name")
    name_input = st.text_input("Employee Name")

    if st.button("Generate Offer Letter") and name_input.strip():
        with st.spinner("Generating offer letter..."):
            result = query_offer_letter(name_input.strip(), employee_df)
            st.markdown("---")
            st.subheader("📄 Offer Letter")
            st.markdown(result)


if __name__ == "__main__":
    main()
