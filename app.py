import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import OpenAI

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Sidebar: Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# FAISS storage path
FAISS_INDEX_PATH = "faiss_store_openai"

main_placeholder = st.empty()

# Initialize LLM
llm = OpenAI(
    temperature=0.9,
    max_tokens=500,
    openai_api_key=os.getenv("OPENAI_API_KEY")  # ✅ explicitly provide key
)

if process_url_clicked:
    # 1. Load data
    loader = UnstructuredURLLoader(urls=[url for url in urls if url.strip() != ""])
    main_placeholder.text("🔄 Loading data from URLs...")
    documents = loader.load()

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("🔄 Splitting text into chunks...")
    docs = text_splitter.split_documents(documents)

    # 3. Embedding and vector indexing
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    main_placeholder.text("🔄 Generating embeddings...")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    # 4. Save FAISS vectorstore safely (no pickle)
    vectorstore_openai.save_local(FAISS_INDEX_PATH)
    main_placeholder.success("✅ FAISS Index built and saved!")

# Question-Answer UI
query = main_placeholder.text_input("Ask a question based on the articles:")

if query:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        result = chain({"question": query}, return_only_outputs=True)

        st.header("📌 Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("📎 Sources:")
            for source in result["sources"].split("\n"):
                st.write(source)

    except Exception as e:
        st.error(f"Error: {str(e)}")
