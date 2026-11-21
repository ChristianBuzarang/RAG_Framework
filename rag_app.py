import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

MODEL_NAME = "llama3.1"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PDF_PATH = "./data/Pushdown Automata.pdf"
PERSIST_DIR = "./chroma_db"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    # 1. Load and Split
    loader = PyPDFLoader(PDF_PATH)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(loader.load())

    # 2. Vector Store
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3. RAG Pipeline
    llm = OllamaLLM(model=MODEL_NAME)
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:

{context}

Question: {question}"""
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "Summarize this document and highlight its main technical concepts."
    result = rag_chain.invoke(question)
    print("\n--- RAG Response ---")
    print(result)


if __name__ == "__main__":
    main()
