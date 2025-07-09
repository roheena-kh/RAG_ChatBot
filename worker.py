import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp

# Initialize global vars
conversation_chain = None
chat_history = []

def init_llm():
    global llm, embeddings
    print("Initializing model and embeddings...")

    # Load local GGUF model
    llm_path = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    llm = LlamaCpp(
        model_path=llm_path,
        temperature=0.1,
        max_tokens=512,
        n_ctx=2048,
        verbose=True
    )

    embeddings = SentenceTransformer("all-MiniLM-L6-v2")

def process_document(file_path):
    global conversation_chain

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(docs, embedding=embeddings)

    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5}),
        return_source_documents=False
    )

def process_prompt(prompt):
    global chat_history

    if conversation_chain is None:
        return "Please upload a PDF document first."

    result = conversation_chain.run(prompt)
    chat_history.append((prompt, result))
    return result

# Initialize everything
init_llm()
