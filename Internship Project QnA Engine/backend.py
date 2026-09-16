import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

app = FastAPI(title="RAG API Backend")

# Enable CORS so your local frontend can talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set environment keys
os.environ["GROQ_API_KEY"] = ""

# Pydantic Schemas for API Requests
class ContextUpdate(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str

# Global instances initialized on start
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = None
rag_chain = None

def rebuild_rag_chain():
    """Helper to initialize or refresh the LangChain retriever components"""
    global vector_store, rag_chain
    if not os.path.exists("sample.txt"):
        with open("sample.txt", "w", encoding="utf-8") as f:
            f.write("Initial default context placeholder.")

    loader = TextLoader("sample.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question using only the provided context.\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Build the initial chain on server launch
rebuild_rag_chain()

@app.post("/update-context")
async def update_context(data: ContextUpdate):
    try:
        with open("sample.txt", "w", encoding="utf-8") as f:
            f.write(data.text)
        
        # Rebuild vector db with the new context text
        rebuild_rag_chain()
        return {"status": "success", "message": "Vector store updated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(data: QueryRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=400, detail="RAG system is not initialized.")
    try:
        response = rag_chain.invoke({"input": data.question})
        
        # Extract sources to pass back to the frontend
        sources = [doc.page_content for doc in response.get("context", [])]
        
        return {
            "answer": response["answer"],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import uvicorn
if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)