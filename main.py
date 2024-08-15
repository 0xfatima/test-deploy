from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
#importing
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, WebBaseLoader, YoutubeLoader, DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import tiktoken
from dotenv import load_dotenv
import json
from typing import List  # Make sure to import List from typing
from pydantic import BaseModel
app = FastAPI()

class QueryRequest(BaseModel):
    query: List[float]  # Expecting a list of floats as the vector
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify the domains you want to allow
    allow_credentials=True,
    allow_methods=["*"],  # or specify the methods you want to allow
    allow_headers=["*"],  # or specify the headers you want to allow
)

#loading video from youtube
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=e-gwvmhyU7A", add_video_info=True)
# Usage
data = loader.load() 


# making chunks of data got from youtube video
tokenizer = tiktoken.get_encoding('p50k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

#defining text_splitter to make chunks
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
)

# splitting data from youtube video using text splitter and tokenizer
texts = text_splitter.split_documents(data)

#defining hugging face embedding variable
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#initializing vector database "Chroma db"
vector_store = Chroma(
    collection_name="data_collection",
    embedding_function=hf_embeddings,
)

#storing data into documents variable to add to chroma db (vector database)

documents= [
    Document(
        page_content=f"Source: {t.metadata['source']}, Title: {t.metadata['title']} \n\nContent: {t.page_content}",
                   metadata=t.metadata
                   )
    for t in texts]

#adding to database
vectorstore_from_texts = vector_store.add_documents(documents=documents)



@app.post("/getData")

def get_internal_data(req:QueryRequest):
    try:
        # Performing similarity search by vector
        results = vector_store.similarity_search_by_vector(
            embedding=req.query, k=1
        )
        
        # Providing the context to LLM by performing similarity vector search
        contexts = [doc.page_content for doc in results]
        return {'response': contexts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    