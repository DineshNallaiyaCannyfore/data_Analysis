<!-- run -->

#run env
#run
import json
import tempfile

import httpx
import duckdb
import numpy as np
import fitz
import psutil
import os
import bs4
from typing import Any, Dict, List
from fastapi import UploadFile
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import re
from goose3 import Goose
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.duckdb import DuckDB
import requests
from bs4 import BeautifulSoup
# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2') 

# Set Gemini key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAto7z8ff4DnyXMZQHYiU9ubu0lsrgwt-o"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Extract text ---
async def extractText(files: List[UploadFile]) -> str:
    text = ""
    for f in files:
        suffix = f.filename.split(".")[-1].lower()
        if suffix == 'pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(await f.read())  
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            for page in doc:
                text += page.get_text()
        elif suffix == "txt":
            # Save uploaded file to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(await f.read())
                tmp_path = tmp.name

            # Now read from the saved txt
            with open(tmp_path, "r", encoding="utf-8") as c:
                text += c.read()
    return text

# --- Embed chunks into vector store ---

def createConnection():
   
    con = duckdb.connect("data.duckdb")
    con.execute("""
        CREATE SEQUENCE IF NOT EXISTS doc_id_seq;
        CREATE TABLE IF NOT EXISTS documents (
            id BIGINT PRIMARY KEY DEFAULT nextval('doc_id_seq'),
            text STRING,
            embedding TEXT
        )
        """)
    return con

def getURLS(text):
    
    # con = createConnection()
    # rows = con.execute("SELECT text FROM documents").fetchall()
    # print("rows",rows)
    # urls = []
    # for row in rows:
    #     urls.extend(re.findall(r'https?://\S+', row[0]))

    # return urls
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    return urls
    
async def embed_chunks(text: str):
    try:
        con = createConnection()
        print("con",con)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        chunks = text_splitter.split_text(text)
        print(f"Split text into {len(chunks)} chunks")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Embedding model loaded successfully")
        
        chunks_inserted = 0
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            try:
                # Use embed_query for single chunks, embed_documents for multiple
                vector = embedding_model.embed_query(chunk) 
                vector_serializable = [float(x) for x in vector] 
                
                con.execute(
                    "INSERT INTO documents (text, embedding) VALUES (?, ?)",
                    [chunk, vector_serializable] 
                )
                chunks_inserted += 1
                print(f"✓ Chunk {i+1} embedded and inserted")
                
            except Exception as chunk_error:
                print(f" Error processing chunk {i+1}: {chunk_error}")
                continue
        
        con.commit()
        con.close()
        print(f"Successfully inserted {chunks_inserted} chunks into database")
        
        return "**********"
        
    except Exception as e:
        print(f"Error in embed_chunks: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure connection is closed even on error
        try:
            con.close()
        except:
            pass
            
        return {
            "status": "error",
            "error": str(e)
        }

        
async def generate_questions(url,Query):
    text = await extract_text_from_urls(url)
    vector_store =  get_vector_store()
    print("vector_store",vector_store)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(Query)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""
    You are a helpful assistant.
    Based on the following context, answer the question:

    Context:
    {context}

    Question:
    {Query}
    """
    response = llm.invoke(prompt)
    print(response)
    # Final prompt with context
    # prompt = f"""
    # You are a helpful assistant.
    # Based on the following context, answer the question:

    # Context:
    # {text}

    # Question:
    # {query}
    # """
    # response = llm.invoke(prompt)
    # print(response)
    # return response.content
    
 #old code   
def extractTextFromURL(urls):
    con = createConnection()
    print(f"Processing URLs: {urls}")
    try:
        g = Goose()
        all_texts = []

        # Extract with Goose
        for url in urls:
            try:
                article = g.extract(url=url)
                if article.cleaned_text:
                    text = article.cleaned_text.strip()
                    all_texts.append(text)
                    print(f"Extracted {len(text)} characters from {url}")
                else:
                    print(f"No text extracted from {url}")
            except Exception as e:
                print(f"Error extracting {url}: {e}")

        # Combine all extracted text
        full_text = "\n".join(all_texts)
        if not full_text:
            print("No text extracted from any URLs")
            return ""

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_text(full_text)
        print(f"Split into {len(chunks)} chunks")

        if not chunks:
            print("No chunks generated after splitting")
            return full_text

        # Generate embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Generating embeddings...")
        vectors = embedding_model.embed_documents(chunks)
        # vector_store = InMemoryVectorStore.from_texts(chunks, embedding_model)
        # return vector_store
        print(f"Generated {len(vectors)} embeddings")

        # Prepare for DB insert
        data_to_insert = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            vector_serializable = [float(x) for x in vector]
            data_to_insert.append((chunk_text, vector_serializable))
            print(f"Prepared chunk {i+1}/{len(chunks)}")

        if not data_to_insert:
            print("No data to insert")
            return full_text

        print(f"Inserting {len(data_to_insert)} chunks into database...")
        con.executemany(
            "INSERT INTO documents (text, embedding) VALUES (?, ?)",
            data_to_insert
        )
        print("Data inserted successfully")

        return full_text

    except Exception as e:
        print(f"Error in extractTextFromURL: {e}")
        import traceback
        traceback.print_exc()
        return ""
    finally:
        con.close()

async def extract_text_from_urls(url_list):
    all_text = ""
    async with httpx.AsyncClient() as client:
        for url in url_list:
            response = await client.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            text = " ".join([p.get_text() for p in soup.find_all("p")])
            all_text += text + "\n"
    print(all_text)
    await embed_chunks(text)
    return all_text

def get_vector_store(db_path: str = "data.duckdb", table_name: str = "documents"):
    """
    Connect to DuckDB and return a LangChain-compatible vector store.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Synchronous connection
    con = duckdb.connect(db_path)

    vector_store = DuckDB(
        connection=con,
        embedding=embedding_model,
        table_name=table_name
    )

    return vector_store
# --- Handle file upload ---
async def handleFileUpload(files: List[UploadFile]) :
    cleanup_database()
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"Available RAM: {available_gb:.2f} GB")

    Query = await extractText(files)    
    print("Query",Query) 
    # vector = await embed_chunks(Query) 
    # print(vector)  
    urls = getURLS(Query)
    print("urls",urls)  
    if urls is None or len(urls) == 0:
        return {
            "status": "200",
            "error": "Urls is missing please add",
            "urls": urls
        }
    # extract_text_from_url(urls)
    extectText = await extract_text_from_urls(urls,)
    generated_qs =await  generate_questions(urls,Query)  

    return {
        "status": "200",
        "generated_questions": "j"
    }
    
def cleanup_database():
    """Clean up database files"""
    db_files = ["data.duckdb", "data.duckdb.wal", "data.duckdb.tmp"]
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                print(f"Deleted: {db_file}")
            except:
                pass
async def generate_questions(url,Query):
    # text = await extract_text_from_urls(url)
    vector_store =  get_vector_store()
    print("vector_store",vector_store)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(Query)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""
    You are a helpful assistant.
    Based on the following context, answer the question:

    Context:
    {context}

    Question:
    {Query}
    """
    response = llm.invoke(prompt)
    print(response)
    # Final prompt with context
    # prompt = f"""
    # You are a helpful assistant.
    # Based on the following context, answer the question:

    # Context:
    # {text}

    # Question:
    # {query}
    # """
    # response = llm.invoke(prompt)
    # print(response)
    # return response.content