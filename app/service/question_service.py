import json
import tempfile
import httpx
import duckdb
import numpy as np
import fitz
import pandas as pd
import docx
import fitz 
import asyncio
import os
import sentence_transformers
from typing import Any, Dict, List
from fastapi import File, UploadFile
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
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.duckdb import DuckDB

from langchain.schema import Document 
from bs4 import BeautifulSoup
model = SentenceTransformer('all-MiniLM-L6-v2') 
print(sentence_transformers.__version__)
# Set Gemini key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAto7z8ff4DnyXMZQHYiU9ubu0lsrgwt-o"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
CON = duckdb.connect(database=":memory:")
VECTOR_STORE = None

async def extractText(files: List[UploadFile]) -> str:
    global VECTOR_STORE
    query_text = ""
    file_text = ""
    for f in files:
        suffix = f.filename.split(".")[-1].lower()
        if suffix == "txt":
            file_text = (await f.read()).decode("utf-8")
            query_text += file_text

        elif suffix == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(await f.read())
                tmp_path = tmp.name
            import fitz
            doc = fitz.open(tmp_path)
            file_text = "".join([page.get_text() for page in doc])

        elif suffix in ["csv"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(await f.read())
                tmp_path = tmp.name
            df = pd.read_csv(tmp_path)
            file_text = df.to_string()

        elif suffix in ["xls", "xlsx"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await f.read())
                tmp_path = tmp.name
            df = pd.read_excel(tmp_path)
            file_text = df.to_string()

        elif suffix in ["docx", "doc"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await f.read())
                tmp_path = tmp.name
            doc = docx.Document(tmp_path)
            file_text = "\n".join([para.text for para in doc.paragraphs])

        if suffix != "txt" :
             print("DDD", file_text)
             await embed_chunks_in_memory_duckdb(file_text, f.filename, suffix)

    return {
            "query": query_text,
            "fileTypeList": file_text
        }
    

def getURLS(text):
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    return urls
    
async def extract_text_from_urls(url_list):
    if url_list:  # checks if not empty
        all_text = ""
        async with httpx.AsyncClient() as client:
            for url in url_list:
                response = await client.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                text = " ".join([p.get_text() for p in soup.find_all("p")])
                all_text += text + "\n"
        print("all_text =", all_text)
        await embed_chunks_in_memory_duckdb(all_text, 'webURLTest', 'txt')
        return all_text

async def embed_chunks_in_memory_duckdb(text: str,metadata,type):
    global VECTOR_STORE, CON
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        chunks = text_splitter.split_text(text)
        print(f"Split text into {len(chunks)} chunks")
        metadata = {
            "source": metadata,  
            "file_type": type 
        }
        documents = [Document(page_content=chunk,  metadata=metadata) for chunk in chunks]
        # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
        embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
        # embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
        print("Embedding model loaded successfully")
        if VECTOR_STORE is None:
            VECTOR_STORE = DuckDB.from_documents(
                documents=documents,
                embedding=embedding_model,
                connection=CON,
                table_name="documents"
            )
        else:
            VECTOR_STORE.add_documents(documents)
        return VECTOR_STORE
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }
   
async def generate_questions(query: str ,opt: str = None):
    global CON
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 30},)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.invoke(query)
    answer = result.get("result")
    print("#######",answer)
    return answer
            
async def handleFileUpload(files: List[UploadFile] = File(...)) :
    print("file",files)
    if not files:
        return {
            "status": 200,
            "error": "Please add the file",
        }
    extractedText = await extractText(files) 
   
    if not extractedText.get('query'):  
        return {
            "status": 200,
            "error": "txt file missing"
        }
    urlResult = getURLS(extractedText.get('query'))
    if (len(urlResult) == 0) and (extractedText.get("fileTypeList") is None):
        return {
            "status": "200",
            "error": "Given file doesn't have data source, please attach url or other doc",
        }
    webtext = await extract_text_from_urls(urlResult)
    result = await generate_questions(extractedText,webtext)
    return {
        "status": "200",
        "generated_answer": result
    }
    