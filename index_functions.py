
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

index_path = "index/mi_cv"
embeddings = FastEmbedEmbeddings()

if os.path.exists(index_path):
    vector_store = Chroma(
        collection_name="cvs",
        embedding_function=embeddings,
        persist_directory=index_path
    )
    print(f"Vector Store leído de {index_path}")


def read_pdf(uploaded_file):
    #read the pdf with langchain pdfs loaders
    # save the file temporarily
    file_name = uploaded_file.name
    temp_file = f"./{file_name}"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_file)
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=9000,
    chunk_overlap=100
    )
    chunks = loader.load_and_split(text_splitter=text_splitter)
    os.remove(temp_file)
    return chunks


def upload_pdf_index(chunks,cv_name):
    #upload the pdf to the index    
    metadata={"cv_name":cv_name, "type_of_documents": 'cv'}
    for chunk in chunks:
        chunk.metadata.update(metadata)
    vector_store.add_documents(documents=chunks)

def load_pdf_index(uploaded_file, cv_name):
    try:
        chunks_files= read_pdf(uploaded_file)
        upload_pdf_index(chunks_files,cv_name)
        return True
    except:
        return False