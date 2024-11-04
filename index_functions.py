
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from functools import lru_cache
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

@lru_cache(maxsize=None)
def get_model(model_name, temperature, max_tokens):
    """
    Returns a language model based on the specified model name, temperature, and max tokens.

    Args:
        model_name (str): The name of the language model.
        temperature (float): The temperature parameter for generating responses.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        ChatGroq: The language model object based on the specified parameters.
    """
    # print(f"Parámetros de modelo {model_name, temperature, max_tokens}")
    llm = {
        "llama3-70b-8192": ChatGroq(temperature=temperature,model_name="llama3-70b-8192", max_tokens=max_tokens),
      "llama3-8b-8192": ChatGroq(temperature=temperature,model_name="llama3-8b-8192", max_tokens=max_tokens),
       "mixtral-8x7b-32768": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
       "gemma-7b-it": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
#        "gemini-1.5-flash-002":ChatVertexAI(model_name="gemini-1.5-flash-002",project="single-cirrus-435319-f1",verbose=True),
#        "gemini-1.5-pro-002":ChatVertexAI(model_name="gemini-1.5-pro-002",project="single-cirrus-435319-f1",verbose=True),
    }
    return llm[model_name]


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

def get_cvsName(chunks,model_name='llama3-70b-8192', temperature=0, max_tokens=200):
    #gets the name of the applicant with LLMs
    cv_name=''
    messages = [   
        ("system", """
            You are an expert in extracting the name of the applicant in a CV. Return ONLY the name and surname.
            The format should be capital letters the first letter of the name and the surname.
            With this information:
            {document_cv}
            """),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ]
    prompt_get_name = ChatPromptTemplate.from_messages(messages=messages)
    llm = get_model(model_name, temperature, max_tokens)

    chain_prompt_cv = (
    prompt_get_name 
    | llm 
    | StrOutputParser()
    )
    
    cv_name = chain_prompt_cv.invoke(
        {
            "input": 'Get me the name of the applicant',
            "document_cv": (''.join([chunk.page_content for chunk in chunks])),
            "chat_history":['']
        }
    )

    return cv_name


def upload_pdf_index(chunks,cv_name):
    #upload the pdf to the index    
    metadata={"cv_name":cv_name, "type_of_documents": 'cv'}
    for chunk in chunks:
        chunk.metadata.update(metadata)
    vector_store.add_documents(documents=chunks)

def load_pdf_index(uploaded_file):
    try:
        chunks_files= read_pdf(uploaded_file)
        cv_name= get_cvsName(chunks_files)
        upload_pdf_index(chunks_files,cv_name)
        return True,cv_name
    except:
        return False,''