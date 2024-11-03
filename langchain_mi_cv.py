import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
import os
from functools import lru_cache
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI

from sympy import im
import index_functions as af
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import traceback
import pandas as pd
from langchain.retrievers import ParentDocumentRetriever


load_dotenv()

index_path = "index/mi_cv"
embeddings = FastEmbedEmbeddings()

if os.path.exists(index_path):
    vector_store = Chroma(
        collection_name="cvs",
        embedding_function=embeddings,
        persist_directory=index_path
    )
    print(f"Vector Store in {index_path}")

def prompts(type_prompt):

    if type_prompt== 'orquestrator_prompt':
        messages = [   
        ("system", """
         You are an orquestrator of a bot.
         You have to check if the question that the user is asking is a question about a cv, or a question about the functionality of the bot.
         If it's about a cv, return ONLY 'cv', if its about the bot return ONLY 'bot'. Remember you have cvs information about {names_cv_bd}
         """),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
        ]
    
    elif type_prompt== 'information_bot':
        messages = [   
        ("system", """
         You have to provide information about the functionality of what you can do.
         You can answer questions related to {names_cv_bd}.
         You are a LLM done with lanchain.
         You can generate a summaty of cvs
         You can do bullet points or tables comparing CVs.

         At the end of the response tell the user 3 questions that he can ask you that are related to all the history information that you have.
         """),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
        ]    
    elif type_prompt== 'related_questions':
        messages = [   
        ("system", """
         You have to tell the user 3 questions that he can ask you that are related to all the history information that you have.
         """),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
        ]  
    else:
        messages = [   
        ("system", """
         You are an expert in extracting information about cvs.
         The information you have to tell the user is based on:
         {retriever}

        At the end of the response tell the user 3 questions that he can ask you that are related to all the history information that you have.
         """),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
        ]
    
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    return prompt


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
    print(f"Parámetros de modelo {model_name, temperature, max_tokens}")
    llm = {
        "llama3-70b-8192": ChatGroq(temperature=temperature,model_name="llama3-70b-8192", max_tokens=max_tokens),
      "llama3-8b-8192": ChatGroq(temperature=temperature,model_name="llama3-8b-8192", max_tokens=max_tokens),
       "mixtral-8x7b-32768": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
       "gemma-7b-it": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
#        "gemini-1.5-flash-002":ChatVertexAI(model_name="gemini-1.5-flash-002",project="single-cirrus-435319-f1",verbose=True),
#        "gemini-1.5-pro-002":ChatVertexAI(model_name="gemini-1.5-pro-002",project="single-cirrus-435319-f1",verbose=True),
    }
    return llm[model_name]



def create_history(messages):
    """
    Creates a ChatMessageHistory object based on the given list of messages.

    Args:
        messages (list): A list of messages, where each message is a dictionary with "role" and "content" keys.

    Returns:
        ChatMessageHistory: A ChatMessageHistory object containing the user and AI messages.

    """
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question, messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192, json_params=None, db_name=None,filter_my_information=True):
    """
    Invokes the language chain model to generate a response based on the given question and chat history.

    Args:
        question (str): The question to be asked.
        messages (list): List of previous chat messages.
        model_name (str, optional): The name of the language chain model to use. Defaults to "llama3-70b-8192".
        temperature (float, optional): The temperature parameter for controlling the randomness of the model's output. Defaults to 0.
        max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 8192.

    Yields:
        str: The generated response from the language chain model.

    """
    
    llm = get_model(model_name, temperature, max_tokens)
    history = create_history(messages)
    aux = {}
    response = ""

    #First chain to determine if they are asking information abour the bot or about the CV
    chain_orquestrator = (
    prompts('orquestrator_prompt')   
    | llm 
    | StrOutputParser()
    )
    names_cv_bd=set()
    for metadata in vector_store._collection.get()['metadatas']:
        if 'cv_name' in metadata.keys():
            names_cv_bd.add(metadata['cv_name'])
    response = chain_orquestrator.invoke(
        {
            "input": question,
            "nombre": "Leti",
            "names_cv_bd": names_cv_bd,
            "chat_history": history.messages
        }
    )
    history.add_user_message(question)
    history.add_ai_message(response)
    print(response)

    if response == 'bot':

        config=     {
        "input": question,
        "chat_history": history.messages,
        "names_cv_bd":[names_cv_bd]
        }

        chain = (
        prompts('information_bot')
        | llm 
        | StrOutputParser()
        )
        
        for chunk in chain.stream(config):
            response+=chunk
            yield chunk

        
    else:
        #related to information about CVs
        # figure years TODO


        if filter_my_information:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,
                    "filter": {'cv_name':'Leticia'}
                }                
            )
        else:            
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5
                }
            )


        config=     {
        "input": question,
        "chat_history": history.messages,
        "names_cv_bd":[names_cv_bd]
        }
                
        chain = (
        {"retriever": retriever, "input": RunnablePassthrough(),"names_cv_bd": RunnablePassthrough()}
        | prompts('cv_info')
        | llm 
        | StrOutputParser()
    )
        
        for chunk in chain.stream(config):
            response+=chunk
            yield chunk
            
    
    
    history.add_user_message(question)
    history.add_ai_message(response)
    
    
    invoke_chain.response = response
    invoke_chain.history = history
    invoke_chain.aux = aux