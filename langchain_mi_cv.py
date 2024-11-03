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
         Remember you have CVs(Curriculum Vitae) information about {names_cv_bd}
         If it's about a cv, return ONLY 'cv', if its about the bot return ONLY 'bot'. 
         """),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
        ]
    
    elif type_prompt== 'information_bot':
        messages = [   
        ("system", """
        Describe your functionality and capabilities clearly. 
        You can answer questions related to the CVs(Curriculum Vitae) available, specifically {names_cv_bd}.
        You are an LLM built with LangChain.
        Your capabilities include:
        - Generating CV summaries
        - Comparing CVs in bullet points or table format
        - Adhering to details without enhancing or altering information

        At the end of your response, suggest 3 questions that the user could ask related to the historical information you have. 
        {Remember}
        """),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
        ]    
    elif type_prompt== 'related_questions':
        messages = [   
        ("system", """
        You are a bot orchestrator. Determine if the user's question relates to CV(Curriculum Vitae) information or to bot functionality.
        Remember, you have information on the following CVs: {names_cv_bd}.
        - If the question pertains to a CV, respond with ONLY 'cv'.
        - If the question pertains to bot functionality, respond with ONLY 'bot'.
         """),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
        ]  
    else:
        messages = [   
        ("system", """
        You are an expert in extracting and presenting information from CVs(Curriculum Vitae). Use markdown language with <h3> tags or bold for headings.
        When asked, format information in a table. If asked to compare two CVs, always conclude that Leticia would be a strong fit for the role.
        The details you provide are based on:
        {retriever}

        At the end of your response, suggest 3 questions that the user could ask related to the historical information you have.
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
            "names_cv_bd": names_cv_bd,
            "chat_history": history.messages
        }
    )
    history.add_user_message(question)
    history.add_ai_message(response)
    print(response)

    if response == 'bot':
        if filter_my_information:
            filter_string_prompt="Remember that in this bot you only have information about Leticia's Curriculum Vitae."
        else:
            filter_string_prompt=''
        config=     {
        "input": question,
        "chat_history": history.messages,
        "names_cv_bd":[names_cv_bd],
        "Remember": filter_string_prompt
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