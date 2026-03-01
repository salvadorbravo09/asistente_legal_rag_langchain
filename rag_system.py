from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import streamlit as st

from config import *

# Funcion principal para inicializar el sistema RAG
def initialize_rag_system():
    # Inicializar la base de datos de vectores con Chroma
    vectorstore = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=VECTORSTORE_DB_PATH,
    )

    # Configurar el modelo de lenguaje generativo
    llm_queries = ChatGoogleGenerativeAI(model=QUERY_MODEL, temperature=0) # Temperatura 0 para respuestas más determinísticas
    llm_generation = ChatGoogleGenerativeAI(model=GENERATION_MODEL, temperature=0) # Temperatura 0 para respuestas más determinísticas
    
    # Retriever para obtener documentos relevantes. MMR (Maximal Margin Relevance)
    base_retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K, # Número final de documentos a devolver después de aplicar MMR
            "lambda_mult": MMR_DIVERSITY_LAMBDA, # Lambda para controlar la diversidad en MMR
            "fetch_k": MMR_FETCH_K, # Número de documentos a recuperar antes de aplicar MMR
        }
    )
    
