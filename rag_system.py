from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import streamlit as st

from config import *
from prompts import *

# Funcion principal para inicializar el sistema RAG
@st.cache_resource # Cacheamos la funcion para evitar reinicializaciones innecesarias y mejorar el rendimiento en Streamlit
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
    
    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)
    
    # MultiQueryRetriever con prompt personalizado
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )
    
    # Prompt para generacion de respuestas
    prompt = PromptTemplate.from_template(RAG_TEMPLATE)
    
    # Funcion para formatear y preprocesar los documentos recuperados antes de pasarlos al prompt
    def format_docs(docs):
        formatted = []
        
        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"

            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if "\\" in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                    
                if 'page' in doc.metadata:
                    header += f", Pagina: {doc.metadata['page']}"
        
            content = doc.page_content.strip() # Eliminar espacios en blanco al inicio y al final
            formatted.append(f"{header}\n{content}")
        
        return "\n\n".join(formatted) # Unir los fragmentos con doble salto de linea para mejor legibilidad
                    
    
    # Cadena LCEL para el sistema RAG
    # Primero, definimos que entra al prompt:
    # El 'context' viene de buscar en Chroma y unir los textos
    # La 'question' viene directa del usuario (RunnablePassthrough la deja pasar tal cual)
    rag_chain = (
        {"context": mmr_multi_retriever | format_docs, "question": RunnablePassthrough()}
        
        # Esos dos datos se inyectan en el prompt
        | prompt
        
        # El prompt lleno se le pasa a llm_generation  para generar la respuesta final
        | llm_generation
        
        # Finalmente, limpiamos la respuesta de la IA para que solo entre texto puro
        | StrOutputParser()
    )
    return rag_chain

def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()
        
        # Obtener la respuesta del sistema RAG
        response = rag_chain.invoke(question)
        
        # Obtener los documentos para mostrarlos al usuario (opcional, para transparencia)
        docs = retriever.get_relevant_documents(question)
        
        # Formatear los documentos para mostrar
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content, # Mostrar solo los primeros 1000 caracteres para evitar saturar la interfaz
                "fuente": doc.metadata.get('source', 'No especificada').split("\\")[-1],
                "pagina": doc.metadata.get('page', "No especificada")
            }
            docs_info.append(doc_info)
        return response, docs_info
    except Exception as e:
        error_message = f"Error al procesar la consulta: {str(e)}"
        return error_message, []
            