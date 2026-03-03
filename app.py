import streamlit as st
from rag_system import query_rag, get_retriever_info

# Configuracion de la pagina
st.set_page_config(
  page_title="Sistema RAG - Asistente Legal",
  page_icon="⚖️",
  layout="wide"
)

# Titulo de la aplicacion
st.title("⚖️ Sistema RAG - Asistente Legal")
st.divider()

# Inicializar el historial de chat
if "messages" not in st.session_state: # Si no existe el historial de mensajes en el estado de la sesión, lo inicializamos como una lista vacía
  st.session_state.messages = []
  
# Sidebar simplificado
with st.sidebar:
  st.header("🗒️ Informacion del Sistema")
  
  # Informacion del retriever
  retriever_info = get_retriever_info()
  
  st.markdown("**🔍 Retriever:**")
  st.info(f"Tipo: {retriever_info['tipo']}")
  
  st.markdown("**🤖 Modelos:**")
  st.info(f"Consultas: {retriever_info['query_model']}\n\nGeneracion: {retriever_info['generation_model']}")
  
  st.divider()
  
  if st.button("🚮 Limpiar Chat", type="secondary", use_container_width=True):
    st.session_state.messages = [] # Limpiamos el historial de mensajes al hacer clic en el botón
    st.rerun() # Recargamos la aplicación para reflejar el cambio
    
# Layour principal con columnas
col1, col2 = st.columns([2, 1])

with col1:
  st.markdown("### 💬 Chat")
  
  # Mostrar historial de mensajes
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

with col2:
  st.markdown("### 🗒️ Documentos Relevantes")
  
  # Mostrar documentos de la ultima consulta
  if st.session_state.messages:
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "assistant" and "docs" in last_message:
      docs = last_message["docs"]
      
      if docs:
        for doc in docs:
          with st.expander(f"🗒️ Fragmento {doc['fragmento']}", expanded=False):
            st.markdown(f"**Fuente:** {doc['fuente']}")
            st.markdown(f"**Página:** {doc['pagina']}")
            st.markdown("**Contenido:**")
            st.text(doc['contenido'])  

# Input del usuario
if prompt := st.chat_input("Escribe tu consulta sobre contratos de arrendamiento..."):
    # Añadir mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generar respuesta
    with st.spinner("🔍 Analizando..."):
        response, docs = query_rag(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response, "docs": docs})
    
    # Recargar para mostrar los nuevos mensajes
    st.rerun()

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666;'>🏛️ Asistente Legal con MMR Retriever</div>", 
    unsafe_allow_html=True
)