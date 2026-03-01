# Configuración de modelos
EMBEDDING_MODEL = "models/gemini-embedding-001"
QUERY_MODEL = "gemini-2.5-flash-lite"
GENERATION_MODEL = "gemini-2.5-flash"

# Configuración de la base de datos de vectores
VECTORSTORE_DB_PATH = "chroma_db"

# Configuración del retriever
SEARCH_TYPE = "mmr" # MMR (Maximal Margin Relevance)
MMR_DIVERSITY_LAMBDA = 0.7 # Lambda para controlar la diversidad en MMR. Valores más altos dan más peso a la relevancia, valores más bajos fomentan la diversidad.
MMR_FETCH_K = 20 # Número de documentos a recuperar antes de aplicar MMR
SEARCH_K = 2 # Número final de documentos a devolver después de aplicar MMR