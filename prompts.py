# Prompt principal para el sistema RAG
# Le indica al modelo que actúe como un asistente legal especializado en contratos de arrendamiento
RAG_TEMPLATE = """Eres un asistente legal especializado en contratos de arrendamiento.
Basandote UNICAMENTE en los siguientes fragmentos de contratos, responde a la pregunta del usuario.

FRAGMENTOS DE CONTRATOS:
{context}

PREGUNTA:
{question}

INSTRUCCIONES:
- Proporciona una respuesta clara y directa basada en la informacion disponible.
- Si encuentras la informacion exacta, citala textualmente cuando sea relevante.
- Incluye todos los detalles importantes: nombres, direcciones, importes, fechas.
- Si la informacion esta incompleta o no esta disponible, indicalo claramente.
- Organiza la informacion de manera estructurada si es necesaria.
- Si hay multiples contratos o personas mencionadas, especifica a cual te refieres.

RESPUESTA:
"""

# Prompt personalizado para el MultiQueryRetriever
MULTI_QUERY_PROMPT = """Eres un experto en analisis de documentos legales especializados en contratos de arrendamiento.
Tu tarea es generar multiples versiones de la consulta del usuario para recuperar documentos relevantes desde una base de datos vectorial.

Al generar variaciones de la consulta, considera:
- Diferentes formas de referirse a personas (nombre completo, apellido, solo nombre).
- Sinonimos legales y terminos tecnicos de arrendamiento
- Variaciones en la formulacion de preguntas sobre aspectos contractuales
- Terminos relacionados con ubicaciones, propiedades y condiciones del contrato

Consulta original: {question}

Genera exactamente 3 versiones alternativas de esta consulta, una por linea, sin numeracion ni viñetas:
"""

# Prompt para analisis de relevancia de documentos
RELEVANCE_PROMPT = """Analiza si el siguiente fragmento de documento es relevante para responder la consulta del usuario.

FRAGMENTO:
{document}

CONSULTA:
{question}

¿Es este fragmento relevante para responder la consulta? Responde con "SI" o "NO" y una breve justificación.
"""

# Prompt para extraccion de entidades clave
ENTITY_EXTRACTION_PROMPT = """Extrae las entidades clave del siguiente texto de contrato de arrendamiento:

TEXTO:
{text}

Identifica y extrae:
- Nombres de personas (arrendador, arrendatario, avalistas)
- Direcciones de propiedades
- Importes monetarios
- Fechas importantes
- Duración del contrato
- Tipo de propiedad

Formato de respuesta:
PERSONAS: [lista de nombres]
DIRECCIONES: [lista de direcciones]
IMPORTES: [lista de cantidades]
FECHAS: [lista de fechas]
DURACIÓN: [periodo del contrato]
TIPO: [tipo de propiedad]
"""
