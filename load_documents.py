"""
Carga los PDFs de 'contratos/' en la base de datos vectorial Chroma.
Uso: python load_documents.py
"""

import os, shutil, glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import EMBEDDING_MODEL, VECTORSTORE_DB_PATH

CONTRACTS_DIR = "contratos"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main():
    # 1. Cargar todos los PDFs
    pdf_paths = sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.pdf")))
    if not pdf_paths:
        print("❌ No se encontraron PDFs en 'contratos/'")
        return

    documents = []
    for path in pdf_paths:
        docs = PyPDFLoader(path).load()
        print(f"📄 {os.path.basename(path)} → {len(docs)} página(s)")
        documents.extend(docs)

    # 2. Dividir en fragmentos
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(documents)
    print(f"\n📝 {len(chunks)} fragmentos generados")

    # 3. Crear vectorstore (reemplaza el anterior)
    if os.path.exists(VECTORSTORE_DB_PATH):
        shutil.rmtree(VECTORSTORE_DB_PATH)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=VECTORSTORE_DB_PATH,
    )
    print(f"✅ {vectorstore._collection.count()} fragmentos almacenados en '{VECTORSTORE_DB_PATH}/'")


if __name__ == "__main__":
    main()
