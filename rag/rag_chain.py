# File: rag_chain.py
# VERSIONE CORRETTA - ESTRAZIONE FILASTROCCHE PERFETTA

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

import os
import shutil
import re

PDF_DIR = 'data/pdfs'
VECTOR_DIR = 'data/vectors'
os.makedirs(VECTOR_DIR, exist_ok=True)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def load_vectorstore():
    faiss_index = os.path.join(VECTOR_DIR, 'index.faiss')
    if os.path.exists(faiss_index):
        print("[INFO] Caricamento vectorstore FAISS esistente...")
        return FAISS.load_local(VECTOR_DIR, embeddings=embedding, allow_dangerous_deserialization=True)
    else:
        print("[INFO] Nessun vectorstore FAISS trovato.")
        return None

vectorstore = load_vectorstore()
last_used_model = None

def extract_poems_from_text(text):
    """Estrae le filastrocche CORRETTAMENTE"""
    poems = []
    
    # Metodo più robusto: trova tutti i titoli prima
    lines = text.split('\n')
    poem_starts = []
    
    for i, line in enumerate(lines):
        if line.strip().startswith('Filastrocca '):
            poem_starts.append(i)
    
    # Estrae ogni filastrocca dal suo inizio fino al prossimo titolo
    for i, start_idx in enumerate(poem_starts):
        # Trova dove finisce questa filastrocca
        if i < len(poem_starts) - 1:
            end_idx = poem_starts[i + 1]
        else:
            end_idx = len(lines)
        
        # Costruisce il testo completo della filastrocca
        poem_lines = lines[start_idx:end_idx]
        poem_text = '\n'.join(poem_lines).strip()
        
        # Rimuove righe vuote alla fine
        while poem_text.endswith('\n'):
            poem_text = poem_text[:-1]
        
        # Solo se ha contenuto significativo (più del titolo)
        if len(poem_lines) > 2 and len(poem_text) > 50:
            poems.append(poem_text)
            
            # Log per debug
            title_line = poem_lines[0] if poem_lines else "Senza titolo"
            print(f"[INFO] Estratta: {title_line}")
            print(f"[INFO] Lunghezza: {len(poem_text)} caratteri")
    
    return poems

def ingest_pdfs(pdf_paths):
    global vectorstore
    all_documents = []
    
    for pdf_path in pdf_paths:
        print(f"[INFO] Caricamento PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"[INFO] Trovate {len(documents)} pagine in {pdf_path}")
        
        # Unisce tutto il contenuto del PDF
        full_text = "\n".join([doc.page_content for doc in documents])
        
        # Estrae le filastrocche come documenti separati
        poems = extract_poems_from_text(full_text)
        
        # Crea un documento per ogni filastrocca
        for i, poem in enumerate(poems):
            # Estrae il titolo per i metadati
            first_line = poem.split('\n')[0]
            title = first_line.replace('Filastrocca ', '').strip()
            
            doc = Document(
                page_content=poem,
                metadata={
                    'source': pdf_path,
                    'poem_index': i,
                    'title': title,
                    'type': 'poem'
                }
            )
            all_documents.append(doc)
            print(f"[INFO] Memorizzata filastrocca: {title}")
    
    print(f"[INFO] Creati {len(all_documents)} documenti (filastrocche)")
    
    if vectorstore is None:
        vectorstore = FAISS.from_documents(all_documents, embedding)
    else:
        vectorstore.add_documents(all_documents)

    vectorstore.save_local(VECTOR_DIR)
    return len(all_documents), all_documents

def find_best_poem_match(question, docs):
    """Trova la filastrocca migliore usando ricerca fuzzy"""
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    best_match = None
    best_score = 0
    
    for doc in docs:
        content_lower = doc.page_content.lower()
        title_lower = doc.metadata.get('title', '').lower()
        
        # Combina titolo e prime righe per la ricerca
        search_text = title_lower + " " + content_lower[:200]
        search_words = set(search_text.split())
        
        # Calcola punteggio di somiglianza
        common_words = question_words.intersection(search_words)
        score = len(common_words)
        
        # Bonus per parole chiave specifiche nel titolo
        title_words = set(title_lower.split())
        title_matches = question_words.intersection(title_words)
        score += len(title_matches) * 2  # Peso doppio per match nel titolo
        
        # Bonus per parole parziali (substring matching)
        for q_word in question_words:
            if len(q_word) > 3:  # Solo per parole significative
                if q_word in title_lower:
                    score += 3
                elif q_word in content_lower:
                    score += 1
        
        # Verifica match parziali inversi (parole del titolo contenute nella domanda)
        for t_word in title_words:
            if len(t_word) > 3:
                for q_word in question_words:
                    if t_word in q_word or q_word in t_word:
                        score += 2
        
        if score > best_score:
            best_score = score
            best_match = doc
        
        # Debug info
        print(f"[DEBUG] Filastrocca: {title_lower[:30]}... Score: {score}")
    
    return best_match

def ask_question(question, model_name='mistral', system_message=None):
    global vectorstore, last_used_model

    if vectorstore is None or model_name != last_used_model:
        vectorstore = load_vectorstore()
        last_used_model = model_name
        if vectorstore is None:
            return "[ERRORE] Nessun documento indicizzato. Caricare un PDF."

    llm = Ollama(model=model_name, temperature=0.1)
    q = ("" if question is None else str(question)).strip()

    if not q:
        return ""
    
    
    # Controlla se è una richiesta di filastrocca
    is_poem_request = any(word in question.lower() for word in ['filastrocca', 'recita', 'dimmi', 'racconta'])
    
    if is_poem_request:
        # Per le filastrocche, prendi tutti i documenti disponibili
        docs = vectorstore.similarity_search(question, k=10)
        
        # Usa la ricerca fuzzy per trovare la migliore corrispondenza
        best_match = find_best_poem_match(question, docs)
        
        if best_match:
            print(f"[INFO] Selezionata filastrocca: {best_match.metadata.get('title', 'Senza titolo')}")
            return best_match.page_content
        else:
            return "Non ho trovato una filastrocca corrispondente alla tua richiesta."
    
    else:
        # Per domande generiche
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Basandoti solo su queste informazioni:

{context}

Rispondi alla domanda: {question}

Se la risposta non è presente, di' "Non trovo questa informazione"."""

        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Errore: {str(e)}"

def get_indexed_chunks():
    if vectorstore is None:
        return []
    docs = vectorstore.similarity_search("", k=100)
    return [doc.page_content for doc in docs]

def clear_vectorstore():
    global vectorstore
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectorstore = None