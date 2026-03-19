# File: rag_chain.py
# VERSIONE CORRETTA - ESTRAZIONE FILASTROCCHE PERFETTA + RISPOSTE CONCISE + COMANDI ROBOT + FILASTROCCHE CASUALI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

import os
import shutil
import re
import random

PDF_DIR = 'data/pdfs'
VECTOR_DIR = 'data/vectors'
OLLAMA_URL = "http://127.0.0.1:11434"
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

def is_robot_command(question):
    """Verifica se la domanda è un comando diretto per il robot"""
    command_keywords = [
        'avanti',
        'dietro', 
        'destra',
        'sinistra',
        'alza braccio destro',
        'alza braccio sinistro'
    ]
    
    question_lower = question.lower().strip()
    
    # Controlla match esatti o parziali
    for cmd in command_keywords:
        if cmd in question_lower:
            return True
    
    return False

def is_random_poem_request(question):
    """Verifica se la richiesta è per una filastrocca casuale"""
    random_patterns = [
        'mi dici una filastrocca',
        'mi reciti una filastrocca',
        'mi racconti una filastrocca',
        'dimmi una filastrocca',
        'raccontami una filastrocca',
        'recitami una filastrocca',
        'una filastrocca casuale',
        'filastrocca random'
    ]
    
    question_lower = question.lower().strip()
    
    for pattern in random_patterns:
        if pattern in question_lower:
            return True
    
    return False

def is_ai_poem_request(question):
    """Verifica se la richiesta è per l'AI di generare una filastrocca"""
    ai_patterns = [
        'chiedi una filastrocca all\'ai',
        'chiedi all\'ai una filastrocca',
        'genera una filastrocca',
        'crea una filastrocca',
        'inventa una filastrocca',
        'scrivi una filastrocca'
    ]
    
    question_lower = question.lower().strip()
    
    for pattern in ai_patterns:
        if pattern in question_lower:
            return True
    
    return False

def get_random_poem():
    """Restituisce una filastrocca casuale dal vectorstore"""
    global vectorstore
    
    # Prova a ricaricare il vectorstore se è None
    if vectorstore is None:
        vectorstore = load_vectorstore()
    
    if vectorstore is None:
        print("[DEBUG] Vectorstore ancora None dopo ricaricamento")
        return None
    
    try:
        # Recupera MOLTI più documenti per essere sicuri
        all_docs = vectorstore.similarity_search("filastrocca", k=200)
        
        print(f"[DEBUG] Documenti totali trovati: {len(all_docs)}")
        
        # Filtra solo le filastrocche (quelle che iniziano con "Filastrocca ")
        poem_docs = [doc for doc in all_docs if doc.page_content.strip().startswith('Filastrocca ')]
        
        print(f"[DEBUG] Filastrocche trovate: {len(poem_docs)}")
        
        # Se non trova nulla, prova a cercare in modo diverso
        if not poem_docs:
            print("[DEBUG] Tentativo alternativo di ricerca...")
            all_docs = vectorstore.similarity_search("", k=200)
            poem_docs = [doc for doc in all_docs if 'Filastrocca' in doc.page_content[:100]]
            print(f"[DEBUG] Filastrocche trovate (metodo alternativo): {len(poem_docs)}")
        
        if not poem_docs:
            return None
        
        # Seleziona una filastrocca casuale
        random_poem = random.choice(poem_docs)
        title = random_poem.metadata.get('title', random_poem.page_content.split('\n')[0])
        print(f"[INFO] Filastrocca casuale selezionata: {title}")
        
        return random_poem.page_content
        
    except Exception as e:
        print(f"[ERROR] Errore in get_random_poem: {e}")
        return None

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

def get_concise_response(question, model_name):
    """Genera una risposta concisa usando un prompt di sistema ottimizzato"""
    try:
        # Usa ChatOllama per avere controllo sui messaggi di sistema
        chat_llm = ChatOllama(model=model_name, base_url=OLLAMA_URL, temperature=0.3)

        # Prompt di sistema per risposte concise
        system_prompt = """Sei un assistente AI che fornisce risposte brevi, precise e dirette.
Regole:
- Rispondi sempre in italiano
- Massimo 2-3 frasi per risposta
- Non usare asterischi, simboli speciali o formattazioni strane
- Vai dritto al punto senza prolissità
- Se la domanda richiede una spiegazione, forniscila in modo semplice e chiaro
- Non aggiungere saluti o convenevoli non richiesti"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]

        response = chat_llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        # Fallback: usa Ollama standard con prompt integrato
        print(f"[DEBUG] ChatOllama non disponibile, uso Ollama standard: {e}")
        llm = Ollama(model=model_name, base_url=OLLAMA_URL, temperature=0.3)
        
        # Integra il prompt direttamente nella domanda
        enhanced_question = f"""Rispondi in modo breve e preciso (massimo 2-3 frasi) in italiano, senza asterischi o simboli speciali:

{question}"""
        
        response = llm.invoke(enhanced_question)
        return response.strip()

def ask_question(question, model_name='mistral', system_message=None):
    global vectorstore, last_used_model

    if model_name != last_used_model:
        last_used_model = model_name

    q = ("" if question is None else str(question)).strip()

    if not q:
        return ""
    
    # PRIORITÀ 1: Controlla se è un comando per il robot
    if is_robot_command(q):
        print(f"[INFO] Rilevato comando robot: {q}")
        return f"comando riconosciuto: {q}"
    
    # PRIORITÀ 2: Controlla se è una richiesta di filastrocca CASUALE (generica, senza specificare quale)
    # Questo DEVE venire PRIMA della ricerca specifica
    if is_random_poem_request(q):
        print(f"[INFO] ⭐ Richiesta filastrocca casuale dal PDF")
        random_poem = get_random_poem()
        if random_poem:
            return random_poem
        else:
            # Se il vectorstore è vuoto, prova a ricaricare
            print("[DEBUG] Nessuna filastrocca trovata, tento di ricaricare il vectorstore...")
            global vectorstore
            vectorstore = load_vectorstore()
            random_poem = get_random_poem()
            if random_poem:
                return random_poem
            else:
                return "Mi dispiace, non ho trovato filastrocche nel database. Carica un PDF con filastrocche prima."
    
    # PRIORITÀ 3: Controlla se è una richiesta di filastrocca all'AI (DEVE ESSERE ESPLICITO)
    if is_ai_poem_request(q):
        print(f"[INFO] Richiesta filastrocca all'AI")
        poem_prompt = "Scrivi una breve filastrocca originale in italiano, semplice e adatta ai bambini."
        return get_concise_response(poem_prompt, model_name)
    
    # PRIORITÀ 4: Controlla se è una richiesta di filastrocca specifica (con parole chiave specifiche)
    is_specific_poem_request = any(word in question.lower() for word in ['filastrocca', 'recita', 'dimmi', 'racconta']) and any(word in question.lower() for word in ['luna', 'animali', 'corta', 'favole', 'mondo', 'errori', 'leggere'])
   
    if is_specific_poem_request and vectorstore is not None:
        print(f"[INFO] Ricerca filastrocca per: {q}")
        docs = vectorstore.similarity_search(question, k=10)
        best_match = find_best_poem_match(question, docs)
        
        if best_match:
            print(f"[INFO] ✅ Trovata filastrocca: {best_match.metadata.get('title', 'Senza titolo')}")
            return best_match.page_content
        else:
            print(f"[INFO] ❌ Nessuna filastrocca trovata, interrogo AI")
            # Se non trova filastrocca, passa all'AI
    
    # PRIORITÀ 5: Per TUTTE le altre domande (o se non ha trovato filastrocca) - interroga l'AI
    print(f"[INFO] Interrogazione AI con modello: {model_name}")
    return get_concise_response(question, model_name)
    
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
