# File: rag_chain2.py
# VERSIONE FINALE - RAG OTTIMIZZATO + TRADUZIONI MIGLIORATE + FILASTROCCHE

# File: rag_chain2.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

import os
import shutil
import re

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
            print(f"[INFO] Estratta filastrocca: {title_line}")
            print(f"[INFO] Lunghezza: {len(poem_text)} caratteri")
    
    return poems

def detect_document_type(text, filename):
    """Rileva il tipo di documento basandosi sul contenuto e nome file"""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    # Verifica se contiene filastrocche (deve essere molto specifico)
    if 'filastrocca ' in text_lower and text_lower.count('filastrocca ') > 5:
        return 'poems'
    
    # Tutti gli altri tipi di documento vengono trattati come generici
    # per garantire l'uso del sistema RAG
    return 'general'

def create_chunks_from_text(text, pdf_path, doc_type, chunk_size=600, chunk_overlap=80):
    """Crea chunks ottimizzati per migliorare la precisione del RAG"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    )
    
    chunks = text_splitter.split_text(text)
    documents = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 25:  # Ignora chunks troppo piccoli
            # Pulisce il chunk
            clean_chunk = chunk.strip()
            
            doc = Document(
                page_content=clean_chunk,
                metadata={
                    'source': pdf_path,
                    'chunk_index': i,
                    'type': doc_type,
                    'filename': os.path.basename(pdf_path)
                }
            )
            documents.append(doc)
    
    return documents

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
        
        # Rileva il tipo di documento
        doc_type = detect_document_type(full_text, pdf_path)
        print(f"[INFO] Tipo documento rilevato: {doc_type}")
        
        if doc_type == 'poems':
            # MANTIENI LA LOGICA ESISTENTE PER LE FILASTROCCHE
            poems = extract_poems_from_text(full_text)
            
            for i, poem in enumerate(poems):
                first_line = poem.split('\n')[0]
                title = first_line.replace('Filastrocca ', '').strip()
                
                doc = Document(
                    page_content=poem,
                    metadata={
                        'source': pdf_path,
                        'poem_index': i,
                        'title': title,
                        'type': 'poem',
                        'filename': os.path.basename(pdf_path)
                    }
                )
                all_documents.append(doc)
                print(f"[INFO] Memorizzata filastrocca: {title}")
        
        else:
            # PER TUTTI GLI ALTRI DOCUMENTI: crea chunks ottimizzati
            chunks = create_chunks_from_text(full_text, pdf_path, doc_type)
            all_documents.extend(chunks)
            print(f"[INFO] Creati {len(chunks)} chunks per documento: {os.path.basename(pdf_path)}")
    
    print(f"[INFO] Creati {len(all_documents)} documenti totali")
    
    if vectorstore is None:
        vectorstore = FAISS.from_documents(all_documents, embedding)
    else:
        vectorstore.add_documents(all_documents)

    vectorstore.save_local(VECTOR_DIR)
    return len(all_documents), all_documents

def calculate_relevance_score(question, doc_content):
    """Calcola quanto un documento è rilevante per la domanda con maggiore precisione"""
    question_lower = question.lower()
    content_lower = doc_content.lower()
    
    question_words = set(question_lower.split())
    content_words = set(content_lower.split())
    
    # Rimuove parole comuni non significative
    stop_words = {'il', 'la', 'di', 'che', 'e', 'a', 'un', 'una', 'in', 'per', 'con', 'su', 'da', 'del', 'della', 'dei', 'delle', 'al', 'alla', 'agli', 'alle', 'nel', 'nella', 'nei', 'nelle', 'sul', 'sulla', 'sui', 'sulle'}
    question_words = question_words - stop_words
    content_words = content_words - stop_words
    
    if len(question_words) == 0:
        return 0
    
    # Calcola sovrapposizione esatta con peso maggiore
    exact_matches = len(question_words.intersection(content_words))
    
    # Calcola match parziali più rigorosi
    partial_matches = 0
    for q_word in question_words:
        if len(q_word) > 4:  # Solo parole davvero significative
            for c_word in content_words:
                if len(c_word) > 4 and (q_word in c_word or c_word in q_word):
                    partial_matches += 0.3
                    break
    
    # Verifica presenza di frasi chiave
    phrase_bonus = 0
    if len(question_words) > 1:
        # Cerca combinazioni di 2-3 parole consecutive
        question_phrases = []
        q_words_list = question_lower.split()
        for i in range(len(q_words_list) - 1):
            phrase = f"{q_words_list[i]} {q_words_list[i+1]}"
            if phrase in content_lower:
                phrase_bonus += 0.5
    
    # Punteggio finale con pesi bilanciati
    total_score = (exact_matches * 1.0 + partial_matches + phrase_bonus) / len(question_words)
    return total_score

def find_best_poem_match(question, docs):
    """Trova la filastrocca migliore usando ricerca fuzzy"""
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    # Filtra solo i documenti di tipo 'poem'
    poem_docs = [doc for doc in docs if doc.metadata.get('type') == 'poem']
    
    best_match = None
    best_score = 0
    
    for doc in poem_docs:
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
        
        # Verifica match parziali inversi
        for t_word in title_words:
            if len(t_word) > 3:
                for q_word in question_words:
                    if t_word in q_word or q_word in t_word:
                        score += 2
        
        if score > best_score:
            best_score = score
            best_match = doc
        
        print(f"[DEBUG] Filastrocca: {title_lower[:30]}... Score: {score}")
    
    return best_match

def get_concise_response(question, model_name):
    """Genera una risposta concisa usando un prompt di sistema ottimizzato"""
    try:
        chat_llm = ChatOllama(model=model_name, base_url=OLLAMA_URL, temperature=0.3)
        
        system_prompt = """Sei un assistente AI che fornisce risposte brevi, precise e dirette sempre in italiano.
Regole per le risposte:
- Rispondi SEMPRE in italiano corretto e naturale
- Se devi tradurre da altre lingue, fallo in modo fluente e accurato
- Usa terminologia italiana appropriata e corrente
- Massimo 2-3 frasi per risposta, vai dritto al punto
- Non usare asterischi, simboli speciali o formattazioni strane
- Se la domanda richiede una spiegazione, forniscila in modo semplice e chiaro
- Non aggiungere saluti o convenevoli non richiesti
- Usa un linguaggio naturale e colloquiale quando appropriato"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        response = chat_llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        print(f"[DEBUG] ChatOllama non disponibile, uso Ollama standard: {e}")
        llm = Ollama(model=model_name, base_url=OLLAMA_URL, temperature=0.3)

        enhanced_question = f"""Rispondi in modo breve e preciso (massimo 2-3 frasi) SEMPRE in italiano naturale e corretto. Se devi tradurre da altre lingue, fallo in modo fluente:

{question}"""
        
        response = llm.invoke(enhanced_question)
        return response.strip()

def get_rag_response(question, model_name, docs):
    """Genera una risposta precisa usando RAG sui documenti trovati"""
    try:
        chat_llm = ChatOllama(model=model_name, base_url=OLLAMA_URL, temperature=0.05)  # Temperatura molto bassa per precisione
        
        # Prepara il contesto dai documenti più rilevanti
        context_parts = []
        for i, doc in enumerate(docs[:3]):  # Usa solo i migliori 3 documenti
            context_parts.append(f"DOCUMENTO {i+1}:\n{doc.page_content.strip()}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        system_prompt = """Sei un assistente AI esperto che analizza documenti con massima precisione.
Regole RIGIDE:
- Rispondi SEMPRE in italiano perfetto e naturale
- Se il documento è in inglese, traduci ACCURATAMENTE mantenendo tutti i dettagli
- Usa ESCLUSIVAMENTE le informazioni presenti nel contesto
- Mantieni tutti i numeri, percentuali, date e nomi ESATTI dal documento
- Traduci terminologia tecnica correttamente (es: "global cooperation" = "cooperazione globale")
- Rispondi in modo diretto e preciso in 2-3 frasi massimo
- Se la domanda chiede informazioni specifiche, fornisci ESATTAMENTE quello che c'è nel documento
- Non riassumere troppo, mantieni i dettagli importanti"""

        user_prompt = f"""CONTESTO DAL DOCUMENTO:
{context}

DOMANDA: {question}

Rispondi con precisione assoluta basandoti SOLO sul contesto. Traduci tutto in italiano mantenendo tutti i dettagli."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = chat_llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        print(f"[DEBUG] Errore in get_rag_response: {e}")
        return f"Errore nell'elaborazione del documento: {str(e)}"

def ask_question(question, model_name='mistral', system_message=None):
    global vectorstore, last_used_model

    if model_name != last_used_model:
        last_used_model = model_name

    q = ("" if question is None else str(question)).strip()

    if not q:
        return ""
    
    # Controlla se è una richiesta di filastrocca specifica
    is_specific_poem_request = any(word in question.lower() for word in ['filastrocca', 'recita', 'dimmi', 'racconta']) and any(word in question.lower() for word in ['luna', 'animali', 'corta', 'favole', 'mondo', 'errori', 'leggere'])
   
    if vectorstore is not None:
        # Cerca nei documenti caricati con più risultati per migliorare la precisione
        docs = vectorstore.similarity_search(question, k=20)
        
        if docs:
            if is_specific_poem_request:
                # MANTIENI LA LOGICA ESISTENTE PER LE FILASTROCCHE (INVARIATA)
                best_match = find_best_poem_match(question, docs)
                if best_match:
                    print(f"[INFO] Selezionata filastrocca: {best_match.metadata.get('title', 'Senza titolo')}")
                    return best_match.page_content
            
            # LOGICA RAG MIGLIORATA PER DOCUMENTI GENERALI
            # Filtra documenti non-poesie
            non_poem_docs = [doc for doc in docs if doc.metadata.get('type') != 'poem']
            
            if non_poem_docs:
                # Calcola rilevanza per ogni documento
                scored_docs = []
                for doc in non_poem_docs:
                    score = calculate_relevance_score(question, doc.page_content)
                    scored_docs.append((doc, score))
                
                # Ordina per rilevanza decrescente
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                # Soglia di rilevanza più rigorosa
                relevant_docs = [doc for doc, score in scored_docs if score > 0.25]  
                
                if relevant_docs:
                    print(f"[INFO] Trovati {len(relevant_docs)} documenti altamente rilevanti nel PDF")
                    print(f"[INFO] Migliore rilevanza: {scored_docs[0][1]:.2f}")
                    print(f"[INFO] Fonte: {relevant_docs[0].metadata.get('filename', 'unknown')}")
                    return get_rag_response(question, model_name, relevant_docs)
                
                # Soglia media di rilevanza
                elif scored_docs[0][1] > 0.12:  # Soglia più bassa ma comunque significativa
                    medium_relevant = [doc for doc, score in scored_docs[:5] if score > 0.12]
                    if medium_relevant:
                        print(f"[INFO] Trovati documenti mediamente rilevanti (score: {scored_docs[0][1]:.2f})")
                        return get_rag_response(question, model_name, medium_relevant)
                
                # Se non raggiunge nemmeno la soglia minima
                print(f"[INFO] Documenti nei PDF non sufficientemente rilevanti (migliore: {scored_docs[0][1]:.2f})")
            else:
                print(f"[INFO] Nessun documento non-poesia trovato nei PDF")
    else:
        print(f"[INFO] Nessun PDF caricato nel sistema")
    
    # FALLBACK GARANTITO: Usa SEMPRE il modello AI selezionato per domande generali
    print(f"[INFO] Interrogo il modello {model_name} per risposta generale")
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