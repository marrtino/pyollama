# File: rag_chain4.py - VERSIONE FINALE DEFINITIVA CORRETTA
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
import json

PDF_DIR = 'data/pdfs'
VECTOR_DIR = 'data/vectors'
LIBRARY_DATA_FILE = 'data/library_catalog.json'
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
library_catalog = []
vectorstore = None
last_used_model = None

def load_vectorstore():
    global vectorstore
    faiss_index = os.path.join(VECTOR_DIR, 'index.faiss')
    if os.path.exists(faiss_index):
        print("[INFO] Caricamento vectorstore FAISS esistente...")
        vectorstore = FAISS.load_local(VECTOR_DIR, embeddings=embedding, allow_dangerous_deserialization=True)
        return vectorstore
    else:
        print("[INFO] Nessun vectorstore FAISS trovato.")
        return None

def load_library_catalog():
    global library_catalog
    if os.path.exists(LIBRARY_DATA_FILE):
        try:
            with open(LIBRARY_DATA_FILE, 'r', encoding='utf-8') as f:
                library_catalog = json.load(f)
            print(f"[INFO] Catalogo biblioteca caricato: {len(library_catalog)} libri")
        except Exception as e:
            print(f"[ERROR] Errore caricamento catalogo: {e}")
            library_catalog = []
    else:
        library_catalog = []

def save_library_catalog():
    global library_catalog
    
    # Rimuovi duplicati basati su ISBN
    seen_isbns = set()
    unique_books = []
    for book in library_catalog:
        if book['isbn'] not in seen_isbns:
            unique_books.append(book)
            seen_isbns.add(book['isbn'])
    
    library_catalog = unique_books
    
    try:
        with open(LIBRARY_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(library_catalog, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Catalogo biblioteca salvato: {len(library_catalog)} libri")
    except Exception as e:
        print(f"[ERROR] Errore salvataggio catalogo: {e}")

def clean_title(title):
    """Pulisce il titolo da prefissi e caratteri indesiderati"""
    if not title:
        return title
    
    # Rimuovi prefissi comuni
    prefixes = ['BIOGRAFIA', 'NARRATIVA', 'SAGGISTICA', 'POESIA', 'FANTASY', 'GIALLO', 'THRILLER']
    for prefix in prefixes:
        if title.upper().startswith(prefix):
            title = title[len(prefix):].strip()
    
    # Pulisci spazi multipli
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def clean_section(section):
    """Pulisce la sezione rimuovendo tab e spazi multipli"""
    if not section:
        return section
    
    section = section.replace('\t', ' ')
    section = re.sub(r'\s+-\s+', ' - ', section)
    section = re.sub(r'\s+', ' ', section)
    return section.strip()

def extract_book_info_with_ai(text_context, isbn, model_name='gemma3:4b'):
    """
    Usa un LLM per estrarre titolo e autore dal contesto testuale
    """
    prompt = f"""Sei un esperto bibliotecario. Analizza questo testo da un catalogo biblioteca.

TESTO DAL CATALOGO:
{text_context}

ISBN DEL LIBRO: {isbn}

COMPITO:
Identifica con MASSIMA PRECISIONE:
1. Il TITOLO COMPLETO del libro
2. Il NOME COMPLETO dell'AUTORE

REGOLE IMPORTANTI:
- Il titolo NON include mai: "NARRATIVA", "BIOGRAFIA", "SAGGISTICA", ecc.
- L'autore è SEMPRE una persona (nome e cognome)
- Cerca il titolo vicino all'ISBN indicato
- NON confondere titoli e autori di libri diversi
- Se non sei sicuro al 100%, scrivi "SCONOSCIUTO"

FORMATO RISPOSTA (usa ESATTAMENTE questo formato):
TITOLO: [solo il titolo del libro]
AUTORE: [solo nome e cognome dell'autore]

Esempi corretti:
TITOLO: 1984
AUTORE: George Orwell

TITOLO: Harry Potter e la Pietra Filosofale
AUTORE: J.K. Rowling

TITOLO: La Divina Commedia
AUTORE: Dante Alighieri"""

    try:
        llm = Ollama(model=model_name, base_url=OLLAMA_URL, temperature=0.1)
        response = llm.invoke(prompt)
        
        print(f"[DEBUG] Risposta AI: {response[:200]}...")
        
        # Parse della risposta
        title = None
        author = None
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('TITOLO:'):
                title = line.replace('TITOLO:', '').strip()
            elif line.startswith('AUTORE:'):
                author = line.replace('AUTORE:', '').strip()
        
        # Pulizia e validazione
        if title and title not in ["SCONOSCIUTO", "Non disponibile", "N/A", ""]:
            title = title.strip('"\'').strip()
            title = clean_title(title)
        else:
            title = None
            
        if author and author not in ["SCONOSCIUTO", "Non disponibile", "N/A", ""]:
            author = author.strip('"\'').strip()
        else:
            author = None
        
        print(f"[DEBUG] Estratto - Titolo: '{title}', Autore: '{author}'")
        return title, author
        
    except Exception as e:
        print(f"[ERROR] Errore AI extraction: {e}")
        return None, None

def extract_library_table_from_pdf(pdf_path, use_ai=True, model_name='gemma3:4b'):
    print(f"\n[INFO] ESTRAZIONE PDF BIBLIOTECA: {pdf_path}")
    print(f"[INFO] Modalità AI: {'ATTIVA' if use_ai else 'DISATTIVA'}")
    print(f"[INFO] Modello: {model_name}")
    
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"[ERROR] Impossibile caricare PDF: {e}")
        return []
    
    full_text = "\n".join([doc.page_content for doc in documents])
    lines = [l.strip() for l in full_text.split('\n')]
    
    books = []
    processed_isbns = set()  # Traccia ISBN già processati
    skip_keywords = ['TITOLO', 'AUTORE', 'ISBN', 'DATA', 'EDIZIONE', 'SEZIONE', 'SCAFFALE', 'PUBBLICAZIONE', 'CATALOGO', 'BIBLIOTECA', 'INDICE', 'GENERALE']
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Trova ISBN completo
        isbn_match = re.search(r'978-\d{2}-\d{2}-\d{5}-\d', line + (lines[i+1] if i+1 < len(lines) else ""))
        
        if isbn_match:
            isbn = isbn_match.group(0)
            
            # Salta se già processato
            if isbn in processed_isbns:
                print(f"[INFO] ISBN {isbn} già processato, salto...")
                i += 1
                continue
            
            processed_isbns.add(isbn)
            
            # Estrai contesto più ampio (25 righe prima e 12 dopo)
            context_lines = []
            for j in range(max(0, i - 25), min(len(lines), i + 12)):
                if lines[j].strip() and not any(kw in lines[j].upper() for kw in skip_keywords):
                    context_lines.append(lines[j].strip())
            
            context_text = "\n".join(context_lines)
            
            # Usa AI per estrarre titolo e autore
            if use_ai:
                print(f"\n[INFO] Analisi AI per ISBN: {isbn}")
                title, author = extract_book_info_with_ai(context_text, isbn, model_name)
                
                # Validazione: se AI fallisce, salta questo libro
                if not title or not author:
                    print(f"[WARNING] AI non ha trovato dati validi per ISBN {isbn}, salto questo libro")
                    i += 1
                    continue
            else:
                print(f"[WARNING] Modalità AI disattivata, salto estrazione per ISBN {isbn}")
                i += 1
                continue
            
            # Validazione finale
            if not title or len(title) < 2:
                print(f"[WARNING] Titolo non valido per ISBN {isbn}, salto")
                i += 1
                continue
            
            # Estrai metadati aggiuntivi
            year = None
            edition = None
            section = None
            shelf = None
            
            for j in range(i + 2, min(len(lines), i + 15)):
                candidate = lines[j].strip()
                
                if not year and re.match(r'^\d{4}$', candidate):
                    year = candidate
                
                if not edition:
                    for pub in ['Feltrinelli', 'Mondadori', 'Bompiani', 'Rizzoli', 'Salani', 'TEA', 'E/O', 'Einaudi', 'Adelphi']:
                        if pub in candidate:
                            edition = candidate.replace(',', '').strip()
                            break
                
                if not section and re.match(r'^[A-Z]\s*-\s*.+', candidate):
                    section = clean_section(candidate)
                
                if not shelf and re.match(r'^[A-Z]\d+-\d+$', candidate):
                    shelf = candidate
            
            book = {
                'title': clean_title(title),
                'author': author if author else "Autore sconosciuto",
                'isbn': isbn,
                'year': year,
                'edition': edition,
                'section': section,
                'shelf': shelf
            }
            
            books.append(book)
            print(f"[INFO] ✓ '{book['title']}' di '{book['author']}'")
            
            i += 8  # Salta meno righe per catturare tutti i libri
        else:
            i += 1
    
    print(f"\n[INFO] Totale libri estratti: {len(books)}")
    return books

def extract_poems_from_text(text):
    poems = []
    lines = text.split('\n')
    poem_starts = []
    for i, line in enumerate(lines):
        if line.strip().startswith('Filastrocca '):
            poem_starts.append(i)
    for i, start_idx in enumerate(poem_starts):
        if i < len(poem_starts) - 1:
            end_idx = poem_starts[i + 1]
        else:
            end_idx = len(lines)
        poem_lines = lines[start_idx:end_idx]
        poem_text = '\n'.join(poem_lines).strip()
        while poem_text.endswith('\n'):
            poem_text = poem_text[:-1]
        if len(poem_lines) > 2 and len(poem_text) > 50:
            poems.append(poem_text)
    return poems

def is_library_query(question):
    question_lower = question.lower().strip()
    
    if re.search(r'\b978[-\s]?\d', question_lower):
        return True
    
    explicit_patterns = [
        r'\b(dove|dov\'è)\s+(si\s+)?trova',
        r'\b(cerco|cerca|trovare)\s+.*(libro|testo)',
        r'\b(c\'è|ci\s+sta|presente|disponibile)',
        r'\bscaffale',
        r'\blibro\s+(di|intitolato|chiamato)',
        r'\blibri\s+di'
    ]
    
    for pattern in explicit_patterns:
        if re.search(pattern, question_lower):
            return True
    
    famous_authors = ['dante', 'manzoni', 'orwell', 'tolkien', 'rowling', 'eco', 'ferrante', 'asimov', 'gibson', 'austen', 'dostoevskij', 'baudelaire', 'harari', 'bryson', 'isaacson', 'brown']
    
    has_author = any(author in question_lower for author in famous_authors)
    if has_author and ('libro' in question_lower or 'scaffale' in question_lower):
        return True
    
    return False

def search_in_library(question):
    global library_catalog
    if not library_catalog:
        load_library_catalog()
    
    if not library_catalog:
        return None
    
    question_lower = question.lower().strip()
    
    question_clean = re.sub(r'[?!.,;:\']', ' ', question_lower)
    stop_words = ['libro', 'libri', 'dove', 'si', 'trova', 'cerco', 'presente', 'c\'è', 'ce', 'ci', 'sta', 'il', 'lo', 'la', 'un', 'una', 'di', 'da', 'del', 'della', 'autore', 'titolo', 'in', 'che', 'sezione', 'scaffale', 'dei', 'delle', 'gli', 'le', 'e']
    
    search_words = [w for w in question_clean.split() if len(w) > 2 and w not in stop_words]
    
    if not search_words:
        return None
    
    best_match = None
    best_score = 0
    
    for book in library_catalog:
        title_lower = book['title'].lower()
        author_lower = book['author'].lower()
        
        score = 0
        
        for word in search_words:
            if word in title_lower:
                score += 10
            if word in author_lower:
                score += 5
        
        if score > best_score:
            best_score = score
            best_match = book
    
    if best_match and best_score >= 10:
        return [best_match]
    
    return None

def format_library_response(books, question):
    if not books:
        return "Mi dispiace, non ho trovato questo libro nel catalogo della biblioteca."
    
    book = books[0]
    question_lower = question.lower()
    
    if any(w in question_lower for w in ['dove', 'scaffale', 'posizione', 'sezione']):
        if book.get('shelf'):
            return f"Scaffale {book['shelf']}"
        else:
            return "Scaffale non disponibile per questo libro."
    
    if any(w in question_lower for w in ['c\'è', 'ce', 'ci sta', 'presente', 'disponibile']):
        return f"Sì, '{book['title']}' è presente."
    
    if 'autore' in question_lower:
        return book['author']
    
    response = f"📚 {book['title']}\n✍️ Autore: {book['author']}"
    if book.get('shelf'):
        response += f"\n📍 Scaffale: {book['shelf']}"
    return response

def is_robot_command(question):
    command_keywords = ['avanti', 'dietro', 'destra', 'sinistra']
    question_lower = question.lower().strip()
    return any(cmd in question_lower for cmd in command_keywords)

def is_specific_poem_request(question):
    question_lower = question.lower().strip()
    if 'filastrocca' in question_lower:
        if any(word in question_lower for word in ['corta', 'matta', 'luna', 'animali', 'della', 'del', 'sui', 'delle']):
            return True
    return False

def get_specific_poem(question):
    global vectorstore
    if vectorstore is None:
        vectorstore = load_vectorstore()
    if vectorstore is None:
        return None
    
    try:
        docs = vectorstore.similarity_search(question, k=10)
        poem_docs = [doc for doc in docs if 'Filastrocca' in doc.page_content[:100]]
        
        if poem_docs:
            return poem_docs[0].page_content
        return None
    except Exception as e:
        print(f"[ERROR] Errore get_specific_poem: {e}")
        return None

def get_random_poem():
    global vectorstore
    if vectorstore is None:
        vectorstore = load_vectorstore()
    if vectorstore is None:
        return None
    try:
        all_docs = vectorstore.similarity_search("filastrocca", k=200)
        poem_docs = [doc for doc in all_docs if doc.page_content.strip().startswith('Filastrocca ')]
        if not poem_docs:
            all_docs = vectorstore.similarity_search("", k=200)
            poem_docs = [doc for doc in all_docs if 'Filastrocca' in doc.page_content[:100]]
        if not poem_docs:
            return None
        random_poem = random.choice(poem_docs)
        return random_poem.page_content
    except Exception as e:
        print(f"[ERROR] Errore get_random_poem: {e}")
        return None

OLLAMA_URL = "http://127.0.0.1:11434"

def get_concise_response(question, model_name):
    try:
        chat_llm = ChatOllama(model=model_name, base_url=OLLAMA_URL, temperature=0.3)
        system_prompt = "Sei un assistente AI che fornisce risposte brevi, precise e dirette. Rispondi sempre in italiano. Massimo 2-3 frasi per risposta."
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        response = chat_llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        llm = Ollama(model=model_name, base_url=OLLAMA_URL, temperature=0.3)
        response = llm.invoke(f"Rispondi in modo breve in italiano:\n{question}")
        return response.strip()

def ask_question(question, model_name='gemma3:4b', system_message=None):
    global vectorstore, last_used_model
    
    if model_name != last_used_model:
        last_used_model = model_name
    
    q = str(question).strip() if question else ""
    if not q:
        return ""
    
    if is_robot_command(q):
        return f"comando riconosciuto: {q}"
    
    if is_specific_poem_request(q):
        specific_poem = get_specific_poem(q)
        if specific_poem:
            return specific_poem
        return "Mi dispiace, non ho trovato questa filastrocca."
    
    if 'filastrocca' in q.lower() and any(w in q.lower() for w in ['dici', 'recita', 'dimmi', 'racconta']):
        random_poem = get_random_poem()
        if random_poem:
            return random_poem
        return "Mi dispiace, non ho trovato filastrocche."
    
    if is_library_query(q):
        books = search_in_library(q)
        return format_library_response(books, q)
    
    return get_concise_response(question, model_name)

def ingest_pdfs(pdf_paths, use_ai_extraction=True, model_name='gemma3:4b'):
    """
    Parametri:
    - pdf_paths: lista di percorsi PDF
    - use_ai_extraction: True per usare AI, False per saltare
    - model_name: modello Ollama da usare per l'estrazione AI (default: gemma3:4b)
    """
    global vectorstore, library_catalog
    
    all_documents = []
    
    for pdf_path in pdf_paths:
        print(f"\n[INFO] === Elaborazione PDF: {pdf_path} ===")
        
        books = extract_library_table_from_pdf(pdf_path, use_ai=use_ai_extraction, model_name=model_name)
        
        if books:
            print(f"[INFO] ✓ PDF riconosciuto come CATALOGO BIBLIOTECA")
            print(f"[INFO] Libri estratti: {len(books)}")
            
            library_catalog.extend(books)
            save_library_catalog()
            
            for book in books:
                book_text = f"Libro: {book['title']}\nAutore: {book['author']}\nISBN: {book['isbn']}"
                if book.get('section'):
                    book_text += f"\nSezione: {book['section']}"
                if book.get('shelf'):
                    book_text += f"\nScaffale: {book['shelf']}"
                
                doc = Document(page_content=book_text, metadata={'source': pdf_path, 'type': 'library_book', 'title': book['title'], 'isbn': book['isbn']})
                all_documents.append(doc)
        else:
            print(f"[INFO] PDF generico - caricamento standard")
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                print(f"[INFO] Caricate {len(documents)} pagine")
                
                full_text = "\n".join([doc.page_content for doc in documents])
                poems = extract_poems_from_text(full_text)
                if poems:
                    for i, poem in enumerate(poems):
                        first_line = poem.split('\n')[0]
                        title = first_line.replace('Filastrocca ', '').strip()
                        doc = Document(page_content=poem, metadata={'source': pdf_path, 'poem_index': i, 'title': title, 'type': 'poem'})
                        all_documents.append(doc)
                        print(f"[INFO] Memorizzata filastrocca: {title}")
                else:
                    all_documents.extend(documents)
            except Exception as e:
                print(f"[ERROR] Errore caricamento PDF: {e}")
    
    print(f"\n[INFO] Totale documenti da indicizzare: {len(all_documents)}")
    
    if all_documents and len(all_documents) > 0:
        try:
            if vectorstore is None:
                print("[INFO] Creazione nuovo vectorstore...")
                vectorstore = FAISS.from_documents(all_documents, embedding)
            else:
                print("[INFO] Aggiunta documenti al vectorstore esistente...")
                vectorstore.add_documents(all_documents)
            
            vectorstore.save_local(VECTOR_DIR)
            print("[INFO] ✓ Vectorstore salvato con successo")
        except Exception as e:
            print(f"[ERROR] Errore salvataggio vectorstore: {e}")
            import traceback
            traceback.print_exc()
    
    return len(all_documents), all_documents

def get_indexed_chunks():
    if vectorstore is None:
        return []
    docs = vectorstore.similarity_search("", k=100)
    return [doc.page_content for doc in docs]

def clear_vectorstore():
    global vectorstore, library_catalog
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectorstore = None
    library_catalog = []
    if os.path.exists(LIBRARY_DATA_FILE):
        os.remove(LIBRARY_DATA_FILE)

vectorstore = load_vectorstore()
load_library_catalog()