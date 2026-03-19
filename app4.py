# App.py - VERSIONE CORRETTA CON AI EXTRACTION E GEMMA3:4B
##################################################
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory
from rag.rag_chain4 import ask_question, ingest_pdfs, get_indexed_chunks, clear_vectorstore
from rag.pdf_manager import save_pdf, list_pdfs, delete_pdf
import os
import subprocess
import time
from fpdf import FPDF
import sys

app = Flask(__name__)
app.secret_key = 'supersegretissimo'
UPLOAD_FOLDER = 'data/pdfs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

##################################
# Funzioni ROS2
##################################

def send_to_ros2(text, url="http://10.3.1.1:5001/send"):
    try:
        import requests
        params = {"text": text}
        requests.get(url, params=params)
        print(f"[DEBUG] ✅ Inviato al nodo ROS: {text}", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] ❌ Errore nell'invio al nodo ROS: {e}", file=sys.stderr)

#############################
## EOF
#############################

def split_string(msg):
    print(f"[DEBUG] Risposta grezza del modello: {msg}", file=sys.stderr)
    if not isinstance(msg, str):
        return "(errore nella risposta del modello)"
    return msg

def get_installed_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        models = []
        for line in lines[1:]:  # Salta l'intestazione
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except subprocess.CalledProcessError as e:
        print(f"Errore nell'esecuzione di 'ollama list': {e}")
        return ['gemma3:4b', 'mistral']  # Fallback

@app.route("/get")
def get_bot_response():
    question = request.args.get('msg')
    model_name = request.args.get('model', "gemma3:4b")
    print(f"[DEBUG] Messaggio ricevuto dal client: {question}", file=sys.stderr)
    print(f"[DEBUG] Modello selezionato: {model_name}", file=sys.stderr)
       
    start_time = time.time()
    answer = ask_question(question, model_name)
    duration = round(time.time() - start_time, 2)
    msgout = split_string(answer)
    return msgout

@app.route('/json')
def json_response():
    question = request.args.get('query')
    model_name = request.args.get('model', "gemma3:4b")
    print(f"[DEBUG] Richiesta /json ricevuta: {question}", file=sys.stderr)
    print(f"[DEBUG] Modello selezionato: {model_name}", file=sys.stderr)

    answer = ask_question(question, model_name)

    # Torna SEMPRE un oggetto JSON con la chiave 'response'
    return jsonify({"response": answer, "action": "ok"})

#########################
# PAGINA PRINCIPALE - SOLO LLM
#########################
@app.route('/')
def index():
    pdf_files = list_pdfs()
    chunks = get_indexed_chunks()
    models = get_installed_models()
    return render_template('index.html', pdf_files=pdf_files, chunks=chunks, models=models)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    print(f"[DEBUG] Richiesta ricevuta: {question}", file=sys.stderr)
    model_name = request.form.get('model', 'gemma3:4b')
    system_message = request.form.get('system_message', '').strip()

    try:
        start_time = time.time()
        answer = ask_question(question, model_name)
        duration = round(time.time() - start_time, 2)
        return jsonify({'answer': answer, 'time': duration})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/clear_chunks', methods=['POST'])
def clear_chunks():
    clear_vectorstore()
    return redirect(url_for('index'))

@app.route("/chunks")
def chunks():
    chunks = get_indexed_chunks()
    return render_template("chunks.html", chunks=chunks)

#########################
# GESTIONE PDF -> /manage
#########################
@app.route('/manage', methods=['GET'])
def manage():
    pdf_files = list_pdfs()
    models = get_installed_models()
    return render_template('manage.html', pdf_files=pdf_files, log_messages=[], models=models)

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist('pdfs')
    paths = []
    logs = []
    
    # Ottieni parametri per l'estrazione AI - DEFAULT: gemma3:4b
    model_name = request.form.get('model', 'gemma3:4b')
    use_ai = request.form.get('use_ai', 'true').lower() == 'true'

    logs.append(f"[CONFIG] Modalità AI: {'ATTIVA' if use_ai else 'DISATTIVA'}")
    logs.append(f"[CONFIG] Modello utilizzato: {model_name}")

    for file in uploaded_files:
        if file.filename.endswith('.pdf'):
            try:
                saved_path = save_pdf(file)
                logs.append(f"[UPLOAD] ✓ Caricato: {file.filename}")
                paths.append(saved_path)
            except Exception as e:
                logs.append(f"[ERROR] ❌ Errore nel caricamento di {file.filename}: {str(e)}")
                print(f"[ERROR] {e}", file=sys.stderr)

    if paths:
        try:
            logs.append(f"[INFO] Inizio elaborazione con {model_name}...")
            
            num_chunks, chunks = ingest_pdfs(paths, use_ai_extraction=use_ai, model_name=model_name)
            
            logs.append(f"[SUCCESS] ✓ Totale documenti indicizzati: {num_chunks}")
            
            # Mostra primi 3 chunks come esempio
            if chunks and len(chunks) > 0:
                logs.append("[INFO] Primi libri indicizzati:")
                for i, chunk in enumerate(chunks[:3]):
                    if hasattr(chunk, 'page_content'):
                        content_preview = chunk.page_content[:100].replace('\n', ' ')
                        logs.append(f"  • {content_preview}...")
            else:
                logs.append("[INFO] Nessun chunk da visualizzare")
                
        except Exception as e:
            logs.append(f"[ERROR] ❌ Errore durante l'ingest: {str(e)}")
            print(f"[ERROR] Traceback completo:", file=sys.stderr)
            import traceback
            traceback.print_exc()

    pdf_files = list_pdfs()
    models = get_installed_models()
    return render_template("manage.html", pdf_files=pdf_files, log_messages=logs, models=models)

@app.route('/delete_pdf/<filename>', methods=['POST'])
def delete_pdf_route(filename):
    delete_pdf(filename)
    logs = [f"[DELETE] ✓ Rimosso: {filename}"]
    pdf_files = list_pdfs()
    models = get_installed_models()
    return render_template("manage.html", pdf_files=pdf_files, log_messages=logs, models=models)

@app.route('/pdfs/<filename>')
def serve_pdf(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/export_chunks')
def export_chunks():
    chunks = get_indexed_chunks()
    response = "\n\n".join(chunks)
    return response, 200, {
        'Content-Type': 'text/plain; charset=utf-8',
        'Content-Disposition': 'attachment; filename=chunks_export.txt'
    }

@app.route('/search_chunks', methods=['GET'])
def search_chunks():
    query = request.args.get('q', '').lower()
    chunks = get_indexed_chunks()
    results = [c for c in chunks if query in c.lower()]
    return render_template("chunks.html", chunks=results, query=query)

@app.route('/export_chunks_pdf')
def export_chunks_pdf():
    chunks = get_indexed_chunks()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=10)
    for chunk in chunks:
        pdf.multi_cell(0, 10, chunk + "\n")
    pdf_output = os.path.join("data", "export_chunks.pdf")
    pdf.output(pdf_output)
    return send_from_directory("data", "export_chunks.pdf", as_attachment=True)

#########################
# AVVIO APP
#########################
if __name__ == '__main__':
    print("=" * 60)
    print("ChatBot RAG con PyOllama v.1.03")
    print("AI Extraction: ATTIVA (gemma3:4b)")
    print("=" * 60)
    print(" ")
    myip = '0.0.0.0'
    app.run(host=myip, debug=True, port=8060)