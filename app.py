# App.py
##################################################
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory
from rag.rag_chain import ask_question, ingest_pdfs, get_indexed_chunks, clear_vectorstore
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
        return []



def get_response(messages: list, model_name="gemma3:4b"):
    print(f"[DEBUG] Messaggi inviati al modello ({model_name}): {messages}", file=sys.stderr)
    try:
        start_time = time()
        response = ollama_client.chat(
            model=model_name,
            messages=messages
        )
        elapsed_time = time() - start_time
        print(f"[DEBUG] Risposta completa ricevuta: {response}", file=sys.stderr)
        print(f"[DEBUG] Tempo di risposta del modello: {elapsed_time:.2f} secondi", file=sys.stderr)
        return response['message']
    except Exception as e:
        print(f"[DEBUG] Errore nella chiamata a Ollama: {e}", file=sys.stderr)
        return {"content": f"(errore: {str(e)})"}

@app.route("/get")
def get_bot_response():
    question = request.args.get('msg')
    model_name = request.args.get('model', "gemma3:4b")
    print(f"[DEBUG] Messaggio ricevuto dal client: {question}", file=sys.stderr)
    print(f"[DEBUG] Modello selezionato: {model_name}", file=sys.stderr)
    # messages = [
    #     {"role": "system", "content": PROMPT_SYSTEM},
    #     {"role": "user", "content": myquery}
    # ]
    # Ignora il system_message dell'utente, usa quello interno ottimizzato
       
    start_time = time.time()
    # Ignora il system_message dell'utente, usa quello interno ottimizzato
    answer = ask_question(question, model_name)
    duration = round(time.time() - start_time, 2)
    msgout = split_string(answer)
    #log_to_file(myquery, msgout)
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
    print(f"[DEBUG] Richiesta  ricevuta: {question}", file=sys.stderr)
    model_name = request.form.get('model', 'mistral')
    system_message = request.form.get('system_message', '').strip()

    try:
        start_time = time.time()
        # Ignora il system_message dell'utente, usa quello interno ottimizzato
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
    return render_template('manage.html', pdf_files=pdf_files, log_messages=[])

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist('pdfs')
    paths = []
    logs = []

    for file in uploaded_files:
        if file.filename.endswith('.pdf'):
            saved_path = save_pdf(file)
            logs.append(f"[UPLOAD] Caricato: {file.filename}")
            paths.append(saved_path)

    if paths:
        num_chunks, chunks = ingest_pdfs(paths)
        logs.append(f"[INGEST] Totale chunk indicizzati: {num_chunks}")
        for i, chunk in enumerate(chunks[:5]):
            logs.append(f"[CHUNK {i}] {chunk.page_content[:80]}...")

    pdf_files = list_pdfs()
    return render_template("manage.html", pdf_files=pdf_files, log_messages=logs)

@app.route('/delete_pdf/<filename>', methods=['POST'])
def delete_pdf_route(filename):
    delete_pdf(filename)
    logs = [f"[DELETE] Rimosso: {filename}"]
    pdf_files = list_pdfs()
    return render_template("manage.html", pdf_files=pdf_files, log_messages=logs)

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
    print("ChatBot with PyOllama v.1.01")
    print(" ")
    myip = '0.0.0.0'
    app.run(host=myip, debug=True, port=8060)
