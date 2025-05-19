#!/usr/bin/python3
import os
from flask import Flask, render_template, request, jsonify
import json
from threading import Thread
from datetime import datetime
from time import time
from ollama import Client
import sys
import requests

# Imposta il path principale
PATH = os.path.expandvars("$HOME/src/marrtinorobot2/marrtinorobot2_chatbot/")
LOG_PATH = os.path.join(PATH, "log")
os.makedirs(LOG_PATH, exist_ok=True)

# Inizializza Ollama
ollama_client = Client(host='http://localhost:11434')

# Funzioni di utilità
def log_to_file(question, bot_answer):
    now = datetime.now()
    data_ora = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(os.path.join(LOG_PATH, "log.txt"), "a") as log_file:
        report = data_ora + "\n" + \
            "[QUESTION]:   " + question + ";" + \
            "[OLLAMA]: " + bot_answer
        log_file.write(report + "\n")

    if bot_answer != "":
        with open(os.path.join(LOG_PATH, "user.txt"), "a") as bot_file:
            bot_file.write("user: " + question + "\n")
            bot_file.write("bot: " + bot_answer + "\n")

def split_string(msg):
    print(f"[DEBUG] Risposta grezza del modello: {msg}", file=sys.stderr)
    if not isinstance(msg, str):
        return "(errore nella risposta del modello)"
    return msg

def send_to_ros2(text, url="http://localhost:5001/send"):
    try:
        params = {"text": text}
        requests.get(url, params=params)
        print(f"[DEBUG] ✅ Inviato al nodo ROS: {text}", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] ❌ Errore nell'invio al nodo ROS: {e}", file=sys.stderr)


def get_response(messages: list, model_name="gemma:2b"):
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

# Crea l'app Flask
app = Flask(__name__)
app.static_folder = 'static'

PROMPT_SYSTEM = (
    "Sei MARRtino, un robot sociale italiano, simpatico e birichino. "
    "Quando qualcuno ti fa una domanda personale o sulla tua origine, "
    "rispondi in modo coerente con la tua identità.\n"
    "Chi ti ha creato? Robotics-3D.\n"
    "Chi è Smarrtino? Smarrtino è un robot birichino creato dalla collaborazione "
    "fra Robotics-3D e i ricercatori dell'Università La Sapienza di Roma."
)



@app.route("/")
def home():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        model_list = response.json().get("models", [])
        model_names = [model["name"] for model in model_list]
        print(f"[DEBUG] Modelli disponibili: {model_names}", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] Errore durante il recupero modelli: {e}", file=sys.stderr)
        model_names = ["codellama:7b"]  # fallback
    return render_template("indexollama.html", models=model_names)



@app.route("/get")
def get_bot_response():
    myquery = request.args.get('msg')
    model_name = request.args.get('model', "gemma:2b")
    print(f"[DEBUG] Messaggio ricevuto dal client: {myquery}", file=sys.stderr)
    print(f"[DEBUG] Modello selezionato: {model_name}", file=sys.stderr)
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": myquery}
    ]
    new_message = get_response(messages, model_name)
    msgout = split_string(new_message['content'])
    log_to_file(myquery, msgout)
    return msgout

@app.route('/bot')
def bot():
    myquery = request.args.get('query')
    model_name = request.args.get('model', "gemma:2b")
    print(f"[DEBUG] Richiesta /bot ricevuta: {myquery}", file=sys.stderr)
    print(f"[DEBUG] Modello selezionato: {model_name}", file=sys.stderr)
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": myquery}
    ]
    new_message = get_response(messages, model_name)
    msgout = split_string(new_message['content'])
    log_to_file(myquery, msgout)
    send_to_ros2(msgout)

    return msgout

@app.route('/json')
def json_response():
    myquery = request.args.get('query')
    model_name = request.args.get('model', "gemma:2b")
    print(f"[DEBUG] Richiesta /json ricevuta: {myquery}", file=sys.stderr)
    print(f"[DEBUG] Modello selezionato: {model_name}", file=sys.stderr)
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": myquery}
    ]
    new_message = get_response(messages, model_name)
    msg = new_message['content']
    msgjson = {
        "response": msg,
        "action": "ok"
    }
    return jsonify(msgjson)

if __name__ == '__main__':
    print("ChatBot with Ollama v.1.01")
    myip = '0.0.0.0'
    app.run(host=myip, debug=True, port=5000)
