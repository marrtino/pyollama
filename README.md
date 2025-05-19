# marrtino_chatbot

sudo apt install python3.12-venv
sudo apt install python3-pip

# per creare enviroment
python3 -m venv myenv

# per attivarlo
source myenv/bin/activate
pip3 --version
@ installazione librerie
pip3 install -r requirements.txt


# requirement
Flask
openai
python-aiml
requests
gtts
websockets
vosk
sounddevice
soundfile
ollama
# per eseguire vosk all'interno del docker installare

pip install vosk sounddevice requests
sudo apt install python3-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libavdevice-dev

pip install psutil matplotlib
# for testing
http://x.x.x.x:5000/
http://x.x.x.x:5000/queryexample?query=ciao
http://127.0.0.1:5000/bot?query=ciao

# open firewall
sudo ufw allow 5000

# prerequisiti per lo speech
sudo apt update
sudo apt install sox ffmpeg -y
sudo apt-get install libsox-fmt-all -y
sudo apt-get install portaudio19-dev -y


# genius.py
# ----------------------------------------------------
# Progetto completo con un chatbot generativo affiancato dall'analisi del sentiment, utilizzando Flask per l'interfaccia web e modelli pre-addestrati per la  generazione del testo e l'analisi del sentiment.

transformers
datasets

# prerequisiti per Whisper

pip install git+https://github.com/openai/whisper.git 
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install wavio

sudo usermod -aG audio $USER


# CHATBOT
pip install transformers torch spacy vaderSentiment
python -m spacy download it_core_news_sm  # Per l'italiano
-

https://huggingface.co/iGeniusAI/Italia-9B-Instruct-v0.1
git clone https://huggingface.co/iGeniusAI/Italia-9B-Instruct-v0.1


# FUNZIONAMENTO ASR
# ----------------------------------------------

al di fuori del docker viene eseguito marrtinorobot2_chatbot/marrtinorobot2_chatbot/talk2http.py che comunica
        marrtinorobot2_chatbot/node_asr.py sulla porta 5002
        marrtinorobot2_chatbot/chatbot.py sulla porta 5500

configurazione marrtinorobot2_chatbot/marrtinorobot2_chatbot/audioconfig.py
