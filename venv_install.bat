cd /d %~dp0

call python -m venv venv
call venv\Scripts\activate.bat 
git clone https://github.com/ai-forever/ru-dalle.git
python.exe -m pip install --upgrade pip
pip install torch==2.0.0
pip install Cython==0.29.33
pip install ruclip==0.0.2
pip install huggingface-hub==0.13.3
pip install -r requirements.txt
move "ru-dalle\rudalle" "rudalle"
cmd.exe


