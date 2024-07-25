from flask import Flask, send_from_directory
import os
import webbrowser
from threading import Timer

app = Flask(__name__)
DIRECTORY = 'custom_app'
PORT = 8000

@app.route('/')
def serve_index():
    return send_from_directory(DIRECTORY, 'index.html')

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory(DIRECTORY, path)

def open_browser():
    webbrowser.open_new_tab(f'http://localhost:{PORT}')

if __name__ == '__main__':
    Timer(.5, open_browser).start()
    app.run(port=PORT, debug=True)
