# api/server.py
import os
from flask import Flask, request, jsonify
from app.query import ask
from app.ingest import ingest_document
import werkzeug

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def query_rag():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question' in request"}), 400

    try:
        answer = ask(question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ingest', methods=['POST'])
def ingest_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        temp_file_path = f"temp_{werkzeug.utils.secure_filename(file.filename)}"
        file.save(temp_file_path)
        
        result = ingest_document(temp_file_path)
        os.remove(temp_file_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
