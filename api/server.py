# api/server.py
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from app.query import ask

load_dotenv()
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
