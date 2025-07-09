from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import worker
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process-document', methods=['POST'])
def process_doc():
    if 'file' not in request.files:
        return jsonify({"botResponse": "Please upload a PDF file."}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    worker.process_document(file_path)

    return jsonify({"botResponse": "Document processed. Ask your question!"})

@app.route('/process-message', methods=['POST'])
def ask_question():
    user_message = request.json.get("userMessage", "")
    response = worker.process_prompt(user_message)
    return jsonify({"botResponse": response})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
