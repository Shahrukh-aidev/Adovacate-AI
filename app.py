from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from haq_engine import ask_haq

app = Flask(__name__)
CORS(app)

# Serve the web app
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# API endpoint — receives question, returns answer
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        answer = ask_haq(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("HAQ Legal AI Server is running!")
    print("Open your browser and go to:")
    print("http://localhost:5000")
    print("=" * 50)
    import os
port = int(os.environ.get("PORT", 5000))
app.run(port=port, debug=False, host='0.0.0.0')
