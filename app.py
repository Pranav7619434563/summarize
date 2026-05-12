import os
import requests
from flask import Flask, render_template, request, jsonify
import pdfplumber
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'  # Vercel allows writing to /tmp
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Hugging Face API Configuration
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# Get token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

def query_huggingface(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read PDF
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
            
            # Clean up file after reading
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if not text.strip():
                return jsonify({'error': 'Could not extract text from PDF'}), 400

            # BART model works best with up to 1024 tokens
            text_input = text[:3000] 
            
            # Call Hugging Face API
            output = query_huggingface({
                "inputs": text_input,
                "parameters": {"max_length": 150, "min_length": 40, "do_sample": False}
            })

            # Handle API responses
            if isinstance(output, list) and len(output) > 0:
                summary = output[0].get('summary_text', 'No summary generated')
                return jsonify({'summary': summary})
            elif isinstance(output, dict) and 'error' in output:
                # If model is loading, tell user to wait
                if "estimated_time" in output:
                    return jsonify({'error': 'Model is starting up on Hugging Face. Please try again in 20 seconds.'}), 503
                return jsonify({'error': output['error']}), 500
            else:
                return jsonify({'error': 'Unexpected response from AI service'}), 500
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400

if __name__ == '__main__':
    app.run(debug=True)