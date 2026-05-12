import sys
import io
import os
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import pdfplumber
from werkzeug.utils import secure_filename

# Set stdout to UTF-8 for Windows compatibility
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load text generation model once at startup
print("Loading AI model...")
generator = pipeline(
    "text-generation",
    model="gpt2"
)
print("Model loaded successfully.")

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
            os.remove(filepath)
            
            if not text.strip():
                return jsonify({'error': 'Could not extract text from PDF'}), 400

            # Reduce text size for GPT-2 context window
            text_preview = text[:1000]
            
            # Generate summary
            prompt = f"Summarize this text:\n{text_preview}"
            result = generator(
                prompt,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=50256
            )
            
            summary = result[0]['generated_text']
            # Optionally remove the prompt from the output if it's included
            if summary.startswith(prompt):
                summary = summary[len(prompt):].strip()
            
            return jsonify({'summary': summary})
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400

if __name__ == '__main__':
    app.run(debug=True)