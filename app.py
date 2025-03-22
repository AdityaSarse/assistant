from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from ctransformers import AutoModelForCausalLM
import PyPDF2
import traceback

app = Flask(__name__, static_folder='static')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists("models"):
    os.makedirs("models")

# Initialize Llama model
try:
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        model_type="llama",
        max_new_tokens=512,
        context_length=2048,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2
    )
except FileNotFoundError:
    print(f"Please download the Llama 2 model and place it at {MODEL_PATH}")
    print("You can download it from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = []
            for page in reader.pages:
                try:
                    text.append(page.extract_text())
                except Exception as e:
                    print(f"Error extracting page: {str(e)}")
                    continue
            return "\n".join(text).strip()
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        traceback.print_exc()
        return None

def extract_text_from_txt(file_path):
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading TXT with {encoding}: {str(e)}")
            continue
    print("Failed to read file with any encoding")
    return None

def get_llama_response(question, context):
    # Create a prompt that includes both the context and question
    prompt = f"""Context: {context}

Question: {question}

Answer: Let me help you with that question based on the provided context."""

    # Generate response from Llama 2
    answer = llm(prompt)

    # Calculate a simple confidence score based on answer length and coherence
    words = answer.split()
    confidence = min(1.0, len(words) / 100)  # Simple heuristic: longer answers up to 100 words are more confident
    
    return answer.strip(), confidence

@app.route('/')
def serve_static():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = None
        try:
            # Get file extension
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ['.pdf', '.txt']:
                return jsonify({'error': 'Only PDF and TXT files are supported'}), 400

            # Save file with a unique name
            filename = os.path.join(UPLOAD_FOLDER, f"upload_{os.urandom(8).hex()}{ext}")
            file.save(filename)
            
            # Extract text based on file type
            if ext == '.pdf':
                text = extract_text_from_pdf(filename)
            else:  # .txt
                text = extract_text_from_txt(filename)
            
            # Validate extracted text
            if text is None:
                return jsonify({'error': 'Failed to extract text from file'}), 500
            
            text = text.strip()
            if not text:
                return jsonify({'error': 'No text content found in file'}), 400
            
            # Limit text length if needed
            if len(text) > 50000:  # Limit to ~50KB of text
                text = text[:50000] + "..."
            
            return jsonify({
                'message': 'File uploaded successfully',
                'text': text
            })
        except Exception as e:
            print(f"Upload error: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up the uploaded file
            if filename and os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
        
    if 'context' not in data:
        return jsonify({'error': 'No context provided'}), 400

    if not data['context'].strip():
        return jsonify({'error': 'Empty context'}), 400

    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                'error': 'Llama 2 model not found. Please download it first.'
            }), 500

        answer, confidence = get_llama_response(data['question'], data['context'])
        
        if not answer.strip():
            return jsonify({'error': 'Model returned empty response'}), 500
        
        return jsonify({
            'answer': answer,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Question error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
