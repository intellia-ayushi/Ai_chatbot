from flask import Flask, request, jsonify, render_template
from ollama import LocalChatbot
import os
from werkzeug.utils import secure_filename
import time
import hashlib
import threading
from queue import Queue
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import gc
from functools import lru_cache
import asyncio
import aiofiles
import tempfile
import pdfplumber
import mimetypes
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Changed to INFO to reduce debug noise
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Reduced to 16MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000
app.config['PROCESSING_TIMEOUT'] = 30  # Reduced timeout
app.config['RESPONSE_TIMEOUT'] = 20    # Reduced timeout
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'doc', 'docx', 'rtf'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize chatbot with fastest model and optimized parameters
chatbot = LocalChatbot(model_name="gemma:2b")

# Global content storage
current_content = None
current_filename = None

# Optimized model parameters for speed
chatbot.model_params = {
    "max_tokens": 512,      # Reduced for faster response
    "temperature": 0.3,     # Lower temperature for faster generation
    "top_p": 0.8,          # Reduced for speed
    "num_ctx": 2048,       # Reduced context window
    "timeout": 15,         # Shorter timeout
    "num_predict": 256,    # Limit prediction length
    "repeat_penalty": 1.1,
    "stop": ["\n\n", "User:", "Human:", "Q:", "Question:"]  # Early stopping
}

# Check if Ollama is running
if not chatbot.check_ollama_connection():
    print("⚠️  Warning: Ollama is not running. Please start Ollama before using the chatbot.")

# Simplified caching system
class SimpleCache:
    def __init__(self, max_size=100):  # Reduced cache size
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest 20% of entries
                to_remove = list(self.cache.keys())[:int(self.max_size * 0.2)]
                for k in to_remove:
                    self.cache.pop(k, None)
            self.cache[key] = value

# Initialize cache
response_cache = SimpleCache(max_size=100)
file_cache = {}

# Reduced thread pool
executor = ThreadPoolExecutor(max_workers=2)

def create_content_hash(content):
    """Create a fast hash of content"""
    return hashlib.md5(content[:1000].encode('utf-8')).hexdigest()[:8]  # Only hash first 1000 chars

def create_question_key(question, content_hash):
    """Create cache key for question-content pair"""
    question_clean = question.lower().strip()[:100]  # Limit question length
    return f"{content_hash}:{hashlib.md5(question_clean.encode()).hexdigest()[:8]}"

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(filepath, max_pages=10):  # Limit pages for speed
    """Extract text from PDF file with page limit"""
    try:
        text = ""
        logger.info(f"Starting PDF extraction from: {filepath}")
        
        with pdfplumber.open(filepath) as pdf:
            total_pages = min(len(pdf.pages), max_pages)  # Limit pages
            logger.info(f"Processing {total_pages} pages (limited)")
            
            for i in range(total_pages):
                try:
                    page = pdf.pages[i]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        # Stop if we have enough content
                        if len(text) > 10000:  # 10KB limit
                            break
                except Exception as page_error:
                    logger.warning(f"Error extracting text from page {i+1}: {page_error}")
                    continue
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text.strip() if text.strip() else None
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None

def smart_chunk_content(content, question="", chunk_size=2000):
    """Smart chunking that prioritizes relevant content"""
    if len(content) <= chunk_size:
        return [content]
    
    # If we have a question, try to find relevant sections first
    if question:
        question_words = set(question.lower().split())
        
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        scored_paragraphs = []
        
        for para in paragraphs:
            if len(para.strip()) < 50:  # Skip very short paragraphs
                continue
            
            # Score paragraph based on question word overlap
            para_words = set(para.lower().split())
            score = len(question_words.intersection(para_words))
            scored_paragraphs.append((score, para))
        
        # Sort by relevance and take top paragraphs
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])
        
        # Build chunk from most relevant paragraphs
        chunk = ""
        for score, para in scored_paragraphs[:5]:  # Top 5 relevant paragraphs
            if len(chunk) + len(para) < chunk_size:
                chunk += para + "\n\n"
            else:
                break
        
        if chunk.strip():
            return [chunk.strip()]
    
    # Fallback to simple chunking
    return [content[:chunk_size]]

@app.route('/')
def index():
    return render_template('index.html')

def process_file_fast(filepath, filename):
    """Fast file processing with size limits"""
    global current_content, current_filename
    
    try:
        # Check file size first
        file_size = os.path.getsize(filepath)
        if file_size > 5 * 1024 * 1024:  # 5MB limit
            logger.warning(f"File too large: {file_size} bytes")
            return None, False, "File too large. Please use files smaller than 5MB."
        
        # Create simple hash
        with open(filepath, 'rb') as f:
            first_chunk = f.read(1024)  # Only read first 1KB for hash
            file_hash = hashlib.md5(first_chunk).hexdigest()[:8]
            
        # Check cache
        if file_hash in file_cache:
            logger.info("Using cached file content")
            current_content = file_cache[file_hash]['content']
            current_filename = filename
            chatbot.content = current_content
            return file_hash, True, None
        
        # Determine file type
        file_extension = filename.lower().split('.')[-1]
        logger.info(f"Processing file: {filename}, extension: {file_extension}")
        
        text_content = None
        
        # Handle PDF files with limits
        if file_extension == 'pdf':
            text_content = extract_text_from_pdf(filepath, max_pages=5)  # Limit to 5 pages
            
        # Handle text files with size limit
        elif file_extension == 'txt':
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read(20000)  # Only read first 20KB
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='latin-1') as f:
                    text_content = f.read(20000)
                    
        # Handle other document types
        else:
            if chatbot.load_file(filepath):
                text_content = chatbot.content[:20000]  # Limit content
        
        if text_content and text_content.strip():
            # Additional size limit
            if len(text_content) > 15000:  # 15KB limit
                text_content = text_content[:15000] + "\n\n[Content truncated for performance...]"
            
            current_content = text_content
            current_filename = filename
            chatbot.content = current_content
            
            # Cache the content
            file_cache[file_hash] = {
                'content': text_content,
                'filename': filename,
                'timestamp': time.time()
            }
            
            logger.info(f"Successfully processed file. Content length: {len(text_content)}")
            return file_hash, False, None
        else:
            return None, False, "No readable content found in file"
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return None, False, str(e)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Fast file upload"""
    global current_content, current_filename
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    uploaded_files=request.files.getlist('file')
    
    # file = request.files['file']
    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'})
    



    processed_files=[]
    combined_content=""
    errors=[]

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)
            filepath=os.parth.join(app.config['UPLOAD_FOLDER'],filename)
            try:
                file.save(filepath)
                file_hash, is_cached, error=process_file_fast(filepath,filename)
                if error:
                    errors.append({file:filename, error:error})
                else:
                    processed_files.appen(filename)
                    combined_content+=current_content+'\n'
            except Exception as e:
                errors.append({file:filename, error:str(e)})
            finally:
                try:
                    os.remove(filepath)
                except:
                    pass
        else:
            errors.append({'file':file.filename, 'error':'unsupported file type'})
    
    current_content=combined_content
    current_filename=", ".join(processed_files)
    duration = round(time.time() - start_time, 2)
    return jsonify({
        'status': 'success',
        'processed_files': processed_files,
        'errors': errors,
        'time_taken': f"{duration} seconds"
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    """Optimized question handling"""
    global current_content
    start_time = time.time()
    
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'No question provided'})
    
    if not current_content:
        return jsonify({'error': 'Please upload a file first'})
    
    # Limit question length
    if len(question) > 200:
        question = question[:200] + "..."
    
    logger.info(f"Processing question: '{question[:50]}...'")
    
    # Create cache key
    content_hash = create_content_hash(current_content)
    cache_key = create_question_key(question, content_hash)
    
    # Check cache
    cached_response = response_cache.get(cache_key)
    if cached_response:
        logger.info("Using cached response")
        return jsonify({
            'response': cached_response,
            'cached': True,
            'response_time': time.time() - start_time
        })
    
    try:
        # Use smart chunking to get most relevant content
        chunks = smart_chunk_content(current_content, question, chunk_size=1500)
        working_content = chunks[0]  # Use most relevant chunk
        
        chatbot.content = working_content
        
        def ask_with_timeout():
            try:
                # Make question very specific and short for faster response
                short_question = f"Answer briefly in 1-2 sentences: {question}"
                
                # Set very aggressive parameters for speed
                chatbot.model_params.update({
                    "max_tokens": 128,      # Very short response
                    "temperature": 0.1,     # Very low for speed
                    "top_p": 0.5,          # Very focused
                    "num_ctx": 512,        # Minimal context
                    "timeout": 8,          # Very short timeout
                    "num_predict": 64,     # Very short prediction
                    "repeat_penalty": 1.0,
                    "stop": [".", "?", "!", "\n"]  # Stop at first sentence
                })
                
                return chatbot.ask_question(short_question)
            except Exception as e:
                logger.error(f"Error in ask_question: {e}")
                return f"Error processing question: {str(e)}"
        
        # Use very short timeout
        future = executor.submit(ask_with_timeout)
        
        try:
            response = future.result(timeout=8)  # 8 second timeout
            
            if not response or response.strip() == "":
                response = "I couldn't find relevant information in the document to answer your question. Please try rephrasing or asking about specific topics mentioned in the text."
            
            # Limit response length for speed
            if len(response) > 1000:
                response = response[:1000] + "..."
            
            # Cache the response
            response_cache.set(cache_key, response)
            
            response_time = time.time() - start_time
            logger.info(f"Generated response in {response_time:.2f} seconds")
            
            return jsonify({
                'response': response,
                'cached': False,
                'response_time': response_time
            })
            
        except TimeoutError:
            logger.error("Question processing timed out after 8 seconds")
            future.cancel()
            
            # Try a simple fallback response
            try:
                # Emergency simple search in content
                question_words = question.lower().split()
                content_lines = working_content.lower().split('\n')
                
                relevant_lines = []
                for line in content_lines:
                    if any(word in line for word in question_words):
                        relevant_lines.append(line.strip())
                        if len(relevant_lines) >= 3:  # Max 3 lines
                            break
                
                if relevant_lines:
                    fallback_response = "Based on a quick search: " + " ".join(relevant_lines)[:200] + "..."
                else:
                    fallback_response = "The model is taking too long. Please try with a simpler question or restart the application."
                    
                return jsonify({
                    'response': fallback_response,
                    'cached': False,
                    'timeout': True,
                    'response_time': time.time() - start_time
                })
            except:
                return jsonify({
                    'error': 'Model timeout. Please restart Ollama service or try a different model.',
                    'timeout': True
                })
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/model', methods=['POST'])
def change_model():
    """Switch model with speed optimization"""
    data = request.get_json()
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({'error': 'No model specified'})
    
    try:
        # Clear cache when switching models
        response_cache.cache.clear()
        
        # Change model
        chatbot.change_model(model_name)
        
        # Set optimized parameters based on model
        if 'gemma:2b' in model_name:
            chatbot.model_params = {
                "max_tokens": 256,
                "temperature": 0.3,
                "top_p": 0.8,
                "num_ctx": 1024,
                "timeout": 10
            }
        else:
            chatbot.model_params = {
                "max_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.8,
                "num_ctx": 2048,
                "timeout": 15
            }
        
        return jsonify({
            'message': f'Switched to model: {model_name}',
            'current_model': chatbot.model_name
        })
    except Exception as e:
        return jsonify({'error': f'Failed to switch model: {str(e)}'})

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        models = chatbot.available_models
        return jsonify({
            'models': models,
            'current_model': chatbot.model_name,
            'recommended': 'gemma:2b'  # Fastest model
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get models: {str(e)}'})

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear caches"""
    try:
        response_cache.cache.clear()
        file_cache.clear()
        gc.collect()
        return jsonify({'message': 'Cache cleared'})
    except Exception as e:
        return jsonify({'error': f'Failed to clear cache: {str(e)}'})

@app.route('/debug', methods=['POST'])
def debug_ollama():
    """Debug Ollama connection and model performance"""
    try:
        debug_info = {}
        
        # Test basic connection
        debug_info['ollama_running'] = chatbot.check_ollama_connection()
        
        # Test simple query
        start_time = time.time()
        try:
            # Very simple test
            test_response = chatbot.ask_question("Hello")
            debug_info['simple_test'] = {
                'success': True,
                'response': test_response[:100] + "..." if len(test_response) > 100 else test_response,
                'time': time.time() - start_time
            }
        except Exception as e:
            debug_info['simple_test'] = {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
        
        # Model info
        debug_info['current_model'] = chatbot.model_name
        debug_info['model_params'] = chatbot.model_params
        
        # System info
        import psutil
        debug_info['system'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory': psutil.virtual_memory().available // (1024*1024)  # MB
        }
        
        return jsonify({'debug_info': debug_info})
        
    except Exception as e:
        return jsonify({'error': f'Debug failed: {str(e)}'})

@app.route('/fix_ollama', methods=['POST'])
def fix_ollama():
    """Try to fix common Ollama issues"""
    try:
        import subprocess
        import requests
        
        fixes_applied = []
        
        # Try to restart Ollama (if running as service)
        try:
            # Check if Ollama is responding
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code != 200:
                fixes_applied.append("Ollama API not responding properly")
        except:
            fixes_applied.append("Ollama connection failed")
        
        # Try to change to an even faster model
        try:
            # List available models and pick the smallest
            models = chatbot.available_models
            if models:
                smallest_model = min(models, key=len)  # Pick shortest name (usually smallest)
                if smallest_model != chatbot.model_name:
                    chatbot.change_model(smallest_model)
                    fixes_applied.append(f"Switched to faster model: {smallest_model}")
        except:
            fixes_applied.append("Could not switch models")
        
        # Clear all caches
        response_cache.cache.clear()
        file_cache.clear()
        gc.collect()
        fixes_applied.append("Cleared all caches")
        
        # Set ultra-aggressive parameters
        chatbot.model_params = {
            "max_tokens": 32,
            "temperature": 0.0,
            "top_p": 0.3,
            "num_ctx": 256,
            "timeout": 5,
            "num_predict": 16
        }
        fixes_applied.append("Applied ultra-fast model parameters")
        
        return jsonify({
            'fixes_applied': fixes_applied,
            'current_model': chatbot.model_name,
            'suggestion': 'Try asking a question now'
        })
        
    except Exception as e:
        return jsonify({'error': f'Fix attempt failed: {str(e)}'})

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get stats"""
    return jsonify({
        'cache_size': len(response_cache.cache),
        'current_model': chatbot.model_name,
        'content_loaded': bool(current_content),
        'current_file': current_filename,
        'content_length': len(current_content) if current_content else 0
    })

if __name__ == '__main__':
    print("🚀 Starting Fast Flask Chatbot...")
    print(f"📊 Using model: {chatbot.model_name}")
    print("⚡ Performance optimizations enabled:")
    print("   - Reduced timeouts (15s max)")
    print("   - Smart content chunking")
    print("   - Limited file sizes (5MB max)")
    print("   - Optimized model parameters")
    print("   - Simplified caching")
    
    app.run(
        debug=False,
        threaded=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False
    )