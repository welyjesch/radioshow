# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flask>=2.3.0",
#     "flask-cors>=4.0.0", 
#     "pydub>=0.25.0",
# ]
# ///

import sys
import audioop
sys.modules['pyaudioop'] = audioop

import os
import re
from pathlib import Path
from typing import List, Dict
from pydub import AudioSegment
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Get the root directory (parent of backend/)
app_dir = Path(__file__).parent.parent
generated_audio_dir = app_dir / "generated_audio"

app = Flask(__name__, static_folder=str(app_dir / 'backend' / 'static'))
CORS(app)

# Serve the frontend
@app.route('/')
def serve_index():
    return send_from_directory(str(app_dir / 'backend' / 'static'), 'index.html')

# ==================== CONFIGURATION ====================
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".wma"}
DEFAULT_OUTPUT_DIR = str(generated_audio_dir)

# ==================== UTILITY FUNCTIONS ====================

def extract_sequence_num(filename: str) -> int:
    # Extract ANY number from filename
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return 1  # Default to sequence 1 if no number found

def get_audio_files(output_dir: str) -> Dict[int, List[str]]:
    print(f"[DEBUG] Scanning directory: {output_dir}")
    print(f"[DEBUG] Supported formats: {AUDIO_EXTENSIONS}")
    
    if not os.path.exists(output_dir):
        print(f"[DEBUG] Directory does not exist: {output_dir}")
        return {}
    
    files_by_sequence = {}
    try:
        all_files = os.listdir(output_dir)
        print(f"[DEBUG] All files in directory: {all_files}")
        
        for filename in sorted(all_files):
            filepath = os.path.join(output_dir, filename)
            ext = Path(filename).suffix
            print(f"[DEBUG] Checking file: {filename}, extension: {ext}")
            
            if os.path.isfile(filepath) and ext in AUDIO_EXTENSIONS:
                seq_num = extract_sequence_num(filename)
                print(f"[DEBUG] File {filename} -> sequence {seq_num}")
                if seq_num is not None:
                    if seq_num not in files_by_sequence:
                        files_by_sequence[seq_num] = []
                    files_by_sequence[seq_num].append(filename)
    except Exception as e:
        print(f"Error scanning directory {output_dir}: {e}")
        return {}
    
    print(f"[DEBUG] Final files_by_sequence: {files_by_sequence}")
    return dict(sorted(files_by_sequence.items()))

def get_audio_duration(filepath: str) -> float:
    try:
        audio = AudioSegment.from_file(filepath)
        return len(audio) / 1000.0
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0.0

# ==================== API ENDPOINTS ====================

@app.route('/api/files', methods=['GET'])
def list_files():
    output_dir = DEFAULT_OUTPUT_DIR
    files_by_seq = get_audio_files(output_dir)
    
    result = []
    for seq, files in files_by_seq.items():
        file_info = []
        for f in files:
            path = os.path.join(output_dir, f)
            # Extract label - remove sequence prefix if present, otherwise use filename without extension
            label = re.sub(r"^\d+-\d+_\s*", "", f)
            label = re.sub(r"\.\w+$", "", label)  # Remove extension
            if not label.strip():
                label = Path(f).stem
            
            file_info.append({
                "filename": f,
                "label": label,
                "duration": get_audio_duration(path)
            })
        result.append({
            "sequence": seq,
            "files": file_info
        })
    
    return jsonify(result)

@app.route('/api/audio/<path:filename>')
def serve_audio(filename):
    output_dir = DEFAULT_OUTPUT_DIR
    filepath = os.path.join(output_dir, filename)
    
    print(f"[DEBUG] serve_audio called: filename={filename}")
    print(f"[DEBUG] output_dir={output_dir}")
    print(f"[DEBUG] full filepath={filepath}")
    print(f"[DEBUG] file exists: {os.path.exists(filepath)}")
    
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return jsonify({"error": "File not found"}), 404
    
    # Determine MIME type based on file extension
    ext = Path(filename).suffix.lower()
    mimetype_map = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac',
        '.m4a': 'audio/mp4',
        '.wma': 'audio/x-ms-wma'
    }
    mimetype = mimetype_map.get(ext, 'application/octet-stream')
    print(f"[DEBUG] Serving with mimetype: {mimetype}")
    return send_from_directory(output_dir, filename, mimetype=mimetype)

@app.route('/api/concatenate', methods=['POST'])
def concatenate():
    data = request.json
    concatenation_list = data.get('list', [])
    output_dir = DEFAULT_OUTPUT_DIR
    
    try:
        combined = AudioSegment.empty()
        for item in concatenation_list:
            if isinstance(item, dict) and "add_silence" in item:
                seconds = item["add_silence"]
                silence = AudioSegment.silent(duration=int(seconds * 1000))
                combined += silence
            else:
                filepath = os.path.join(output_dir, item)
                if os.path.exists(filepath):
                    audio = AudioSegment.from_file(filepath)
                    combined += audio
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"radioshow_output_{timestamp}.wav"
        final_path = os.path.join(output_dir, filename)
        combined.export(final_path, format="wav")
        
        return jsonify({"status": "success", "filename": filename})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print(f"[DEBUG] App directory: {app_dir}")
    print(f"[DEBUG] Generated audio directory: {generated_audio_dir}")
    print(f"[DEBUG] Static files path: {app_dir / 'backend' / 'static'}")
    app.run(debug=True, port=5000)

# ==================== UTILITY FUNCTIONS ====================

def extract_sequence_num(filename: str) -> int:
    # Extract ANY number from filename
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return 1  # Default to sequence 1 if no number found

def get_audio_files(output_dir: str) -> Dict[int, List[str]]:
    print(f"[DEBUG] Scanning directory: {output_dir}")
    print(f"[DEBUG] Supported formats: {AUDIO_EXTENSIONS}")
    
    if not os.path.exists(output_dir):
        print(f"[DEBUG] Directory does not exist: {output_dir}")
        return {}
    
    files_by_sequence = {}
    try:
        all_files = os.listdir(output_dir)
        print(f"[DEBUG] All files in directory: {all_files}")
        
        for filename in sorted(all_files):
            filepath = os.path.join(output_dir, filename)
            ext = Path(filename).suffix
            print(f"[DEBUG] Checking file: {filename}, extension: {ext}")
            
            if os.path.isfile(filepath) and ext in AUDIO_EXTENSIONS:
                seq_num = extract_sequence_num(filename)
                print(f"[DEBUG] File {filename} -> sequence {seq_num}")
                if seq_num is not None:
                    if seq_num not in files_by_sequence:
                        files_by_sequence[seq_num] = []
                    files_by_sequence[seq_num].append(filename)
    except Exception as e:
        print(f"Error scanning directory {output_dir}: {e}")
        return {}
    
    print(f"[DEBUG] Final files_by_sequence: {files_by_sequence}")
    return dict(sorted(files_by_sequence.items()))

def get_audio_duration(filepath: str) -> float:
    try:
        audio = AudioSegment.from_file(filepath)
        return len(audio) / 1000.0
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0.0

# ==================== API ENDPOINTS ====================

@app.route('/api/files', methods=['GET'])
def list_files():
    output_dir = request.args.get('dir', DEFAULT_OUTPUT_DIR)
    files_by_seq = get_audio_files(output_dir)
    
    result = []
    for seq, files in files_by_seq.items():
        file_info = []
        for f in files:
            path = os.path.join(output_dir, f)
            # Extract label - remove sequence prefix if present, otherwise use filename without extension
            label = re.sub(r"^\d+-\d+_\s*", "", f)
            label = re.sub(r"\.\w+$", "", label)  # Remove extension
            if not label.strip():
                label = Path(f).stem
            
            file_info.append({
                "filename": f,
                "label": label,
                "duration": get_audio_duration(path)
            })
        result.append({
            "sequence": seq,
            "files": file_info
        })
    
    return jsonify(result)

@app.route('/api/audio/<path:filename>')
def serve_audio(filename):
    output_dir = request.args.get('dir', DEFAULT_OUTPUT_DIR)
    # Determine MIME type based on file extension
    ext = Path(filename).suffix.lower()
    mimetype_map = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac',
        '.m4a': 'audio/mp4',
        '.wma': 'audio/x-ms-wma'
    }
    mimetype = mimetype_map.get(ext, 'application/octet-stream')
    return send_from_directory(output_dir, filename, mimetype=mimetype)

@app.route('/api/concatenate', methods=['POST'])
def concatenate():
    data = request.json
    concatenation_list = data.get('list', [])
    output_dir = data.get('dir', DEFAULT_OUTPUT_DIR)
    
    try:
        combined = AudioSegment.empty()
        for item in concatenation_list:
            if isinstance(item, dict) and "add_silence" in item:
                seconds = item["add_silence"]
                silence = AudioSegment.silent(duration=int(seconds * 1000))
                combined += silence
            else:
                filepath = os.path.join(output_dir, item)
                if os.path.exists(filepath):
                    audio = AudioSegment.from_file(filepath)
                    combined += audio
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"radioshow_output_{timestamp}.wav"
        final_path = os.path.join(output_dir, filename)
        combined.export(final_path, format="wav")
        
        return jsonify({"status": "success", "filename": filename})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
