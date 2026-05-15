# /// script dependencies
# flask
# flask-cors
# pydub
# ///

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from pydub import AudioSegment
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app)

# Serve the frontend
@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

# ==================== CONFIGURATION ====================
SUPPORTED_FORMATS = {".wav", ".mp3"}
DEFAULT_OUTPUT_DIR = "voices"

# ==================== UTILITY FUNCTIONS ====================

def extract_sequence_num(filename: str) -> Optional[int]:
    match = re.match(r"(\d+)-\d+_", filename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def get_audio_files(output_dir: str) -> Dict[int, List[str]]:
    if not os.path.exists(output_dir):
        return {}
    
    files_by_sequence = {}
    try:
        for filename in sorted(os.listdir(output_dir)):
            filepath = os.path.join(output_dir, filename)
            if os.path.isfile(filepath) and Path(filename).suffix.lower() in SUPPORTED_FORMATS:
                seq_num = extract_sequence_num(filename)
                if seq_num is not None:
                    if seq_num not in files_by_sequence:
                        files_by_sequence[seq_num] = []
                    files_by_sequence[seq_num].append(filename)
    except Exception as e:
        print(f"Error scanning directory {output_dir}: {e}")
        return {}
    
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
            file_info.append({
                "filename": f,
                "label": re.sub(r"^\d+-\d+_", "", f).replace(Path(f).suffix, ""),
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
    mimetype = 'audio/wav' if ext == '.wav' else 'audio/mpeg'
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
