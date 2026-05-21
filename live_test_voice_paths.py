import os
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def live_test_voice_parsing():
    print("\n--- LIVE TEST: Voice Path Parsing ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    voice_paths_file = os.path.join(script_dir, "voice_paths.json")
    sample_script_file = os.path.join(script_dir, "sample_script.txt")
    
    if not os.path.exists(voice_paths_file):
        print(f"❌ FAILURE: voice_paths.json not found at {voice_paths_file}")
        return
    if not os.path.exists(sample_script_file):
        print(f"❌ FAILURE: sample_script.txt not found at {sample_script_file}")
        return

    try:
        with open(voice_paths_file, "r") as f:
            raw_paths = json.load(f)
            
        with open(sample_script_file, "r") as f:
            script_content = f.read()
            
        print(f"Loaded raw paths: {json.dumps(raw_paths, indent=2)}")
        
        # Simulate the logic used in generate_audio.py
        voice_paths = {k.upper(): os.path.join(script_dir, v) for k, v in raw_paths.items()}
        
        print("\n--- Parsing Sample Script for Voice Keys ---")
        import re
        # Find all [VOICE_NAME] patterns
        found_voices = re.findall(r'\[([A-Z\s]+)\]', script_content)
        unique_voices = sorted(list(set(found_voices)))
        
        print(f"Voices found in script: {unique_voices}")
        
        print("\n--- Verifying Voice Path Resolution ---")
        for voice in unique_voices:
            voice_key = voice.strip().upper()
            if voice_key in voice_paths:
                path = voice_paths[voice_key]
                exists = os.path.exists(path)
                status = "✅ EXISTS" if exists else "❌ MISSING"
                print(f"{voice_key}: {path} -> {status}")
            else:
                print(f"{voice_key}: ❌ NOT FOUND IN voice_paths.json")
            
    except Exception as e:
        print(f"❌ ERROR during parsing: {e}")

if __name__ == "__main__":
    live_test_voice_parsing()
