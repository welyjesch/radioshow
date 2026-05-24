import os
import json
import logging
from cloud_cfg_provider import get_cfg_settings_batch_from_cloud

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def live_test_cfg_generation():
    print("\n--- LIVE TEST: CFG Generation via Ollama ---")
    # USER: Change this API key manually
    api_key = ""
    
    test_cases = [
        "Hello there! (excited)",
        "I can't believe this is happening... (sad)",
        "Get out of my way! (angry)",
        "Just a normal day. (neutral)",
        "What on earth is that? (surprised)"
    ]
    
    for text in test_cases:
        print(f"\nTesting text: {text}")
        try:
            settings = get_cfg_settings_batch_from_cloud([text], api_key)[text]
            print(f"Result: {json.dumps(settings, indent=2)}")
            
            # Validation
            required_keys = {'exaggeration', 'cfg_weight', 'temperature'}
            if all(k in settings for k in required_keys):
                print("✅ SUCCESS: All required keys present.")
            else:
                print(f"❌ FAILURE: Missing keys. Found: {list(settings.keys())}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    live_test_cfg_generation()
