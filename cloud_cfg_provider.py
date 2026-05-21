import requests
import json
import logging

logger = logging.getLogger(__name__)

def get_cfg_settings_from_cloud(text_line: str, api_key: str) -> dict:
    """
    Queries the gemma4:31b-cloud model via Ollama API to generate CFG settings 
    based on the text line and its emotional cues.
    """
    url = "https://ollama.com/api/generate" # Note: Replace with actual cloud endpoint if different
    
    # Example presets to guide the model
    emotion_presets = {
        "excited": {"exaggeration": 0.95, "cfg_weight": 0.2, "temperature": 1.3},
        "happy": {"exaggeration": 0.8, "cfg_weight": 0.3, "temperature": 1.1},
        "enthusiastic": {"exaggeration": 0.9, "cfg_weight": 0.25, "temperature": 1.2},
        "sad": {"exaggeration": 0.1, "cfg_weight": 0.9, "temperature": 0.4},
        "angry": {"exaggeration": 0.85, "cfg_weight": 0.2, "temperature": 1.0},
        "frustrated": {"exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.9},
        "calm": {"exaggeration": 0.2, "cfg_weight": 0.8, "temperature": 0.5},
        "neutral": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.7},
        "confused": {"exaggeration": 0.4, "cfg_weight": 0.6, "temperature": 0.8},
        "surprised": {"exaggeration": 0.8, "cfg_weight": 0.3, "temperature": 1.0},
        "tired": {"exaggeration": 0.05, "cfg_weight": 0.95, "temperature": 0.3},
        "worried": {"exaggeration": 0.3, "cfg_weight": 0.7, "temperature": 0.6},
    }

    prompt = (
        f"Analyze the following line of text, including any cues in parentheses, and determine the appropriate "
        f"audio generation CFG settings (exaggeration, cfg_weight, temperature).\n\n"
        f"Reference Presets:\n{json.dumps(emotion_presets, indent=2)}\n\n"
        f"Text Line: \"{text_line}\"\n\n"
        f"Respond ONLY with a JSON object containing the keys 'exaggeration', 'cfg_weight', and 'temperature'. "
        f"Do not include any conversational text or markdown formatting."
    )

    payload = {
        "model": "gemma4:31b-cloud",
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Ollama usually returns the text in the 'response' field
        settings = json.loads(result.get("response", "{}"))
        return settings
    except Exception as e:
        logger.error(f"Error querying cloud model for CFG settings: {e}")
        # Return a neutral default if the API call fails
        return emotion_presets["neutral"]
