import requests
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def get_cfg_settings_batch_from_cloud(text_lines: List[str], api_key: str) -> Dict[str, dict]:
    """
    Queries the gemma4:31b-cloud model via Ollama API to generate CFG settings 
    for a batch of text lines.
    
    Returns a dictionary where keys are the original text lines and values are the CFG settings.
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

    # Format the lines for the prompt
    lines_formatted = "\n".join([f"- {line}" for line in text_lines])

    prompt = (
        f"Analyze the following lines of text, including any cues in parentheses, and determine the appropriate "
        f"audio generation CFG settings (exaggeration, cfg_weight, temperature) for EACH line.\n\n"
        f"CONSTRAINTS:\n"
        f"- set cfg_weight at max of 0.7\n"
        f"- set exaggeration at min of 0.6\n"
        f"- all cfg temperature should be set at 1.0\n\n"
        f"CRITICAL: Pay special attention to directorial cues enclosed in parentheses. For example, if the text contains '(angry)', "
        f"you MUST use the reference emotion preset parameters as the primary basis for the settings.\n\n"
        f"Reference Presets:\n{json.dumps(emotion_presets, indent=2)}\n\n"
        f"Lines to analyze:\n{lines_formatted}\n\n"
        f"Respond ONLY with a JSON object where the keys are the exact text of the lines and the values are objects containing 'exaggeration', 'cfg_weight', and 'temperature'. "
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
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        settings_map = json.loads(result.get("response", "{}"))
        logger.info(f"Cloud CFG batch settings retrieved for {len(text_lines)} lines.")
        return settings_map
    except Exception as e:
        logger.error(f"Error querying cloud model for batch CFG settings: {e}")
        # Return neutral defaults for all lines in the batch if the API call fails
        return {line: emotion_presets["neutral"] for line in text_lines}
