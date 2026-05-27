import requests
import json
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_best_preset(text_line: str, voice_presets: List[str], api_key: str) -> str:
    """
    Queries the gemma4:31b-cloud model via Ollama API to select the best voice preset
    from the provided list of filenames in diurnal_voicebank for a single text line.
    
    Returns the selected preset filename.
    """
    url = "https://ollama.com/api/generate"
    
    presets_formatted = ", ".join(voice_presets)

    prompt = (
        f"Analyze the following line of text and select the most appropriate "
        f"voice preset from the provided list of preset filenames to best capture the emotion and delivery.\n\n"
        f"AVAILABLE PRESETS:\n{presets_formatted}\n\n"
        f"Line to analyze: \"{text_line}\"\n\n"
        f"Respond ONLY with a JSON object containing a single key 'preset' whose value is the selected preset filename (e.g. \"warm.mp3\"). "
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
        
        response_data = json.loads(result.get("response", "{}"))
        selected_preset = response_data.get("preset", "")
        if selected_preset in voice_presets:
            logger.info(f"Selected preset: {selected_preset} for text: '{text_line[:30]}'")
            return selected_preset
            
        # Fuzzy fallback or direct match in case of minor mismatch (e.g. prefix matches)
        clean_selected = selected_preset.strip().lower()
        for preset in voice_presets:
            if preset.lower() == clean_selected or preset.lower().startswith(clean_selected):
                return preset
                
    except Exception as e:
        logger.error(f"Error querying cloud model for preset selection: {e}")
        
    # Default fallback: neutral.mp3 or first available preset
    fallback_preset = "neutral.mp3" if "neutral.mp3" in voice_presets else (voice_presets[0] if voice_presets else "")
    logger.warning(f"Fallback to preset: {fallback_preset}")
    return fallback_preset
