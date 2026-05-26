import requests
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def get_preset_batch_from_cloud(seq_ids: List[str], text_lines: List[str], voice_presets: List[str], api_key: str) -> Dict[str, Dict]:
    """
    Queries the gemma4:31b-cloud model via Ollama API to select the best voice preset 
    from a provided list for a batch of text lines.
    
    Returns a dictionary where keys are the sequence IDs and values are objects containing the preset and transformed text.
    """
    url = "https://ollama.com/api/generate"
    
    # Format the lines with their IDs for the prompt
    lines_formatted = "\n".join([f"ID {sid}: {line}" for sid, line in zip(seq_ids, text_lines)])
    presets_formatted = ", ".join(voice_presets)

    prompt = (
        f"Analyze the following lines of text, including any cues in parentheses, and select the most appropriate "
        f"voice preset from the provided list for EACH line to best capture the emotion and delivery.\n\n"
        f"Additionally, transform each line by inserting appropriate emotive tags from the ALLOWED LIST below. "
        f"CRITICAL: Paralingual tags must be placed ONLY at the very beginning or the very end of the sentence to avoid disrupting delivery (e.g., \"[fear] winter is coming!\" or \"I am your father! [cough]\"). "
        f"If a pause is needed, insert an ellipsis (...) inline instead of a pause tag.\n\n"
        f"ALLOWED EMOTIVE TAGS:\n"
        f"[advertisement], [angry], [chuckle], [clear throat], [cough], [crying], [dramatic], [fear], [gasp], [groan], [happy], [laugh], [narration], [sarcastic], [shush], [sigh], [sniff], [surprised], [whispering]\n\n"
        f"AVAILABLE PRESETS:\n{presets_formatted}\n\n"
        f"Lines to analyze:\n{lines_formatted}\n\n"
        f"Respond ONLY with a JSON object where the keys are the IDs (e.g., \"1\", \"2\") and the values are another object containing:\n"
        f"1. 'preset': The selected preset name (string).\n"
        f"2. 'transformed_text': The line with allowed emotive tags and ellipses inserted.\n"
        f"3. 'p_pause': A float representing the duration of silence (in seconds) to append after the line (e.g., 0.5, 1.0, or 0.0 for no pause).\n"
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
        
        preset_map = json.loads(result.get("response", "{}"))
        logger.info(f"Cloud preset batch selection retrieved for {len(text_lines)} lines.")
        return preset_map
    except Exception as e:
        logger.error(f"Error querying cloud model for batch preset selection: {e}")
        # Fallback: return the first preset and original text for all lines if the API call fails
        fallback_preset = voice_presets[0] if voice_presets else "default"
        return {line: {"preset": fallback_preset, "transformed_text": line} for line in text_lines}
