# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "torch==2.7.1",
#     "torchaudio==2.7.1",
#     "numpy>=2.2.6",
#     "stable-audio-3 @ git+https://github.com/Stability-AI/stable-audio-3.git",
#     "beautifulsoup4",
# ]
# ///

import os
import json
import re
import argparse
import logging
import torch
import numpy as np
from typing import List, Dict
import torchaudio as ta
from stable_audio_3 import StableAudioModel
# Removed script_logger import

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sanitize_filename(text, max_length=16):
    """Convert text to safe filename."""
    clean = text.lower()
    clean = re.sub(r'[^a-z0-9\s]', '', clean)
    clean = re.sub(r'\s+', '_', clean.strip())
    return clean[:max_length].strip('_')

def truncate_text(text, max_length=16):
    """Truncate text and sanitize for filename."""
    return sanitize_filename(text, max_length)

def save_audio_file(audio_tensor, segment, gen_idx, sample_rate, output_dir):
    """Saves SFX audio using the naming convention: NNNN-NN_SFX_description.wav"""
    seq_padded = str(segment["sequence"]).zfill(4)
    gen_padded = str(gen_idx + 1).zfill(2)
    sfx_part = truncate_text(segment["description"])
    filename = f"{seq_padded}-{gen_padded}_SFX_{sfx_part}.wav"
    filepath = os.path.join(output_dir, filename)
    
    # Create unique ID: <seq_no>_<sfx/dia>_<gen_no>_<speaker>
    # For SFX, speaker is usually 'SFX' or the description
    unique_id = f"{segment['sequence']}_sfx_{gen_idx + 1}_SFX"
    
    try:
        # Ensure audio_tensor is a torch.Tensor for torchaudio.save
        if isinstance(audio_tensor, np.ndarray):
            audio_tensor = torch.from_numpy(audio_tensor)
        if torch.is_tensor(audio_tensor):
            audio_tensor = audio_tensor.detach().cpu().float()

        # Ensure it's 2D [channels, time] — required by torchaudio.save
        logger.info(f"Audio tensor shape before reshape: {audio_tensor.shape}, ndim={audio_tensor.ndim}")
        if audio_tensor.ndim == 3:
            # [batch, channels, time] -> select first batch item
            audio_tensor = audio_tensor[0]
        if audio_tensor.ndim == 1:
            # [time] -> [1, time]
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor [channels, time] after reshape, got {audio_tensor.ndim}D with shape {audio_tensor.shape}")
        
        logger.info(f"Saving {filename}: shape={audio_tensor.shape}, sr={sample_rate}")
        ta.save(filepath, audio_tensor, sample_rate)
        
        # Log to script.json
        try:
            map_path = os.path.join(output_dir, "script.json")
            data = []
            if os.path.exists(map_path):
                with open(map_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, dict):
                            # Convert legacy dict format to list
                            data = [{"id": k, **v} for k, v in data.items()]
                    except json.JSONDecodeError:
                        pass
            
            # Update existing entry or add new one
            updated = False
            for item in data:
                if item.get('id') == unique_id:
                    item.update({
                        "filename": filename,
                        "type": "sfx",
                        "sequence": segment["sequence"],
                        "gen_idx": gen_idx + 1,
                        "description": segment["description"],
                        "text": segment.get("original_text", segment["description"])
                    })
                    updated = True
                    break
            
            if not updated:
                data.append({
                    "id": unique_id,
                    "filename": filename,
                    "type": "sfx",
                    "sequence": segment["sequence"],
                    "gen_idx": gen_idx + 1,
                    "description": segment["description"],
                    "text": segment.get("original_text", segment["description"])
                })
            
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as log_e:
            logger.error(f"Failed to update script.json: {log_e}")
            
        return filepath
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return None

class SFXGenerator:
    def __init__(self):
        self.model = None
    
    def initialize(self):
        """Load Stable Audio 3 model once."""
        if self.model is None:
            logger.info("Loading Stable Audio 3 model (small-sfx)...")
            try:
                # StableAudioModel.from_pretrained does not accept 'token' argument
                # Use huggingface_hub login or environment variable HF_TOKEN
                self.model = StableAudioModel.from_pretrained("small-sfx")
                logger.info("Stable Audio 3 model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Stable Audio 3 model: {e}")
                self.model = None
    
    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            logger.info("Unloading Stable Audio model...")
            del self.model
            self.model = None
        torch.cuda.empty_cache()
    
    def get_sample_rate(self):
        """Return the model's native sample rate."""
        if self.model is not None:
            return self.model.model.sample_rate
        return 44100  # Stable Audio 3 default

    def generate(self, segment: Dict, gen_count: int) -> List[torch.Tensor]:
        """Generate N audio versions for an SFX segment."""
        if self.model is None:
            self.initialize()
        
        sfx_description = segment['description']
        
        logger.info(f"Generating {gen_count} SFX versions: {sfx_description[:40]}...")
        
        audio_tensors = []
        for gen_idx in range(gen_count):
            try:
                # Generate audio using Stable Audio 3
                # Returns tensor of shape [batch, channels, samples]
                audio = self.model.generate(
                    prompt=sfx_description, 
                    duration=7,
                )
                
                # Index into batch dimension to get [channels, samples]
                # This matches the official CLI: audio[i].cpu()
                audio_2d = audio[0].cpu()
                
                audio_tensors.append(audio_2d)
                logger.info(f"  Generated SFX version {gen_idx + 1}/{gen_count} (shape: {audio_2d.shape})")
            except Exception as e:
                logger.error(f"Failed to generate SFX version {gen_idx + 1}: {e}")
                continue
        
        return audio_tensors

def main():
    parser = argparse.ArgumentParser(description="Generate SFX from sfx_tasks.json using Stable Audio 3")
    parser.add_argument("--tasks-file", type=str, required=True, help="Path to sfx_tasks.json")
    args = parser.parse_args()

    if not os.path.exists(args.tasks_file):
        logger.error(f"Tasks file not found: {args.tasks_file}")
        return

    # Output directory is the same as tasks file directory
    output_dir = os.path.dirname(os.path.abspath(args.tasks_file))

    with open(args.tasks_file, "r") as f:
        tasks = json.load(f)

    if not tasks:
        logger.info("No SFX tasks to process.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    sfx_gen = SFXGenerator()
    sfx_gen.initialize()  # Load model upfront to get sample_rate
    total_files = 0
    failed_count = 0
    sample_rate = sfx_gen.get_sample_rate()
    logger.info(f"Model sample rate: {sample_rate} Hz")

    logger.info("=" * 60)
    logger.info(f"Processing {len(tasks)} SFX tasks")
    logger.info("=" * 60)

    for task in tasks:
        try:
            audio_samples = sfx_gen.generate(task, task["gen_count"])
            
            for gen_idx, audio_np in enumerate(audio_samples):
                filepath = save_audio_file(audio_np, task, gen_idx, sample_rate, output_dir)
                if filepath:
                    total_files += 1
                else:
                    failed_count += 1
        except Exception as e:
            logger.error(f"Failed to process SFX task {task['sequence']}: {e}")
            failed_count += task["gen_count"] if isinstance(task["gen_count"], int) else 1
            continue

    sfx_gen.unload()

    logger.info("=" * 60)
    logger.info(f"SFX Generation complete!")
    logger.info(f"Total SFX files generated: {total_files}")
    logger.info(f"Failed generations: {failed_count}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
