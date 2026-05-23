# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "tangoflux",
#     "torch",
#     "scipy==1.13.0",
# ]
# ///

import os
import json
import re
import argparse
import logging
import torch
from typing import List, Dict
from tangoflux import TangoFluxInference
import scipy.io.wavfile as wavfile

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

def save_audio_file(audio_np, segment, gen_idx, sample_rate, output_dir):
    """Saves SFX audio using the naming convention: NNNN-NN_SFX_description.wav"""
    seq_padded = str(segment["sequence"]).zfill(4)
    gen_padded = str(gen_idx + 1).zfill(2)
    sfx_part = truncate_text(segment["description"])
    filename = f"{seq_padded}-{gen_padded}_SFX_{sfx_part}.wav"
    filepath = os.path.join(output_dir, filename)
    
    try:
        wavfile.write(filepath, sample_rate, audio_np)
        return filepath
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return None

class SFXGenerator:
    def __init__(self):
        self.model = None
    
    def initialize(self):
        """Load TangoFlux model once."""
        if self.model is None:
            logger.info("Loading TangoFlux model...")
            self.model = TangoFluxInference(name='declare-lab/TangoFlux')
            logger.info("TangoFlux model loaded.")
    
    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            logger.info("Unloading TangoFlux model...")
            del self.model
            self.model = None
        torch.cuda.empty_cache()
    
    def generate(self, segment: Dict, gen_count: int) -> List[torch.Tensor]:
        """Generate N audio versions for an SFX segment."""
        self.initialize()
        
        sfx_description = segment['description']
        
        logger.info(f"Generating {gen_count} SFX versions: {sfx_description[:40]}...")
        
        audio_tensors = []
        for gen_idx in range(gen_count):
            try:
                # TangoFlux doesn't explicitly take a torch.Generator in the provided snippet, 
                # but we can vary the prompt slightly or rely on internal randomness if needed.
                # For now, we follow the provided example.
                audio = self.model.generate(
                    sfx_description, 
                    steps=50, 
                    duration=10
                )
                
                audio_tensors.append(audio)
                logger.info(f"  Generated SFX version {gen_idx + 1}/{gen_count}")
            except Exception as e:
                logger.error(f"Failed to generate SFX version {gen_idx + 1}: {e}")
                continue
        
        return audio_tensors

def main():
    parser = argparse.ArgumentParser(description="Generate SFX from sfx_tasks.json using AudioLDM2")
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
    total_files = 0
    failed_count = 0
    sample_rate = 16000

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
            failed_count += len(task["gen_count"]) if isinstance(task["gen_count"], int) else 1
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
