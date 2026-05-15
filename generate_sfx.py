# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "diffusers==0.27.2",
#     "transformers==4.38.2",
#     "huggingface-hub==0.22.2",
#     "accelerate==0.28.0",
#     "torch",
#     "scipy==1.13.0",
# ]
# ///

import os
import json
import argparse
import logging
import torch
from typing import List, Dict
from diffusers import AudioLDM2Pipeline, DPMSolverMultistepScheduler
import scipy.io.wavfile as wavfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_audio_file(audio_np, segment, gen_idx, sample_rate, output_dir):
    """Saves SFX audio using the naming convention: <sequence>_<SFX>_<gen>.wav"""
    seq = segment["sequence"]
    desc = segment["description"].replace(" ", "_")[:30]
    filename = f"<{seq}>_SFX_{desc}_{gen_idx + 1}.wav"
    filepath = os.path.join(output_dir, filename)
    
    try:
        wavfile.write(filepath, sample_rate, audio_np)
        return filepath
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return None

class SFXGenerator:
    def __init__(self):
        self.pipe = None
    
    def initialize(self):
        """Load AudioLDM2 model once."""
        if self.pipe is None:
            logger.info("Loading AudioLDM2 model...")
            model_id = "cvssp/audioldm2"
            self.pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.enable_model_cpu_offload()
            logger.info("AudioLDM2 model loaded.")
    
    def unload(self):
        """Unload model from memory."""
        if self.pipe is not None:
            logger.info("Unloading AudioLDM2 model...")
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
    
    def generate(self, segment: Dict, gen_count: int) -> List[torch.Tensor]:
        """Generate N audio versions for an SFX segment."""
        self.initialize()
        
        sfx_description = segment['description']
        negative_prompt = "Low quality, average quality, muffled, noisy"
        
        logger.info(f"Generating {gen_count} SFX versions: {sfx_description[:40]}...")
        
        audio_tensors = []
        for gen_idx in range(gen_count):
            try:
                seed = 42 + gen_idx
                generator = torch.Generator("cuda").manual_seed(seed)
                
                audio = self.pipe(
                    prompt=sfx_description,
                    negative_prompt=negative_prompt,
                    num_inference_steps=10,
                    audio_length_in_s=10.0,
                    generator=generator
                )
                
                audio_np = audio.audios[0]
                audio_tensors.append(audio_np)
                logger.info(f"  Generated SFX version {gen_idx + 1}/{gen_count}")
            except Exception as e:
                logger.error(f"Failed to generate SFX version {gen_idx + 1}: {e}")
                continue
        
        return audio_tensors

def main():
    parser = argparse.ArgumentParser(description="Generate SFX from sfx_tasks.json using AudioLDM2")
    parser.add_argument("--tasks-file", type=str, required=True, help="Path to sfx_tasks.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated audio")
    args = parser.parse_args()

    if not os.path.exists(args.tasks_file):
        logger.error(f"Tasks file not found: {args.tasks_file}")
        return

    with open(args.tasks_file, "r") as f:
        tasks = json.load(f)

    if not tasks:
        logger.info("No SFX tasks to process.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
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
                filepath = save_audio_file(audio_np, task, gen_idx, sample_rate, args.output_dir)
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
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
