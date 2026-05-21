# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch==2.6.0",
#     "torchaudio==2.6.0",
#     "transformers==5.2.0",
#     "numpy>=1.24.0,<2.0.0",
#     "librosa==0.11.0",
#     "s3tokenizer",
#     "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master",
#     "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git",
# ]
# ///

import os
import sys
import re
import json
import argparse
import logging
import torch
import torchaudio as ta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from chatterbox.tts_turbo import ChatterboxTurboTTS
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
from cloud_cfg_provider import get_cfg_settings_from_cloud

# ==================== CONFIGURATION ====================

# Load voice paths from voice_paths.json if it exists
VOICE_PATHS = {}
if os.path.exists("voice_paths.json"):
    try:
        with open("voice_paths.json", "r") as f:
            raw_paths = json.load(f)
            VOICE_PATHS = {k.upper(): v for k, v in raw_paths.items()}
    except (json.JSONDecodeError, FileNotFoundError):
        pass

DEFAULT_VOICE_KEY = "default_voice"
OUTPUT_DIR = "generated_audio"
DEFAULT_GENERATION_COUNT = 7

GENERATION_MODIFIERS = [
    (0.7, 1.2, 0.9),  # Version 0: Subdued
    (1.1, 0.8, 1.1),  # Version 1: Slightly exaggerated
    (0.5, 1.5, 0.7),  # Version 2: Very stable/flat
    (1.0, 1.0, 1.0),  # Version 3: Base (Standard)
    (1.3, 0.7, 1.2),  # Version 4: High energy
    (0.8, 1.1, 0.8),  # Version 5: Muted/Soft
    (1.5, 0.6, 1.4),  # Version 6: Extreme/Emotive
]

# ==================== UTILITIES ====================

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def sanitize_filename(text, max_length=16):
    """Convert text to safe filename."""
    clean = text.lower()
    clean = re.sub(r'[^a-z0-9\s]', '', clean)
    clean = re.sub(r'\s+', '_', clean.strip())
    return clean[:max_length].strip('_')

def truncate_text(text, max_length=16):
    """Truncate text and sanitize for filename."""
    return sanitize_filename(text, max_length)

def split_sentences(text):
    """Split text into lines."""
    return text.splitlines()

def get_generation_modifiers(gen_idx: int) -> Tuple[float, float, float]:
    """Get parameter modifiers for a given generation index.
    
    Returns (exaggeration_multiplier, cfg_weight_multiplier, temperature_multiplier).
    For generations beyond the defined list, uses the base (4th generation) modifiers.
    """
    if gen_idx < len(GENERATION_MODIFIERS):
        return GENERATION_MODIFIERS[gen_idx]
    else:
        # All generations after the 7th use the 4th generation (base) settings
        return GENERATION_MODIFIERS[3]

# ==================== SCRIPT PARSER ====================

def parse_script(script_path: str) -> List[Dict]:
    """Parse script file with [SPEAKER] and [SFX: description] tags."""
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script file not found: {script_path}")
    
    with open(script_path, "r") as f:
        full_text = f.read()
    
    segments = []
    lines = full_text.split('\n')
    seq_counter = 0
    current_block_text_lines = []
    current_speaker_name = None
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for SFX tag
        sfx_match = re.match(r'^\[SFX:\s*(.+?)\]\s*(.*)', line_stripped)
        if sfx_match:
            # Process accumulated dialogue for previous speaker
            if current_block_text_lines and current_speaker_name:
                # Use the lines as they are, without joining and re-splitting by punctuation
                for line_text in current_block_text_lines:
                    if line_text:
                        seq_counter += 1
                        # Remove everything between the first '[' and the last ']' for the speaker tag
                        clean_sentence = re.sub(r'\[.*?\]', '', line_text).strip()
                        clean_sentence = re.sub(r'\s+', ' ', clean_sentence)
                        
                        segments.append({
                            'type': 'dialogue',
                            'sequence': seq_counter,
                            'speaker': current_speaker_name,
                            'text': clean_sentence,
                            'original_text': line_text,
                            'explicit_emotion': None
                        })
                current_block_text_lines = []
                current_speaker_name = None
            
            # Add SFX segment
            sfx_description = sfx_match.group(1).strip()
            if sfx_description:
                seq_counter += 1
                segments.append({
                    'type': 'sfx',
                    'sequence': seq_counter,
                    'sfx_description': sfx_description,
                    'original_text': sfx_description
                })
            continue
        
        # Check for speaker tag
        speaker_match = re.match(r'^\[([A-Za-z0-9_ \-]+)\]\s*(.*)', line_stripped)
        if speaker_match:
            # Process accumulated text for previous speaker
            if current_block_text_lines and current_speaker_name:
                # Use the lines as they are, without joining and re-splitting by punctuation
                for line_text in current_block_text_lines:
                    if line_text:
                        seq_counter += 1
                        # Remove everything between the first '[' and the last ']' for the speaker tag
                        clean_sentence = re.sub(r'\[.*?\]', '', line_text).strip()
                        clean_sentence = re.sub(r'\s+', ' ', clean_sentence)
                        
                        segments.append({
                            'type': 'dialogue',
                            'sequence': seq_counter,
                            'speaker': current_speaker_name,
                            'text': clean_sentence,
                            'original_text': line_text,
                            'explicit_emotion': None
                        })
                current_block_text_lines = []
            
            # Set new speaker
            current_speaker_name = speaker_match.group(1).upper()
            remaining_text = speaker_match.group(2).strip()
            
            # Remove parenthesized content for TTS
            text_for_tts = re.sub(r'\s*\([^)]+\)\s*', ' ', remaining_text).strip()
            text_for_tts = re.sub(r'\s+', ' ', text_for_tts).strip()
            
            if text_for_tts:
                current_block_text_lines.append(text_for_tts)
            
            # Store text for later use
            if current_speaker_name and remaining_text and not current_block_text_lines:
                current_block_text_lines = [text_for_tts]
            
            continue
        
        # Regular text line
        if line_stripped:
            current_block_text_lines.append(line_stripped)
    
    # Process remaining accumulated text
    if current_block_text_lines and current_speaker_name:
        # Use the lines as they are, without joining and re-splitting by punctuation
        for line_text in current_block_text_lines:
            if line_text:
                seq_counter += 1
                # Remove everything between the first '[' and the last ']' for the speaker tag
                clean_sentence = re.sub(r'\[.*?\]', '', line_text).strip()
                clean_sentence = re.sub(r'\s+', ' ', clean_sentence)
                
                segments.append({
                    'type': 'dialogue',
                    'sequence': seq_counter,
                    'speaker': current_speaker_name,
                    'text': clean_sentence,
                    'original_text': line_text
                })
    
    logger.info(f"Parsed {len(segments)} segments from {script_path}")
    return segments

# ==================== AUDIO GENERATORS ====================

class DialogueGenerator:
    def __init__(self, api_key: str):
        self.model = None
        self.api_key = api_key
    
    def initialize(self):
        """Load models once."""
        if self.model is None:
            logger.info("Loading ChatterboxTurboTTS model...")
            self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    
    def unload(self):
        """Unload models from memory."""
        if self.model is not None:
            logger.info("Unloading ChatterboxTurboTTS model...")
            del self.model
            self.model = None
        torch.cuda.empty_cache()
    
    def generate(self, segment: Dict, gen_count: int) -> List[torch.Tensor]:
        """Generate N audio versions for a dialogue segment."""
        self.initialize()
        
        speaker_name = segment['speaker']
        text = segment['text']
        original_text = segment['original_text']
        
        # Get voice path
        voice_path = VOICE_PATHS.get(speaker_name.upper(), VOICE_PATHS.get(DEFAULT_VOICE_KEY.upper()))
        if not voice_path or not os.path.exists(voice_path):
            logger.warning(f"Voice file not found for {speaker_name}. Using fallback.")
            voice_path = VOICE_PATHS.get(DEFAULT_VOICE_KEY.upper())
        
        # Get CFG settings from cloud model
        params = get_cfg_settings_from_cloud(original_text, self.api_key)
        
        # Strip parentheses-enclosed tags (emotion/delivery markers) from text
        # These should not be spoken aloud
        text_for_tts = re.sub(r'\s*\([^)]+\)\s*', ' ', text).strip()
        text_for_tts = re.sub(r'\s+', ' ', text_for_tts)
        
        logger.info(f"Generating {gen_count} versions for '{speaker_name}': {text_for_tts[:40]}...")
        
        audio_tensors = []
        final_params = []
        for gen_idx in range(gen_count):
            try:
                # Get modifiers for this generation
                exaggeration_mult, cfg_weight_mult, temperature_mult = get_generation_modifiers(gen_idx)
                
                # Apply modifiers to base parameters
                modified_exaggeration = params['exaggeration'] * exaggeration_mult
                modified_cfg_weight = params['cfg_weight'] * cfg_weight_mult
                modified_temperature = params['temperature'] * temperature_mult
                
                audio = self.model.generate(
                    text_for_tts,
                    audio_prompt_path=voice_path,
                    exaggeration=modified_exaggeration,
                    cfg_weight=modified_cfg_weight,
                    temperature=modified_temperature
                )
                audio_tensors.append(audio)
                final_params.append({
                    'exaggeration': modified_exaggeration,
                    'cfg_weight': modified_cfg_weight,
                    'temperature': modified_temperature
                })
                logger.info(f"  Generated version {gen_idx + 1}/{gen_count} (modifiers: {exaggeration_mult}x, {cfg_weight_mult}x, {temperature_mult}x)")
            except Exception as e:
                logger.error(f"Failed to generate version {gen_idx + 1}: {e}")
                final_params.append(None)
                continue
        
        return audio_tensors, final_params

# ==================== FILE HANDLER ====================

def save_audio_file(audio_tensor: torch.Tensor, segment: Dict, gen_idx: int, 
                    sample_rate: int, output_dir: str) -> Optional[str]:
    """Save audio tensor to WAV file with naming convention."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        seq_padded = str(segment['sequence']).zfill(4)
        gen_padded = str(gen_idx + 1).zfill(2)
        
        if segment['type'] == 'dialogue':
            speaker_name = segment['speaker'].replace(' ', '_')
            text_part = truncate_text(segment['text'])
            filename = f"{seq_padded}-{gen_padded}_{speaker_name}_{text_part}.wav"
        else:  # sfx
            sfx_part = truncate_text(segment['sfx_description'])
            filename = f"{seq_padded}-{gen_padded}_SFX_{sfx_part}.wav"
        
        filepath = os.path.join(output_dir, filename)
        
        # Convert tensor to numpy and save
        audio_np = audio_tensor.cpu().numpy()
        
        # Handle multi-channel
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        
        # Transpose to (samples, channels) for scipy
        if audio_np.shape[0] < audio_np.shape[1]:
            audio_np = audio_np.T
        
        # Normalize to int16
        audio_np = (audio_np * 32767).astype('int16')
        
        wavfile.write(filepath, sample_rate, audio_np)
        logger.info(f"Saved: {filename}")
        return filepath
    
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")
        return None

# ==================== MAIN ORCHESTRATION ====================

def main():
    parser = argparse.ArgumentParser(description="Generate audio from script with dialogue and SFX.")
    parser.add_argument("script_path", help="Path to script file (.txt)")
    parser.add_argument("--gen-count", "-c", type=int, default=DEFAULT_GENERATION_COUNT,
                       help=f"Number of versions to generate per segment (default: {DEFAULT_GENERATION_COUNT})")
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR,
                       help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--apikey", type=str, default=os.environ.get("OLLAMA_API_KEY", "your_api_key_here"),
                       help="API key for cloud CFG provider (defaults to OLLAMA_API_KEY env var)")
    
    args = parser.parse_args()
    
    logger.info(f"Script: {args.script_path}")
    logger.info(f"Generation count: {args.gen_count}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Parse script
    try:
        segments = parse_script(args.script_path)
    except Exception as e:
        logger.error(f"Failed to parse script: {e}")
        sys.exit(1)
    
    if not segments:
        logger.error("No segments found in script.")
        sys.exit(1)
    
    # Separate dialogue and SFX segments
    dialogue_segments = [s for s in segments if s['type'] == 'dialogue']
    sfx_segments_to_process = [s for s in segments if s['type'] == 'sfx']
    
    logger.info(f"Found {len(dialogue_segments)} dialogue segments and {len(sfx_segments_to_process)} SFX segments")
    
    total_files = 0
    failed_count = 0
    
    # ==================== PASS 1: PROCESS DIALOGUE WITH CHATTERBOX ====================
    logger.info("=" * 60)
    logger.info("PASS 1: Processing dialogue segments (Chatterbox)")
    logger.info("=" * 60)
    
    dialogue_gen = DialogueGenerator(api_key=args.apikey)
    
    for segment in dialogue_segments:
        try:
            audio_tensors, final_params = dialogue_gen.generate(segment, args.gen_count)
            sample_rate = dialogue_gen.model.sr
            
            # Save generated audio files
            for gen_idx, audio_tensor in enumerate(audio_tensors):
                filepath = save_audio_file(audio_tensor, segment, gen_idx, sample_rate, args.output_dir)
                if filepath:
                    total_files += 1
                    
                    # Log metadata
                    filename = os.path.basename(filepath)
                    duration = len(AudioSegment.from_file(filepath)) / 1000.0
                    
                    # Get the specific CFG values for this generation index
                    gen_params = final_params[gen_idx]
                    
                    metadata_entry = {
                        "filename": filename,
                        "transcription": segment['text'],
                        "duration": duration,
                        "exaggeration": gen_params['exaggeration'],
                        "cfg_weight": gen_params['cfg_weight'],
                        "temperature": gen_params['temperature']
                    }
                    
                    metadata_path = os.path.join(args.output_dir, "generation_metadata.json")
                    metadata = []
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            try:
                                metadata = json.load(f)
                            except json.JSONDecodeError:
                                metadata = []
                    
                    metadata.append(metadata_entry)
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=4)
                else:
                    failed_count += 1
        
        except Exception as e:
            logger.error(f"Failed to process dialogue segment {segment['sequence']}: {e}")
            failed_count += 1
            continue
    
    logger.info(f"Dialogue pass complete. Generated {total_files} files, {failed_count} failed.")
    
    # Unload Chatterbox to free VRAM
    dialogue_gen.unload()
    
    # ==================== EXPORT SFX TASKS ====================
    if sfx_segments_to_process:
        sfx_tasks_path = os.path.join(args.output_dir, "sfx_tasks.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        sfx_tasks = []
        for segment in sfx_segments_to_process:
            sfx_tasks.append({
                "sequence": segment["sequence"],
                "description": segment["sfx_description"],
                "gen_count": args.gen_count
            })
            
        try:
            with open(sfx_tasks_path, "w") as f:
                json.dump(sfx_tasks, f, indent=4)
            logger.info(f"Exported {len(sfx_tasks)} SFX tasks to {sfx_tasks_path}")
        except Exception as e:
            logger.error(f"Failed to export SFX tasks: {e}")

    # ==================== SUMMARY ====================
    logger.info("=" * 60)
    logger.info(f"Generation complete!")
    logger.info(f"Total dialogue files generated: {total_files}")
    logger.info(f"Failed generations: {failed_count}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
