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
from transformers import pipeline
from chatterbox.tts_turbo import ChatterboxTurboTTS
import scipy.io.wavfile as wavfile

# ==================== CONFIGURATION ====================

EMOTION_PARAMETERS: Dict[str, Dict[str, float]] = {
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

DEFAULT_VOICE_PATH = "default_voice.wav"
OUTPUT_DIR = "generated_audio"
DEFAULT_GENERATION_COUNT = 7

# Generation modifiers for parameter diversity
# Maps generation index to multipliers for (exaggeration, cfg_weight, temperature)
GENERATION_MODIFIERS = [
    (0.75, 0.75, 0.75),   # Gen 0: -25%, -25%, -25%
    (1.0, 0.75, 0.75),    # Gen 1: 0%, -25%, -25%
    (1.0, 1.0, 0.75),     # Gen 2: 0%, 0%, -25%
    (1.0, 1.0, 1.0),      # Gen 3: 0%, 0%, 0% (base)
    (1.25, 1.0, 1.0),     # Gen 4: 25%, 0%, 0%
    (1.25, 1.25, 1.0),    # Gen 5: 25%, 25%, 0%
    (1.25, 1.25, 0.75),   # Gen 6: 25%, 25%, -25%
]

# Load voice paths from voice_paths.json if it exists
VOICE_PATHS = {}
if os.path.exists("voice_paths.json"):
    try:
        with open("voice_paths.json", "r") as f:
            VOICE_PATHS = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        pass

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
    """Split text into sentences."""
    sentences = re.findall(r'[^.!?]+(?:[.!?]+|$)', text)
    return [s.strip() for s in sentences if s.strip()]

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
                block_full_text = " ".join(current_block_text_lines).strip()
                if block_full_text:
                    sentences = split_sentences(block_full_text)
                    for sentence in sentences:
                        seq_counter += 1
                        # Strip parentheses-enclosed tags from the sentence before storing
                        clean_sentence = re.sub(r'\s*\([^)]+\)\s*', ' ', sentence).strip()
                        clean_sentence = re.sub(r'\s+', ' ', clean_sentence)
                        
                        segments.append({
                            'type': 'dialogue',
                            'sequence': seq_counter,
                            'speaker': current_speaker_name,
                            'text': clean_sentence,
                            'original_text': sentence,
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
        speaker_match = re.match(r'^\[([A-Za-z0-9_ ]+)\]\s*(.*)', line_stripped)
        if speaker_match:
            # Process accumulated text for previous speaker
            if current_block_text_lines and current_speaker_name:
                block_full_text = " ".join(current_block_text_lines).strip()
                if block_full_text:
                    sentences = split_sentences(block_full_text)
                    for sentence in sentences:
                        seq_counter += 1
                        # Strip parentheses-enclosed tags from the sentence before storing
                        clean_sentence = re.sub(r'\s*\([^)]+\)\s*', ' ', sentence).strip()
                        clean_sentence = re.sub(r'\s+', ' ', clean_sentence)
                        
                        segments.append({
                            'type': 'dialogue',
                            'sequence': seq_counter,
                            'speaker': current_speaker_name,
                            'text': clean_sentence,
                            'original_text': sentence,
                            'explicit_emotion': None
                        })
                current_block_text_lines = []
            
            # Set new speaker
            current_speaker_name = speaker_match.group(1).upper()
            remaining_text = speaker_match.group(2).strip()
            
            # Extract explicit emotion from parentheses
            explicit_emotion = None
            parenthesized = re.findall(r'\(([^)]+)\)', remaining_text)
            for phrase in parenthesized:
                words = re.split(r'[,;\s]+', phrase)
                for word in words:
                    if word.lower() in EMOTION_PARAMETERS:
                        explicit_emotion = word.lower()
                        break
                if explicit_emotion:
                    break
            
            # Remove parenthesized content for TTS
            text_for_tts = re.sub(r'\s*\([^)]+\)\s*', ' ', remaining_text).strip()
            text_for_tts = re.sub(r'\s+', ' ', text_for_tts).strip()
            
            if text_for_tts:
                current_block_text_lines.append(text_for_tts)
            
            # Store emotion info for later use
            if current_speaker_name and remaining_text and not current_block_text_lines:
                current_block_text_lines = [text_for_tts]
            
            continue
        
        # Regular text line
        if line_stripped:
            current_block_text_lines.append(line_stripped)
    
    # Process remaining accumulated text
    if current_block_text_lines and current_speaker_name:
        block_full_text = " ".join(current_block_text_lines).strip()
        if block_full_text:
            sentences = split_sentences(block_full_text)
            for sentence in sentences:
                seq_counter += 1
                # Strip parentheses-enclosed tags from the sentence before storing
                clean_sentence = re.sub(r'\s*\([^)]+\)\s*', ' ', sentence).strip()
                clean_sentence = re.sub(r'\s+', ' ', clean_sentence)
                
                segments.append({
                    'type': 'dialogue',
                    'sequence': seq_counter,
                    'speaker': current_speaker_name,
                    'text': clean_sentence,
                    'original_text': sentence,
                    'explicit_emotion': None
                })
    
    logger.info(f"Parsed {len(segments)} segments from {script_path}")
    return segments

# ==================== AUDIO GENERATORS ====================

class DialogueGenerator:
    def __init__(self):
        self.model = None
        self.classifier = None
    
    def initialize(self):
        """Load models once."""
        if self.model is None:
            logger.info("Loading ChatterboxTurboTTS model...")
            self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
            logger.info("Loading emotion classifier...")
            self.classifier = pipeline("text-classification", 
                                      model="j-hartmann/emotion-english-distilroberta-base", 
                                      top_k=None)
    
    def unload(self):
        """Unload models from memory."""
        if self.model is not None:
            logger.info("Unloading ChatterboxTurboTTS model...")
            del self.model
            self.model = None
        if self.classifier is not None:
            del self.classifier
            self.classifier = None
        torch.cuda.empty_cache()
    
    def detect_emotion(self, text: str) -> str:
        """Detect dominant emotion from text."""
        try:
            results = self.classifier(text)[0]
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            dominant = sorted_results[0]
            return dominant['label'].lower()
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}. Using neutral.")
            return "neutral"
    
    def generate(self, segment: Dict, gen_count: int) -> List[torch.Tensor]:
        """Generate N audio versions for a dialogue segment."""
        self.initialize()
        
        speaker_name = segment['speaker']
        text = segment['text']
        original_text = segment['original_text']
        
        # Get voice path
        voice_path = VOICE_PATHS.get(speaker_name, DEFAULT_VOICE_PATH)
        if not os.path.exists(voice_path):
            logger.warning(f"Voice file not found: {voice_path}. Using default.")
            voice_path = DEFAULT_VOICE_PATH
        
        # Detect emotion
        emotion = self.detect_emotion(text)
        params = EMOTION_PARAMETERS.get(emotion, EMOTION_PARAMETERS['neutral'])
        
        # Strip parentheses-enclosed tags (emotion/delivery markers) from text
        # These should not be spoken aloud
        text_for_tts = re.sub(r'\s*\([^)]+\)\s*', ' ', text).strip()
        text_for_tts = re.sub(r'\s+', ' ', text_for_tts)
        
        logger.info(f"Generating {gen_count} versions for '{speaker_name}': {text_for_tts[:40]}... "
                   f"(emotion: {emotion})")
        
        audio_tensors = []
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
                logger.info(f"  Generated version {gen_idx + 1}/{gen_count} (modifiers: {exaggeration_mult}x, {cfg_weight_mult}x, {temperature_mult}x)")
            except Exception as e:
                logger.error(f"Failed to generate version {gen_idx + 1}: {e}")
                continue
        
        return audio_tensors

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
    
    dialogue_gen = DialogueGenerator()
    
    for segment in dialogue_segments:
        try:
            audio_tensors = dialogue_gen.generate(segment, args.gen_count)
            sample_rate = dialogue_gen.model.sr
            
            # Save generated audio files
            for gen_idx, audio_tensor in enumerate(audio_tensors):
                filepath = save_audio_file(audio_tensor, segment, gen_idx, sample_rate, args.output_dir)
                if filepath:
                    total_files += 1
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
