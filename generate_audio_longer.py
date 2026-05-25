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
#     "nltk",
#     "beautifulsoup4",
# ]
# ///

import os
import sys
import torch
import torchaudio as ta
import re
import json
import logging
import nltk
nltk.download('punkt_tab')
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from chatterbox.tts import ChatterboxTTS
from pydub import AudioSegment
from cloud_cfg_provider import get_cfg_settings_batch_from_cloud

DEFAULT_VOICE_KEY = "default_voice"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_audio")
DEFAULT_GENERATION_COUNT = 7

# ==================== UTILITIES ====================

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

logger = setup_logger()

class VoiceConfig:
    def __init__(self):
        self.paths = {}
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self.script_dir, "voice_paths.json")
        self.load_config()

    def load_config(self):
        logger.info(f"Checking for voice paths config at: {self.config_file}")
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    raw_paths = json.load(f)
                    self.paths = {k.upper(): os.path.join(self.script_dir, v) for k, v in raw_paths.items()}
                    logger.info(f"Loaded voice paths map:\n{json.dumps(self.paths, indent=2)}")
            except Exception as e:
                logger.error(f"Error loading voice_paths.json: {e}")
        else:
            logger.warning(f"voice_paths.json not found at {self.config_file}")

    def get_path(self, speaker_name: str) -> Optional[str]:
        return self.paths.get(speaker_name.upper())

def log_to_script_map(entry, map_path):
    """Helper to log generation metadata to a JSON map."""
    try:
        data = []
        if os.path.exists(map_path):
            with open(map_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        
        # Update existing entry or add new one
        updated = False
        for item in data:
            if item.get('id') == entry.get('id'):
                item.update(entry)
                updated = True
                break
        
        if not updated:
            data.append(entry)
            
        with open(map_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error updating script map: {e}")

# Initialize voice config
voice_config = VoiceConfig()

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
    """Split text into individual sentences using NLTK."""
    try:
        return nltk.sent_tokenize(text)
    except Exception as e:
        logger.error(f"NLTK sent_tokenize failed: {e}")
        return text.splitlines()


# ==================== SCRIPT PARSER ====================

def process_dialogue_block(lines, speaker, seq_counter, segments):
    """Helper to process accumulated dialogue lines into segments."""
    # Tokenize each line individually to preserve the script's natural
    # line structure. Joining all lines first causes NLTK's Punkt tokenizer
    # to treat newlines as whitespace, merging lines and re-splitting at
    # statistical boundaries which produces incoherent phrase fragments.
    split_lines = []
    for individual_line in lines:
        stripped = individual_line.strip()
        if stripped:
            split_lines.extend(split_sentences(stripped))
    
    for line_text in split_lines:
        if line_text:
            seq_counter += 1
            clean_sentence = re.sub(r'\[.*?\]', '', line_text).strip()
            clean_sentence = re.sub(r'\s+', ' ', clean_sentence)
            
            segments.append({
                'type': 'dialogue',
                'sequence': seq_counter,
                'speaker': speaker,
                'text': clean_sentence,
                'original_text': line_text,
                'explicit_emotion': None
            })
    return seq_counter

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
                seq_counter = process_dialogue_block(current_block_text_lines, current_speaker_name, seq_counter, segments)
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
                seq_counter = process_dialogue_block(current_block_text_lines, current_speaker_name, seq_counter, segments)
                current_block_text_lines = []
            
            # Set new speaker
            current_speaker_name = speaker_match.group(1).upper()
            remaining_text = speaker_match.group(2).strip()
            
            # Remove parenthesized content for TTS
            text_for_tts = re.sub(r'\s*\([^)]+\)\s*', ' ', remaining_text).strip()
            text_for_tts = re.sub(r'\s+', ' ', text_for_tts).strip()
            
            if text_for_tts:
                current_block_text_lines.append(text_for_tts)
            
            continue
        
        # Regular text line
        if line_stripped:
            current_block_text_lines.append(line_stripped)
    
    # Process remaining accumulated text
    if current_block_text_lines and current_speaker_name:
        seq_counter = process_dialogue_block(current_block_text_lines, current_speaker_name, seq_counter, segments)
    
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
            logger.info("Loading ChatterboxTTS model...")
            self.model = ChatterboxTTS.from_pretrained(device="cuda")
    
    def unload(self):
        """Unload models from memory."""
        if self.model is not None:
            logger.info("Unloading ChatterboxTTS model...")
            del self.model
            self.model = None
        torch.cuda.empty_cache()
    
    def generate_batch(self, segments: List[Dict], gen_count: int) -> List[Tuple[List[torch.Tensor], List[dict]]]:
        """Generate audio for a batch of dialogue segments."""
        self.initialize()
        
        # 1. Batch request CFG settings from cloud
        original_texts = [s['original_text'] for s in segments]
        cfg_map = get_cfg_settings_batch_from_cloud(original_texts, self.api_key)
        
        batch_results = []
        
        for segment in segments:
            speaker_name = segment['speaker']
            text = segment['text']
            original_text = segment['original_text']
            
            # Get voice path
            speaker_name_upper = speaker_name.upper()
            voice_path = voice_config.get_path(speaker_name_upper)
            
            if not voice_path or not os.path.exists(voice_path):
                fallback_path = voice_config.get_path(DEFAULT_VOICE_KEY)
                logger.warning(f"Voice file '{voice_path}' not found for speaker '{speaker_name}'. Falling back to default: '{fallback_path}'")
                voice_path = fallback_path
                
            if not voice_path or not os.path.exists(voice_path):
                logger.error(f"Critical: Resolved voice path is invalid or missing: '{voice_path}'.")

            # Get params from the batch map
            params = cfg_map.get(original_text, {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.7})
            
            # Strip parentheses-enclosed tags from text
            text_for_tts = re.sub(r'\s*\([^)]+\)\s*', ' ', text).strip()
            text_for_tts = re.sub(r'\s+', ' ', text_for_tts).strip()
            
            # Split text into chunks of maximum 24 words
            words = text_for_tts.split()
            chunks = [' '.join(words[i:i + 24]) for i in range(0, len(words), 24)]
            
            audio_tensors = []
            final_params = []
            for gen_idx in range(gen_count):
                try:
                    modified_exaggeration = params['exaggeration']
                    modified_cfg_weight = params['cfg_weight']
                    modified_temperature = params['temperature']
                    
                    chunk_audios = []
                    for chunk in chunks:
                        audio = self.model.generate(
                            chunk,
                            audio_prompt_path=voice_path,
                            exaggeration=modified_exaggeration,
                            cfg_weight=modified_cfg_weight,
                            temperature=modified_temperature
                        )
                        chunk_audios.append(audio)
                    
                    # Log the source text and parameters used for this generation
                    logger.info(f"  [Gen {gen_idx + 1}] Text: {text_for_tts} | Params: exaggeration={modified_exaggeration}, cfg_weight={modified_cfg_weight}, temperature={modified_temperature}")

                    full_audio = torch.cat(chunk_audios, dim=-1)
                    audio_tensors.append(full_audio)
                    final_params.append({
                        'exaggeration': modified_exaggeration,
                        'cfg_weight': modified_cfg_weight,
                        'temperature': modified_temperature
                    })
                except Exception as e:
                    logger.error(f"Failed to generate version {gen_idx + 1} for text '{text[:20]}...': {e}")
                    final_params.append(None)
                    continue
            
            batch_results.append((audio_tensors, final_params))
            
        return batch_results

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
        
        # Ensure tensor is 2D [channels, time]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        ta.save(filepath, audio_tensor.cpu(), sample_rate)
        logger.info(f"Saved: {filename}")
        
        # Log to script map
        log_to_script_map({
            'id': f"{seq_padded}_{'dia' if segment['type'] == 'dialogue' else 'sfx'}_{gen_padded}_{segment.get('speaker', 'N/A').replace(' ', '_')}",
            'filename': filename,
            'text': segment.get('text', segment.get('sfx_description', '')),
            'speaker': segment.get('speaker', 'N/A'),
            'sequence': segment['sequence'],
            'gen_idx': gen_idx + 1,
            'params': None # Params are handled in the main loop for dialogue
        }, os.path.join(output_dir, "script.json"))
        
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
    
    # Process dialogue segments in batches of 20
    batch_size = 20
    for i in range(0, len(dialogue_segments), batch_size):
        batch = dialogue_segments[i : i + batch_size]
        try:
            # generate_batch returns a list of (audio_tensors, final_params) for each segment in the batch
            batch_results = dialogue_gen.generate_batch(batch, args.gen_count)
            sample_rate = dialogue_gen.model.sr
            
            for segment, (audio_tensors, final_params) in zip(batch, batch_results):
                # Save generated audio files
                for gen_idx, audio_tensor in enumerate(audio_tensors):
                    filepath = save_audio_file(audio_tensor, segment, gen_idx, sample_rate, args.output_dir)
                    if filepath:
                        # Update the log entry with actual params
                        seq_padded = str(segment['sequence']).zfill(4)
                        gen_padded = str(gen_idx + 1).zfill(2)
                        speaker_name = segment['speaker'].replace(' ', '_')
                        asset_id = f"{seq_padded}_dia_{gen_padded}_{speaker_name}"
                        
                        log_to_script_map({
                            'id': asset_id,
                            'params': final_params[gen_idx] if gen_idx < len(final_params) else None
                        }, os.path.join(args.output_dir, "script.json"))

                        total_files += 1
                        
                        # Log metadata
                        filename = os.path.basename(filepath)
                        duration = len(AudioSegment.from_file(filepath)) / 1000.0
                        
                        # Get the specific CFG values for this generation index
                        gen_params = final_params[gen_idx] if gen_idx < len(final_params) else None
                        
                        metadata_entry = {
                            "filename": filename,
                            "transcription": segment['text'],
                            "duration": duration,
                            "exaggeration": gen_params['exaggeration'] if gen_params else None,
                            "cfg_weight": gen_params['cfg_weight'] if gen_params else None,
                            "temperature": gen_params['temperature'] if gen_params else None
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
            logger.error(f"Failed to process dialogue batch starting at index {i}: {e}")
            failed_count += len(batch)
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