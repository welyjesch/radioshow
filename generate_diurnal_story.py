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
from chatterbox.tts_turbo import ChatterboxTurboTTS
from pydub import AudioSegment
from preset_provider import get_preset_batch_from_cloud

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

class DiurnalVoiceConfig:
    def __init__(self):
        self.voicebank_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diurnal_voicebank")
        self.deliveries = []
        self.load_deliveries()

    def load_deliveries(self):
        logger.info(f"Checking for diurnal voicebank at: {self.voicebank_dir}")
        if os.path.exists(self.voicebank_dir):
            try:
                # Get all files in the directory
                self.deliveries = sorted(os.listdir(self.voicebank_dir))
                logger.info(f"Loaded {len(self.deliveries)} delivery files from diurnal_voicebank")
            except Exception as e:
                logger.error(f"Error loading diurnal_voicebank: {e}")
        else:
            logger.warning(f"diurnal_voicebank directory not found at {self.voicebank_dir}")

    def get_voice_path(self, delivery_name: str) -> Optional[str]:
        if not delivery_name:
            return None
        
        # If the delivery_name is exactly one of our files, use it
        if delivery_name in self.deliveries:
            return os.path.join(self.voicebank_dir, delivery_name)
        
        # Otherwise, try to find a match (e.g., "angry" matching "angry.mp3" or "angry.wav")
        for d in self.deliveries:
            if d.startswith(delivery_name) and d.split('.')[-1] in ['mp3', 'wav']:
                return os.path.join(self.voicebank_dir, d)
        
        return None

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

# Initialize diurnal voice config
voice_config = DiurnalVoiceConfig()
GENERATION_LOG_PATH = os.path.join(OUTPUT_DIR, "generation_log.json")

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

# Punctuation characters that are valid emergency split points
_SPLIT_PUNCTUATION = (',', ';', ':', '—', '–')

def _split_at_punctuation(text, target_words=20, max_words=40):
    """Split a single oversized chunk (>max_words) at punctuation boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    chunks = []
    current_start = 0
    
    while current_start < len(words):
        remaining = words[current_start:]
        if len(remaining) <= max_words:
            chunks.append(' '.join(remaining))
            break
        
        split_idx = None
        for i in range(target_words - 1, min(len(remaining), max_words)):
            word = remaining[i]
            if any(word.endswith(p) for p in _SPLIT_PUNCTUATION):
                split_idx = i + 1
                break
        
        if split_idx is None:
            split_idx = max_words
        
        chunks.append(' '.join(remaining[:split_idx]))
        current_start += split_idx
    
    return chunks if chunks else [text]

def merge_short_sentences(sentences, min_words=20, max_words=40):
    """Merge adjacent short sentences until their combined word count exceeds min_words."""
    if not sentences:
        return []
    
    merged = []
    buffer = []
    buffer_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        buffer.append(sentence)
        buffer_word_count += sentence_word_count
        
        if buffer_word_count > min_words:
            merged.append(' '.join(buffer))
            buffer = []
            buffer_word_count = 0
    
    if buffer:
        merged.append(' '.join(buffer))
    
    final = []
    for chunk in merged:
        if len(chunk.split()) > max_words:
            final.extend(_split_at_punctuation(chunk, min_words, max_words))
        else:
            final.append(chunk)
    
    return final

# ==================== SCRIPT PARSER ====================

def process_dialogue_block(lines, speaker, seq_counter, segments):
    """Helper to process accumulated dialogue lines into segments."""
    all_sentences = []
    for individual_line in lines:
        stripped = individual_line.strip()
        if stripped:
            all_sentences.extend(split_sentences(stripped))
    
    merged_chunks = merge_short_sentences(all_sentences, min_words=20, max_words=40)
    
    for line_text in merged_chunks:
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
        
        sfx_match = re.match(r'^\[SFX:\s*(.+?)\]\s*(.*)', line_stripped)
        if sfx_match:
            if current_block_text_lines and current_speaker_name:
                seq_counter = process_dialogue_block(current_block_text_lines, current_speaker_name, seq_counter, segments)
                current_block_text_lines = []
                current_speaker_name = None
            
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
        
        speaker_match = re.match(r'^\[([A-Za-z0-9_ \-]+)\]\s*(.*)', line_stripped)
        if speaker_match:
            speaker_name = speaker_match.group(1).strip()
            # Exclude emotive tags and pauses from being detected as speakers
            if re.match(r'^(sigh|laughs|chuckles|whistles|coughs|pause:?\s*\d*)$', speaker_name.lower()):
                speaker_match = None # Treat as normal text, not a speaker change
            else:
                if current_block_text_lines and current_speaker_name:
                    seq_counter = process_dialogue_block(current_block_text_lines, current_speaker_name, seq_counter, segments)
                    current_block_text_lines = []
                
                current_speaker_name = speaker_name.upper()
                remaining_text = speaker_match.group(2).strip()
                
                text_for_tts = re.sub(r'\s*\([^)]+\)\s*', ' ', remaining_text).strip()
                text_for_tts = re.sub(r'\s+', ' ', text_for_tts).strip()
                
                if text_for_tts:
                    current_block_text_lines.append(text_for_tts)
                
                continue
        
        if line_stripped:
            current_block_text_lines.append(line_stripped)
            
    if current_block_text_lines and current_speaker_name:
        seq_counter = process_dialogue_block(current_block_text_lines, current_speaker_name, seq_counter, segments)
    
    logger.info(f"Parsed {len(segments)} segments from {script_path}")
    return segments

# ==================== AUDIO GENERATORS ====================

class DialogueGenerator:
    def __init__(self, api_key: str):
        self.model = None
        self.api_key = api_key
        self.voice_config = voice_config
    
    def initialize(self):
        if self.model is None:
            logger.info("Loading ChatterboxTurboTTS model...")
            self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    
    def unload(self):
        if self.model is not None:
            logger.info("Unloading ChatterboxTTS model...")
            del self.model
            self.model = None
        torch.cuda.empty_cache()
    
    def generate_batch(self, segments: List[Dict], gen_count: int) -> List[Tuple[List[torch.Tensor], List[dict]]]:
        self.initialize()
        
        # Use sequence IDs to pick the best voicebank filename
        seq_ids = [str(s['sequence']) for s in segments]
        preset_map = get_preset_batch_from_cloud(seq_ids, [s['original_text'] for s in segments], self.voice_config.deliveries, self.api_key)
        
        batch_results = []
        generation_log = {}
        
        for segment in segments:
            text = segment['text']
            original_text = segment['original_text']
            
            # Static CFG settings
            params = {"exaggeration": 1.0, "cfg_weight": 0.2, "temperature": 0.1}
            
            # Get the AI-selected preset from the map using sequence ID
            seq_key = str(segment['sequence'])
            preset_data = preset_map.get(seq_key, {})
            delivery_name = preset_data.get('preset')
            text_for_tts = text
            
            voice_path = self.voice_config.get_voice_path(delivery_name)
            
            if not voice_path:
                # Fallback to the first available file in the voicebank if no delivery specified or found
                if self.voice_config.deliveries:
                    fallback_file = self.voice_config.deliveries[0]
                    voice_path = os.path.join(self.voice_config.voicebank_dir, fallback_file)
                    logger.warning(f"Delivery '{delivery_name}' not found or not specified. Falling back to: {fallback_file}")
                else:
                    logger.error("Critical: No voice files found in diurnal_voicebank and no valid delivery specified.")
                    voice_path = None

            # Ensure we don't have parentheses-enclosed tags in the final TTS text
            text_for_tts = re.sub(r'\s*\([^)]+\)\s*', ' ', text_for_tts).strip()
            text_for_tts = re.sub(r'\s+', ' ', text_for_tts).strip()
            
            words = text_for_tts.split()
            chunks = [' '.join(words[i:i + 24]) for i in range(0, len(words), 24)]
            
            audio_tensors = []
            final_params = []
            for gen_idx in range(gen_count):
                try:
                    if not voice_path:
                        raise ValueError("No valid voice path available for generation")

                    chunk_audios = []
                    for chunk in chunks:
                        audio = self.model.generate(
                            chunk,
                            audio_prompt_path=voice_path,
                        )
                        chunk_audios.append(audio)
                    
                    logger.info(f"  [Gen {gen_idx + 1}] Text: {text_for_tts} | Delivery: {delivery_name}")

                    full_audio = torch.cat(chunk_audios, dim=-1)
                    
                    # Handle post-line pause (p_pause)
                    p_pause = preset_data.get('p_pause', 0.0)
                    p_pause = max(0.5, min(p_pause, 5.0)) if p_pause > 0 else 0.0
                    if p_pause > 0:
                        sample_rate = self.model.sr
                        silence_duration_samples = int(p_pause * sample_rate)
                        silence_tensor = torch.zeros(full_audio.shape[0], silence_duration_samples).to(full_audio.device)
                        full_audio = torch.cat([full_audio, silence_tensor], dim=-1)

                    audio_tensors.append(full_audio)
                    final_params.append({
                        'delivery': delivery_name,
                        'p_pause': p_pause
                    })
                except Exception as e:
                    logger.error(f"Failed to generate version {gen_idx + 1} for text '{text[:20]}...': {e}")
                    final_params.append(None)
                    continue
            
            # Log the final construction for this segment
            seq_key = f"{segment['sequence']}"
            generation_log[seq_key] = {
                "filepath": voice_path,
                "line": text_for_tts,
                "preset": delivery_name
            }
            
            batch_results.append((audio_tensors, final_params))
            
        # Save the log to a flat JSON file
        try:
            os.makedirs(os.path.dirname(GENERATION_LOG_PATH), exist_ok=True)
            with open(GENERATION_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(generation_log, f, indent=4)
            logger.info(f"Generation log saved to {GENERATION_LOG_PATH}")
        except Exception as e:
            logger.error(f"Failed to save generation log: {e}")
            
        return batch_results

# ==================== FILE HANDLER ====================

def save_audio_file(audio_tensor: torch.Tensor, segment: Dict, gen_idx: int, 
                    sample_rate: int, output_dir: str) -> Optional[str]:
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        seq_padded = str(segment['sequence']).zfill(4)
        gen_padded = str(gen_idx + 1).zfill(2)
        
        if segment['type'] == 'dialogue':
            # Since there's only one character, we use "Diurnal" as the speaker name in filename
            speaker_name = "Diurnal"
            text_part = truncate_text(segment['text'])
            filename = f"{seq_padded}-{gen_padded}_{speaker_name}_{text_part}.wav"
        else:  # sfx
            sfx_part = truncate_text(segment['sfx_description'])
            filename = f"{seq_padded}-{gen_padded}_SFX_{sfx_part}.wav"
        
        filepath = os.path.join(output_dir, filename)
        
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        ta.save(filepath, audio_tensor.cpu(), sample_rate)
        logger.info(f"Saved: {filename}")
        
        return filepath
    
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")
        return None

# ==================== MAIN ORCHESTRATION ====================

def main():
    parser = argparse.ArgumentParser(description="Generate audio from script for Diurnal voicebank.")
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
    
    try:
        segments = parse_script(args.script_path)
    except Exception as e:
        logger.error(f"Failed to parse script: {e}")
        sys.exit(1)
    
    if not segments:
        logger.error("No segments found in script.")
        sys.exit(1)
    
    dialogue_segments = [s for s in segments if s['type'] == 'dialogue']
    sfx_segments_to_process = [s for s in segments if s['type'] == 'sfx']
    
    logger.info(f"Found {len(dialogue_segments)} dialogue segments and {len(sfx_segments_to_process)} SFX segments")
    
    total_files = 0
    failed_count = 0
    
    logger.info("=" * 60)
    logger.info("PASS 1: Processing dialogue segments (Chatterbox)")
    logger.info("=" * 60)
    
    dialogue_gen = DialogueGenerator(api_key=args.apikey)
    
    batch_size = 20
    for i in range(0, len(dialogue_segments), batch_size):
        batch = dialogue_segments[i : i + batch_size]
        try:
            batch_results = dialogue_gen.generate_batch(batch, args.gen_count)
            sample_rate = dialogue_gen.model.sr
            
            for segment, (audio_tensors, final_params) in zip(batch, batch_results):
                for gen_idx, audio_tensor in enumerate(audio_tensors):
                    filepath = save_audio_file(audio_tensor, segment, gen_idx, sample_rate, args.output_dir)
                    if filepath:
                        seq_padded = str(segment['sequence']).zfill(4)
                        gen_padded = str(gen_idx + 1).zfill(2)
                        asset_id = f"{seq_padded}_dia_{gen_padded}_Diurnal"
                        
                        log_to_script_map({
                            'id': asset_id,
                            'params': final_params[gen_idx] if gen_idx < len(final_params) else None
                        }, os.path.join(args.output_dir, "script.json"))

                        total_files += 1
                        
                        filename = os.path.basename(filepath)
                        duration = len(AudioSegment.from_file(filepath)) / 1000.0
                        
                        gen_params = final_params[gen_idx] if gen_idx < len(final_params) else None
                        
                        metadata_entry = {
                            "filename": filename,
                            "transcription": segment['text'],
                            "duration": duration,
                            "exaggeration": gen_params['exaggeration'] if gen_params else None,
                            "cfg_weight": gen_params['cfg_weight'] if gen_params else None,
                            "temperature": gen_params['temperature'] if gen_params else None,
                            "delivery": gen_params['delivery'] if gen_params else None
                        }
                        logger.info(f"Metadata generated: {metadata_entry}")
                        
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
    dialogue_gen.unload()
    
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

    logger.info("=" * 60)
    logger.info(f"Generation complete!")
    logger.info(f"Total dialogue files generated: {total_files}")
    logger.info(f"Failed generations: {failed_count}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
