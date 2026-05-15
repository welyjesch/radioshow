# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "streamlit>=1.28.0",
#     "pydub>=0.25.1",
#     "numpy>=1.24.0,<2.0.0",
#     "librosa==0.11.0",
# ]
# ///

import streamlit as st
import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pydub import AudioSegment
from pydub.generators import Sine
import numpy as np
import librosa

# ==================== CONFIGURATION ====================

OUTPUT_DIR = st.session_state.get("output_dir", "generated_audio")
SUPPORTED_FORMATS = {".wav", ".mp3"}

# ==================== UTILITY FUNCTIONS ====================

def extract_sequence_num(filename: str) -> Optional[int]:
    """Extract sequence number from filename format: <XXXX-XX>_..."""
    match = re.match(r"<(\d+)", filename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def get_audio_files() -> Dict[int, List[str]]:
    """Scan output directory and group audio files by sequence number."""
    if not os.path.exists(OUTPUT_DIR):
        return {}
    
    files_by_sequence = {}
    
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        if Path(filename).suffix.lower() in SUPPORTED_FORMATS:
            seq_num = extract_sequence_num(filename)
            if seq_num is not None:
                if seq_num not in files_by_sequence:
                    files_by_sequence[seq_num] = []
                files_by_sequence[seq_num].append(filename)
    
    return dict(sorted(files_by_sequence.items()))

def get_audio_duration(filepath: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        audio = AudioSegment.from_file(filepath)
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except Exception as e:
        st.error(f"Error reading {filepath}: {e}")
        return 0.0

def generate_silence(duration_seconds: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate silence as numpy array."""
    return np.zeros(int(duration_seconds * sample_rate))

def get_file_label(filename: str) -> str:
    """Create a readable label for a file."""
    # Remove sequence number and extension
    label = re.sub(r"^<\d+-\d+>_", "", filename)
    label = Path(label).stem
    # Get duration
    filepath = os.path.join(OUTPUT_DIR, filename)
    duration = get_audio_duration(filepath)
    return f"{label} ({duration:.2f}s)"

def concatenate_audio(concatenation_list: List) -> Tuple[Optional[bytes], str]:
    """
    Concatenate audio files and silence segments.
    
    concatenation_list contains either:
    - Filenames (strings) to concatenate
    - {"add_silence": seconds} to add silence
    """
    try:
        combined = AudioSegment.empty()
        
        for item in concatenation_list:
            if isinstance(item, dict) and "add_silence" in item:
                # Generate silence
                seconds = item["add_silence"]
                silence = AudioSegment.silent(duration=int(seconds * 1000))
                combined += silence
            else:
                # Load audio file
                filepath = os.path.join(OUTPUT_DIR, item)
                if os.path.exists(filepath):
                    audio = AudioSegment.from_file(filepath)
                    combined += audio
        
        # Export to bytes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"radioshow_output_{timestamp}.wav"
        
        # Export to memory
        export_bytes = combined.export(format="wav")
        
        return export_bytes.getvalue(), filename
    except Exception as e:
        st.error(f"Error concatenating audio: {e}")
        return None, ""

def format_duration_total(total_seconds: float) -> str:
    """Format total duration as MM:SS."""
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes}:{seconds:02d}"

# ==================== STREAMLIT APP ====================

st.set_page_config(page_title="Audio Concatenator", layout="wide")

st.title("🎙️ Audio Concatenator")

# Initialize session state
if "audio_files" not in st.session_state:
    st.session_state.audio_files = get_audio_files()

if "selections" not in st.session_state:
    st.session_state.selections = {}

if "concatenation_list" not in st.session_state:
    st.session_state.concatenation_list = []

if "silence_modals" not in st.session_state:
    st.session_state.silence_modals = {}

if "current_player" not in st.session_state:
    st.session_state.current_player = None

if "play_key" not in st.session_state:
    st.session_state.play_key = 0

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    custom_output_dir = st.text_input(
        "Output Directory",
        value=OUTPUT_DIR,
        help="Directory where generated audio files are stored"
    )
    if custom_output_dir != OUTPUT_DIR:
        st.session_state.output_dir = custom_output_dir
        OUTPUT_DIR = custom_output_dir
        st.session_state.audio_files = get_audio_files()
    
    if st.button("🔄 Refresh Files", use_container_width=True):
        st.session_state.audio_files = get_audio_files()
        st.rerun()

# Main content
audio_files = st.session_state.audio_files

if not audio_files:
    st.warning("No audio files found in the output directory. Run generate_audio.py first.")
else:
    st.info(f"Found {len(audio_files)} sequence(s) with audio files")
    
    # Sequence selection section
    st.header("📋 Select Audio for Each Sequence")
    
    for seq_num in audio_files:
        files = audio_files[seq_num]
        
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"Sequence {seq_num:04d}")
            
            with col2:
                if seq_num in st.session_state.selections:
                    selected_file = st.session_state.selections[seq_num]
                    duration = get_audio_duration(os.path.join(OUTPUT_DIR, selected_file))
                    st.metric("Duration", f"{duration:.2f}s")
            
            # Radio options for this sequence
            options = [get_file_label(f) for f in files]
            
            # Get current selection or default to first
            if seq_num not in st.session_state.selections:
                st.session_state.selections[seq_num] = files[0]
            
            selected_label = get_file_label(st.session_state.selections[seq_num])
            
            selected_idx = options.index(selected_label) if selected_label in options else 0
            
            selection = st.radio(
                "Options:",
                options=options,
                index=selected_idx,
                key=f"seq_{seq_num}",
                horizontal=True,
                label_visibility="collapsed"
            )
            
            # Update selection
            selected_file_idx = options.index(selection)
            st.session_state.selections[seq_num] = files[selected_file_idx]
            
            # Add silence button for this sequence
            if st.button(f"➕ Add Silence After Sequence {seq_num:04d}", key=f"silence_btn_{seq_num}"):
                st.session_state.silence_modals[seq_num] = True
                st.rerun()
            
            # Silence modal dialog
            if st.session_state.silence_modals.get(seq_num, False):
                st.divider()
                silence_col1, silence_col2, silence_col3 = st.columns([2, 1, 1])
                
                with silence_col1:
                    silence_seconds = st.number_input(
                        f"Silence duration (seconds):",
                        min_value=0.1,
                        max_value=60.0,
                        value=1.0,
                        step=0.1,
                        key=f"silence_input_{seq_num}"
                    )
                
                with silence_col2:
                    if st.button("✅ Confirm", key=f"silence_confirm_{seq_num}"):
                        # Store silence action to be applied after this sequence
                        st.session_state.silence_modals[seq_num] = False
                        # Mark that silence should be added (will be built into concatenation list)
                        if "pending_silence" not in st.session_state:
                            st.session_state.pending_silence = {}
                        st.session_state.pending_silence[seq_num] = silence_seconds
                        st.success(f"Will add {silence_seconds}s silence after sequence {seq_num:04d}")
                
                with silence_col3:
                    if st.button("❌ Cancel", key=f"silence_cancel_{seq_num}"):
                        st.session_state.silence_modals[seq_num] = False
                        st.rerun()
    
    # Build concatenation list
    st.divider()
    st.header("🎵 Concatenation Preview")
    
    concatenation_list = []
    total_duration = 0.0
    
    for seq_num in sorted(audio_files.keys()):
        # Add selected file
        selected_file = st.session_state.selections[seq_num]
        concatenation_list.append(selected_file)
        filepath = os.path.join(OUTPUT_DIR, selected_file)
        duration = get_audio_duration(filepath)
        total_duration += duration
        
        # Add silence if pending
        if "pending_silence" in st.session_state and seq_num in st.session_state.pending_silence:
            silence_duration = st.session_state.pending_silence[seq_num]
            concatenation_list.append({"add_silence": silence_duration})
            total_duration += silence_duration
    
    st.session_state.concatenation_list = concatenation_list
    
    # Display concatenation order
    preview_items = []
    for idx, item in enumerate(concatenation_list, 1):
        if isinstance(item, dict) and "add_silence" in item:
            seconds = item["add_silence"]
            preview_items.append(f"{idx}. **Silence** ({seconds}s)")
        else:
            filepath = os.path.join(OUTPUT_DIR, item)
            duration = get_audio_duration(filepath)
            preview_items.append(f"{idx}. {item} ({duration:.2f}s)")
    
    if preview_items:
        st.write("**Concatenation Order:**")
        for item in preview_items:
            st.write(item)
        
        st.metric("Total Duration", format_duration_total(total_duration))
    
    # Player section (bottom bar)
    st.divider()
    st.header("▶️ Preview Player")
    
    player_col1, player_col2, player_col3 = st.columns([2, 1, 1])
    
    with player_col1:
        selected_to_preview = st.selectbox(
            "Select file to preview:",
            options=sorted(audio_files.keys()),
            format_func=lambda x: f"Sequence {x:04d}",
            key="preview_selector"
        )
    
    with player_col2:
        if st.button("▶️ Play Selected", use_container_width=True):
            seq_num = selected_to_preview
            selected_file = st.session_state.selections[seq_num]
            filepath = os.path.join(OUTPUT_DIR, selected_file)
            
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    st.session_state.current_player = f.read()
                st.session_state.play_key += 1
                st.rerun()
    
    with player_col3:
        if st.button("🎬 Play All", use_container_width=True):
            # Create full concatenation
            audio_bytes, _ = concatenate_audio(concatenation_list)
            if audio_bytes:
                st.session_state.current_player = audio_bytes
                st.session_state.play_key += 1
                st.rerun()
    
    if st.session_state.current_player:
        st.audio(st.session_state.current_player, format="audio/wav", key=f"player_{st.session_state.play_key}")
    
    # Export section
    st.divider()
    st.header("💾 Export Final Audio")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"**Total Duration:** {format_duration_total(total_duration)}")
    
    with col2:
        if st.button("🚀 Create & Download Final Audio", use_container_width=True, type="primary"):
            with st.spinner("Concatenating audio..."):
                audio_bytes, filename = concatenate_audio(concatenation_list)
                
                if audio_bytes:
                    st.success(f"✅ Audio created: {filename}")
                    st.download_button(
                        label="📥 Download",
                        data=audio_bytes,
                        file_name=filename,
                        mime="audio/wav",
                        use_container_width=True
                    )
                    
                    # Save to output directory as well
                    output_path = os.path.join(OUTPUT_DIR, filename)
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)
                    st.info(f"Also saved to: {output_path}")

st.divider()
st.caption("Audio Concatenator v1.0 | Made with Streamlit")
