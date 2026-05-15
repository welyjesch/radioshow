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

def get_audio_files(output_dir: str = "generated_audio") -> Dict[int, List[str]]:
    """Scan output directory and group audio files by sequence number."""
    if not os.path.exists(output_dir):
        return {}
    
    files_by_sequence = {}
    
    try:
        for filename in sorted(os.listdir(output_dir)):
            filepath = os.path.join(output_dir, filename)
            # Only process files (not directories)
            if os.path.isfile(filepath) and Path(filename).suffix.lower() in SUPPORTED_FORMATS:
                seq_num = extract_sequence_num(filename)
                if seq_num is not None:
                    if seq_num not in files_by_sequence:
                        files_by_sequence[seq_num] = []
                    files_by_sequence[seq_num].append(filename)
    except Exception as e:
        st.error(f"Error scanning directory {output_dir}: {e}")
        return {}
    
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
    return label

def get_file_label_with_duration(filename: str, output_dir: str) -> str:
    """Create a readable label for a file with duration."""
    label = get_file_label(filename)
    filepath = os.path.join(output_dir, filename)
    duration = get_audio_duration(filepath)
    return f"{label} ({duration:.2f}s)"

def concatenate_audio(concatenation_list: List, output_dir: str) -> Tuple[Optional[bytes], str]:
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
                filepath = os.path.join(output_dir, item)
                if os.path.exists(filepath):
                    audio = AudioSegment.from_file(filepath)
                    combined += audio
                else:
                    st.warning(f"File not found: {filepath}")
        
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

# Initialize session state for output directory
if "output_dir" not in st.session_state:
    st.session_state.output_dir = "generated_audio"

if "audio_files" not in st.session_state:
    st.session_state.audio_files = {}

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

if "pending_silence" not in st.session_state:
    st.session_state.pending_silence = {}

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    custom_output_dir = st.text_input(
        "Output Directory",
        value=st.session_state.output_dir,
        help="Directory where generated audio files are stored"
    )
    if custom_output_dir != st.session_state.output_dir:
        st.session_state.output_dir = custom_output_dir
    
    if st.button("🔄 Refresh Files", use_container_width=True):
        st.rerun()
    
    # Debug info
    with st.expander("📊 Debug Info"):
        output_dir = st.session_state.output_dir
        st.write(f"**Output Dir:** `{output_dir}`")
        st.write(f"**Dir Exists:** {os.path.exists(output_dir)}")
        
        # Scan and show what's actually in the directory
        if os.path.exists(output_dir):
            all_items = os.listdir(output_dir)
            files_only = [f for f in all_items if os.path.isfile(os.path.join(output_dir, f))]
            audio_files = [f for f in files_only if Path(f).suffix.lower() in SUPPORTED_FORMATS]
            
            st.write(f"**Total items in dir:** {len(all_items)}")
            st.write(f"**Files (not dirs):** {len(files_only)}")
            st.write(f"**Audio files (.wav/.mp3):** {len(audio_files)}")
            
            if audio_files:
                st.write("**Audio Files Found:**")
                for f in audio_files[:10]:  # Show first 10
                    st.code(f)
                if len(audio_files) > 10:
                    st.write(f"... and {len(audio_files) - 10} more")

# Load audio files
output_dir = st.session_state.output_dir
audio_files = get_audio_files(output_dir)
st.session_state.audio_files = audio_files

# Main content
if not audio_files:
    st.warning("No audio files found in the output directory. Run generate_audio.py first.")
    st.info(f"Looking in: `{output_dir}`")
else:
    st.info(f"Found {len(audio_files)} sequence(s) with audio files")
    
    # Sequence selection section
    st.header("📋 Audio Selection & Preview")
    
    # Table header
    col1, col2, col3, col4 = st.columns([1.2, 2.5, 1.5, 1])
    with col1:
        st.write("**Seq**")
    with col2:
        st.write("**Audio File Options**")
    with col3:
        st.write("**Add Silence**")
    with col4:
        st.write("**Preview**")
    
    st.divider()
    
    # Table rows
    for seq_num in audio_files:
        files = audio_files[seq_num]
        
        col1, col2, col3, col4 = st.columns([1.2, 2.5, 1.5, 1])
        
        with col1:
            st.write(f"**{seq_num:04d}**")
        
        with col2:
            # Get current selection or default to first
            if seq_num not in st.session_state.selections:
                st.session_state.selections[seq_num] = files[0]
            
            # Find index of current selection
            current_selection = st.session_state.selections[seq_num]
            try:
                current_idx = files.index(current_selection)
            except ValueError:
                current_idx = 0
            
            # Selectbox for file selection
            selected_file = st.selectbox(
                label="Select audio:",
                options=files,
                index=current_idx,
                format_func=lambda f: get_file_label_with_duration(f, output_dir),
                key=f"file_select_{seq_num}",
                label_visibility="collapsed"
            )
            
            # Update selection
            st.session_state.selections[seq_num] = selected_file
        
        with col3:
            if st.button("➕ Add Silence", key=f"silence_btn_{seq_num}", use_container_width=True):
                st.session_state.silence_modals[seq_num] = True
                st.rerun()
            
            # Silence modal
            if st.session_state.silence_modals.get(seq_num, False):
                silence_seconds = st.number_input(
                    f"Silence (seconds):",
                    min_value=0.1,
                    max_value=60.0,
                    value=1.0,
                    step=0.1,
                    key=f"silence_input_{seq_num}"
                )
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅", key=f"silence_confirm_{seq_num}", use_container_width=True):
                        st.session_state.silence_modals[seq_num] = False
                        st.session_state.pending_silence[seq_num] = silence_seconds
                        st.success(f"Silence: {silence_seconds}s")
                        st.rerun()
                with col_cancel:
                    if st.button("❌", key=f"silence_cancel_{seq_num}", use_container_width=True):
                        st.session_state.silence_modals[seq_num] = False
                        st.rerun()
        
        with col4:
            if st.button("▶️ Play", key=f"play_btn_{seq_num}", use_container_width=True):
                selected_file = st.session_state.selections[seq_num]
                filepath = os.path.join(output_dir, selected_file)
                
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        st.session_state.current_player = f.read()
                    st.session_state.play_key += 1
                    st.rerun()
                else:
                    st.error(f"File not found: {filepath}")
        
        st.divider()
    
    # Show pending silences
    if st.session_state.pending_silence:
        st.write("**Pending Silence Additions:**")
        for seq_num, duration in sorted(st.session_state.pending_silence.items()):
            st.info(f"Silence ({duration}s) after sequence {seq_num:04d}")
    
    # Build concatenation list
    st.header("🎵 Concatenation Preview")
    
    concatenation_list = []
    total_duration = 0.0
    
    for seq_num in sorted(audio_files.keys()):
        # Add selected file
        selected_file = st.session_state.selections[seq_num]
        concatenation_list.append(selected_file)
        filepath = os.path.join(output_dir, selected_file)
        duration = get_audio_duration(filepath)
        total_duration += duration
        
        # Add silence if pending
        if seq_num in st.session_state.pending_silence:
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
            filepath = os.path.join(output_dir, item)
            duration = get_audio_duration(filepath)
            preview_items.append(f"{idx}. {item} ({duration:.2f}s)")
    
    if preview_items:
        st.write("**Order:**")
        for item in preview_items:
            st.write(item)
        
        st.metric("Total Duration", format_duration_total(total_duration))
    
    # Player section
    st.divider()
    st.header("▶️ Full Preview")
    
    player_col1, player_col2 = st.columns([4, 1])
    
    with player_col2:
        if st.button("🎬 Play All", use_container_width=True, type="primary"):
            audio_bytes, _ = concatenate_audio(concatenation_list, output_dir)
            if audio_bytes:
                st.session_state.current_player = audio_bytes
                st.session_state.play_key += 1
                st.rerun()
    
    with player_col1:
        if st.session_state.current_player:
            st.audio(st.session_state.current_player, format="audio/wav", key=f"player_{st.session_state.play_key}")
    
    # Export section
    st.divider()
    st.header("💾 Export Final Audio")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"**Total Duration:** {format_duration_total(total_duration)}")
    
    with col2:
        if st.button("🚀 Export", use_container_width=True, type="primary"):
            with st.spinner("Concatenating audio..."):
                audio_bytes, filename = concatenate_audio(concatenation_list, output_dir)
                
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
                    output_path = os.path.join(output_dir, filename)
                    try:
                        with open(output_path, "wb") as f:
                            f.write(audio_bytes)
                        st.info(f"Also saved to: {output_path}")
                    except Exception as e:
                        st.error(f"Error saving to {output_path}: {e}")

st.divider()
st.caption("Audio Concatenator v1.0 | Made with Streamlit")
