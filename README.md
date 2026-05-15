# Audio Generation System: Chatterbox + AudioLDM2

Generate dialogue and sound effects audio from a script file with emotion-based voice modulation.

## Quick Start

Due to dependency conflicts between the TTS and SFX models, the process is split into two stages.

### Stage 1: Dialogue Generation
```bash
# Generate dialogue and create sfx_tasks.json
uv run generate_audio.py sample_script.txt --output-dir my_audio
```

### Stage 2: SFX Generation
```bash
# Generate SFX based on the tasks file created in Stage 1
uv run generate_sfx.py --tasks-file my_audio/sfx_tasks.json --output-dir my_audio
```

## Script Format

Create a `.txt` file with speaker dialogue and sound effects:

```
[SPEAKER_NAME] Dialogue text here.
[SPEAKER_NAME] (emotion) More dialogue with emotion tag.
[SFX: sound effect description]
```

### Supported Emotions
Explicit emotion tags (in parentheses) control voice modulation:
- `excited`, `happy`, `enthusiastic`
- `sad`, `angry`, `frustrated`
- `calm`, `neutral`, `confused`
- `surprised`, `tired`, `worried`

Or emotion is auto-detected from dialogue text if no explicit tag is present.

### Examples

```
[NARRATOR] The story begins.
[ELIAS] (angry) This is unacceptable!
[SFX: explosion with deep rumble, dust settling]
[BOARD_MEMBER_VANE] (calm) Let's discuss this rationally.
```

## Voice Configuration

Create `voice_paths.json` in the same directory with speaker-to-voice mappings:

```json
{
  "NARRATOR": "voices/narrator.wav",
  "ELIAS": "voices/elias.wav",
  "BOARD_MEMBER_VANE": "voices/roger.wav"
}
```

Voice files should be `.wav` files containing natural speech samples (3-5 seconds) used as reference for voice cloning by Chatterbox.

If a speaker isn't in the mapping, the system will use the first reference voice found as fallback.

## Output Format

Generated files use naming convention:
```
<sequence>_<speaker>_<text_summary>.wav
<sequence>_SFX_<description_summary>_<gen>.wav
```

Examples:
- `<0001-01>_NARRATOR_The story begins.wav`
- `<0002-01>_SFX_explosion_with_deep_1.wav`

**Sequence**: 4-digit counter across all segments (dialogue + SFX)
**Generation**: Version number for SFX (1, 2, 3, etc.)

## Parameters

### Dialogue Generation
```bash
uv run generate_audio.py <script> [options]

Positional:
  script                Script file path (.txt)

Options:
  --gen-count, -c N     Generate N versions per segment (default: 1)
  --output-dir, -o DIR  Output directory (default: generated_audio)
```

### SFX Generation
```bash
uv run generate_sfx.py --tasks-file <path> --output-dir <path>

Options:
  --tasks-file          Path to sfx_tasks.json created by generate_audio.py
  --output-dir          Directory to save generated SFX
```

## How It Works

1. **Parse Script**: Extracts speakers, dialogue, emotions, and SFX tags
2. **Dialogue Generation**: 
   - Detects or uses explicit emotion
   - Maps emotion to TTS parameters (CFG weight, exaggeration, temperature)
   - Uses Chatterbox model with speaker voice reference
   - Exports `sfx_tasks.json` for the SFX stage
3. **SFX Generation**:
   - Reads `sfx_tasks.json`
   - Generates audio from text descriptions using AudioLDM2
   - Uses AudioLDM2 diffusion model
   - Creates multiple versions by varying random seed
4. **File Output**: 
   - Saves each generation as individual WAV file
   - Names reflect sequence, generation count, speaker/type, and content summary

## Features

- ✅ Emotion-based voice modulation for dialogue
- ✅ Multiple generations per segment (for quality control, backups, variations)
- ✅ Mix dialogue and sound effects in single script
- ✅ Individual file output (no concatenation)
- ✅ Error recovery (skips failed segments, continues batch)
- ✅ Progress logging and summary reporting

## Requirements

- Python 3.12+
- GPU with CUDA support (for Chatterbox and AudioLDM2)
- ~20 GB VRAM recommended for both models
- uv package manager

## Installation

All dependencies are specified in the uv script header. Just run:

```bash
uv run generate_audio.py --help
```

## Troubleshooting

**CUDA not available**: Ensure PyTorch is installed with CUDA support for your GPU
**Voice file not found**: Check `voice_paths.json` paths are correct and files exist
**Out of memory**: Reduce `--gen-count` or process script in smaller chunks
**Model download fails**: Models will auto-download on first run; ensure internet connection

## Performance

- Single generation: ~30-60 seconds per 30-second script (depending on segment count)
- Emotion detection: ~1 second per segment
- SFX generation: ~15-30 seconds per description
- Multiple generations: Linear scaling (3x gen-count ≈ 3x time)

---

# Stage 3: Audio Concatenation (Streamlit Web UI)

After generating dialogue and SFX audio, use the Streamlit web UI to select, preview, and concatenate audio segments.

## Running the Concatenator

```bash
streamlit run concatenate_audio.py
```

This launches a web UI at `http://localhost:8501`

## Features

### Sequence Selection
- View all generated audio files grouped by sequence number
- For each sequence, select from available versions (if multiple generations were created)
- See duration of each audio file
- Radio button interface for easy selection

### Adding Silence
- Click "Add Silence After Sequence X" button to insert silence gaps
- Modal dialog prompts for silence duration (0.1 to 60 seconds)
- Silence appears as a marker in the concatenation preview
- Useful for natural pauses between dialogue and SFX

### Audio Preview
- **Play Selected**: Preview individual sequence audio files
- **Play All**: Preview the complete concatenation
- Built-in player at the bottom of the page

### Export
- Shows total duration of final concatenated audio
- **Create & Download** button to generate final audio
- Automatically saves as `radioshow_output_<YYYYMMDD_HHMMSS>.wav`
- File saved both locally and available for download

## Workflow Example

1. Run dialogue generation:
   ```bash
   uv run generate_audio.py sample_script.txt
   ```

2. Run SFX generation:
   ```bash
   uv run generate_sfx.py --tasks-file generated_audio/sfx_tasks.json
   ```

3. Open concatenator:
   ```bash
   streamlit run concatenate_audio.py
   ```

4. For each sequence:
   - Select your preferred version from radio options
   - Click "Add Silence" to insert gaps if needed
   - Confirm silence duration in modal

5. Click "Play All" to preview the complete concatenation

6. Click "Create & Download Final Audio" to generate the final MP3/WAV

## Concatenation List Format

Internally, each concatenation is represented as a list:
```python
[
    "<0001-01>_NARRATOR_intro.wav",
    {"add_silence": 1.5},
    "<0002-01>_SFX_explosion.wav",
    {"add_silence": 2.0},
    "<0003-01>_NARRATOR_outro.wav"
]
```

The processor iterates through this list:
- String items: Load and append the audio file
- `{"add_silence": N}`: Generate and append N seconds of silence

## Output

Final concatenated audio is saved as:
- Format: WAV (lossless, full quality)
- Filename: `radioshow_output_<YYYYMMDD_HHMMSS>.wav`
- Location: Same directory as input files (configurable in sidebar)
