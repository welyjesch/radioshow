# Audio Generation System: Chatterbox + AudioLDM2

Generate dialogue and sound effects audio from a script file with emotion-based voice modulation.

## Quick Start

```bash
# Generate single version of each segment
uv run generate_audio.py sample_script.txt

# Generate 3 versions of each segment (for quality control)
uv run generate_audio.py sample_script.txt --gen-count 3

# Specify output directory
uv run generate_audio.py sample_script.txt -c 2 --output-dir my_audio
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
<sequence>-<generation>_<speaker>_<text_summary>.wav
<sequence>-<generation>_SFX_<description_summary>.wav
```

Examples:
- `0001-01_NARRATOR_The story begins.wav`
- `0001-02_NARRATOR_The story begins.wav` (second generation of same line)
- `0002-01_SFX_explosion with deep.wav`
- `0002-02_SFX_explosion with deep.wav` (alternate version)

**Sequence**: 4-digit counter across all segments (dialogue + SFX)
**Generation**: 2-digit version number (01, 02, 03, etc.)

## Parameters

```bash
uv run generate_audio.py <script> [options]

Positional:
  script                Script file path (.txt)

Options:
  --gen-count, -c N     Generate N versions per segment (default: 1)
  --output-dir, -o DIR  Output directory (default: generated_audio)
```

## How It Works

1. **Parse Script**: Extracts speakers, dialogue, emotions, and SFX tags
2. **Dialogue Generation**: 
   - Detects or uses explicit emotion
   - Maps emotion to TTS parameters (CFG weight, exaggeration, temperature)
   - Uses Chatterbox model with speaker voice reference
3. **SFX Generation**:
   - Generates audio from text descriptions
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
