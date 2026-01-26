# Hindi Phoneme Recognizer

A wav2vec2-based system for Hindi speech to phoneme recognition with timestamps.

## Features

- Fine-tune wav2vec2 on Common Voice Hindi for phoneme-level recognition
- Transcribe Hindi audio to timestamped phonemes
- Output in SRT subtitle format
- CTC segmentation for accurate alignment

## Installation

```bash
uv sync
```

## Usage

### Fine-tune on Common Voice Hindi

```bash
uv run python main.py --finetune
```

### Transcribe audio to phonemes

```bash
uv run python main.py --input audio.wav
```

### With reference text for better alignment

```bash
uv run python main.py --input audio.wav --text "नमस्ते"
```

### Convert text to phonemes (no audio)

```bash
uv run python main.py --convert "नमस्ते"
```

## Options

```
-f, --finetune          Fine-tune wav2vec2 on Common Voice Hindi
-i, --input FILE        Input audio file to transcribe
-o, --output PATH       Output path (SRT for transcription, dir for training)
-m, --model PATH        Path to trained model (default: outputs/phoneme-model)
--text TEXT             Reference text for CTC segmentation
--epochs N              Training epochs (default: 15)
--batch-size N          Training batch size (default: 8)
--lr RATE               Learning rate (default: 3e-5)
```

## Phoneme Set

The system uses IPA phonemes derived from Hindi Devanagari script. See `w2v2/phoneme.py` for the complete mapping.
