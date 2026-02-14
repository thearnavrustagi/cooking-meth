#!/usr/bin/env python3
"""
Hindi Phoneme Recognizer

A wav2vec2-based system for Hindi speech to phoneme recognition with timestamps.

Usage:
    # Fine-tune on Common Voice Hindi
    uv run python main.py --finetune
    
    # Transcribe audio to timestamped phonemes
    uv run python main.py --input audio.wav
    
    # With custom model path
    uv run python main.py --input audio.wav --model outputs/my-model
"""

import argparse
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


# ---------------------------------------------------------------------------
# Phoneme categories for Hindi IPA – used by --results analysis
# ---------------------------------------------------------------------------
PHONEME_CATEGORIES = {
    "Vowels (short)": ["ə", "i", "u"],
    "Vowels (long)": ["ɑː", "iː", "uː", "eː", "oː"],
    "Diphthongs": ["aːi", "aːu"],
    "Aspirated stops": ["kʰ", "gʰ", "cʰ", "ɟʰ", "ʈʰ", "ɖʰ", "t̪ʰ", "d̪ʰ", "pʰ", "bʰ"],
    "Unaspirated stops": ["k", "g", "c", "ɟ", "ʈ", "ɖ", "t̪", "d̪", "p", "b"],
    "Nasals": ["ŋ", "ɲ", "ɳ", "n", "m", "ⁿ"],
    "Fricatives": ["ɕ", "ʂ", "s", "ɦ", "h", "f", "z", "x", "ɣ"],
    "Approximants/Liquids": ["j", "r", "l", "v", "ɾ", "ɽ", "ɽʱ"],
    "Syllabic": ["ɹ̩", "l̩", "ɹ̩ː", "l̩ː"],
    "Complex": ["kʃ", "t̪ɾ", "gj", "ɕc"],
    "Borrowed": ["q", "x", "ɣ", "z", "f"],
}

# Reverse lookup: phoneme -> category name
PHONEME_TO_CATEGORY = {}
for _cat, _phones in PHONEME_CATEGORIES.items():
    for _p in _phones:
        if _p not in PHONEME_TO_CATEGORY:          # first assignment wins
            PHONEME_TO_CATEGORY[_p] = _cat


def get_device(requested: str = None) -> str:
    """
    Get the best available device for inference.
    
    Args:
        requested: Explicitly requested device ("cuda", "mps", "cpu", or None for auto)
    
    Returns:
        Device string to use
    """
    if requested:
        # User explicitly requested a device
        if requested == "cuda" and not torch.cuda.is_available():
            console.print("[yellow]Warning: CUDA requested but not available, falling back to CPU[/yellow]")
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            console.print("[yellow]Warning: MPS requested but not available, falling back to CPU[/yellow]")
            return "cpu"
        return requested
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        console.print(f"[green]✓[/green] Using CUDA: {device_name}")
        return "cuda"
    elif torch.backends.mps.is_available():
        console.print("[green]✓[/green] Using MPS (Apple Silicon)")
        return "mps"
    else:
        console.print("[dim]Using CPU (no GPU detected)[/dim]")
        return "cpu"


def print_banner():
    """Print a nice banner."""
    banner = Text()
    banner.append("Hindi Phoneme Recognizer\n", style="bold magenta")
    banner.append("wav2vec2-based speech to phoneme recognition", style="dim")
    console.print(Panel(banner, border_style="blue"))


def cmd_finetune(args):
    """Run fine-tuning on Hindi speech dataset."""
    from w2v2.train import train, TrainConfig, DataConfig
    from w2v2.data import DataConfig as DC
    
    console.rule("[bold blue]Fine-tuning Mode")
    
    # Configure training
    train_config = TrainConfig(
        output_dir=args.output or "outputs/phoneme-model-sustained",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        temporal_smoothness=getattr(args, 'temporal_smoothness', 0.2),
        blank_penalty=getattr(args, 'blank_penalty', 0.1),
    )
    
    # Configure dataset
    data_config = DC(
        dataset_type=args.dataset,
        num_proc=args.num_workers,
        max_audio_length=args.max_audio_length,
        max_train_samples=args.max_samples,
        streaming=args.streaming,
    )
    
    # Run training
    train(train_config, data_config, force_reprocess=getattr(args, 'force_reprocess', False))


def cmd_distill(args):
    """Run knowledge distillation to create a 5ms model from a 20ms teacher."""
    from w2v2.train import distill, DistillationConfig
    from w2v2.data import DataConfig as DC
    
    console.rule("[bold blue]Knowledge Distillation Mode (20ms → 5ms)")
    
    # Configure distillation
    distill_config = DistillationConfig(
        teacher_model_path=args.model,
        student_output_dir=args.distill_output or "outputs/phoneme-model-5ms",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.distill_lr,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    
    # Configure dataset
    data_config = DC(
        dataset_type=args.dataset,
        num_proc=args.num_workers,
        max_audio_length=args.max_audio_length,
        max_train_samples=args.max_samples,
        streaming=args.streaming,
    )
    
    # Run distillation
    distill(distill_config, data_config, force_reprocess=getattr(args, 'force_reprocess', False))


def cmd_transcribe(args):
    """Transcribe audio file to phonemes."""
    from w2v2.inference import transcribe_audio
    from lib.logger import error
    
    raw_mode = getattr(args, 'raw_srt', False)
    mode_label = "Raw Transcription Mode" if raw_mode else "Transcription Mode"
    console.rule(f"[bold blue]{mode_label}")
    
    # Validate input
    audio_path = Path(args.input)
    if not audio_path.exists():
        error(f"Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_srt = args.output
    else:
        suffix = ".raw.srt" if raw_mode else ".srt"
        output_srt = str(audio_path.with_suffix(suffix))
    
    # Get device (auto-detect if not specified)
    device = get_device(args.device)
    
    # Run transcription
    extend_timestamps = not getattr(args, 'no_extend', False)
    
    phonemes, timestamps = transcribe_audio(
        audio_path=str(audio_path),
        model_path=args.model,
        output_srt=output_srt,
        reference_text=args.text,
        device=device,
        raw_output=raw_mode,
        extend_timestamps=extend_timestamps and not raw_mode,  # Don't extend in raw mode
    )
    
    console.print(f"\n[green]✓[/green] Output saved to: [cyan]{output_srt}[/cyan]")
    
    if raw_mode:
        console.print(f"[dim]Note: Raw output shows exact model predictions with '_' for blank/pad tokens[/dim]")
    elif extend_timestamps:
        console.print(f"[dim]Note: Phoneme durations extended to fill gaps (use --no-extend for exact model timing)[/dim]")
    
    return phonemes, timestamps


def cmd_convert(args):
    """Convert Devanagari text to phonemes (no audio)."""
    from w2v2.phoneme import text_to_phoneme_list, devanagari_to_ipa
    from lib.logger import print_phonemes
    
    console.rule("[bold blue]Text to Phoneme Conversion")
    
    text = args.convert
    
    # Convert
    ipa = devanagari_to_ipa(text)
    phonemes = text_to_phoneme_list(text)
    
    console.print(f"[bold]Input:[/bold] {text}")
    console.print(f"[bold]IPA:[/bold] {ipa}")
    console.print(f"[bold]Phonemes:[/bold] {' '.join(phonemes)}")
    
    return phonemes


def cmd_read_srt(args):
    """Read an SRT file and display phonemes and Hindi text."""
    from w2v2.phoneme import srt_to_hindi, parse_srt_file
    
    console.rule("[bold blue]SRT to Text Conversion")
    
    srt_path = Path(args.srt)
    if not srt_path.exists():
        console.print(f"[red]Error:[/red] SRT file not found: {srt_path}")
        sys.exit(1)
    
    # Parse and convert
    phoneme_string, hindi_string = srt_to_hindi(str(srt_path))
    
    console.print(f"\n[bold cyan]Phonemes:[/bold cyan]")
    console.print(phoneme_string)
    
    console.print(f"\n[bold cyan]Hindi (approximate):[/bold cyan]")
    console.print(hindi_string)
    
    # Also show some stats
    entries = parse_srt_file(str(srt_path))
    console.print(f"\n[dim]Total phoneme entries: {len(entries)}[/dim]")
    
    return phoneme_string, hindi_string


def cmd_transcribe_base(args):
    """Transcribe audio using the base wav2vec2-xlsr Hindi model."""
    from w2v2.inference import transcribe_with_base_model
    
    console.rule("[bold blue]Hindi Speech-to-Text (Base Model)")
    
    # Validate input
    audio_path = Path(args.transcribe)
    if not audio_path.exists():
        console.print(f"[red]Error:[/red] Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Get device (auto-detect if not specified)
    device = get_device(args.device)
    
    # Run transcription
    transcription = transcribe_with_base_model(
        audio_path=str(audio_path),
        model_name=args.model if args.model and not os.path.isdir(args.model) else args.base_model,
        device=device,
    )
    
    console.print(f"\n[bold cyan]Transcription:[/bold cyan]")
    console.print(f"[green]{transcription}[/green]")
    
    # Optionally save to file
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(transcription)
        console.print(f"\n[dim]Saved to: {args.output}[/dim]")
    
    return transcription


def cmd_hindi_srt(args):
    """Transcribe audio to time-aligned Hindi text SRT file."""
    from w2v2.inference import transcribe_hindi_to_srt
    
    console.rule("[bold blue]Hindi Speech-to-Text SRT Generation")
    
    # Validate input
    audio_path = Path(args.hindi_srt)
    if not audio_path.exists():
        console.print(f"[red]Error:[/red] Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_srt = args.output
    else:
        output_srt = str(audio_path.with_suffix(".srt"))
    
    # Get device (auto-detect if not specified)
    device = get_device(args.device)
    
    # Run transcription
    items, timestamps = transcribe_hindi_to_srt(
        audio_path=str(audio_path),
        output_srt=output_srt,
        model_name=args.model if args.model and not os.path.isdir(args.model) else args.base_model,
        device=device,
        group_words=args.group_words,
        char_level=args.char_level,
    )
    
    console.print(f"\n[green]✓[/green] Hindi SRT saved to: [cyan]{output_srt}[/cyan]")
    unit = "characters" if args.char_level else "words"
    console.print(f"[dim]Total {unit}: {len(items)}[/dim]")
    
    return items, timestamps


def get_latest_checkpoint(base_dir: str = "outputs") -> str:
    """Find the most recent model checkpoint in the given directory."""
    if not os.path.exists(base_dir):
        return "outputs/phoneme-model"  # Fallback
        
    # Check if base_dir itself is a model (has config.json)
    if os.path.exists(os.path.join(base_dir, "config.json")):
        return base_dir

    # Search for directories containing config.json
    model_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files:
            model_dirs.append(root)
            
    if not model_dirs:
        return "outputs/phoneme-model"  # Fallback
        
    # Return the one with the latest modification time
    return max(model_dirs, key=lambda d: os.path.getmtime(d))


def cmd_evaluate_base(args):
    """Evaluate base wav2vec2-xlsr Hindi model on IndicVoices validation set."""
    from w2v2.inference import BaseModelTranscriber
    from w2v2.data import authenticate_huggingface, load_hf_token
    from datasets import load_dataset, Audio
    from tqdm import tqdm
    import numpy as np
    from collections import Counter
    import evaluate
    from lib.logger import info, print_results
    
    console.rule("[bold blue]Base Model Evaluation (IndicVoices Validation Set)")
    
    # Get device
    device = get_device(args.device)
    
    # Setup transcriber
    model_name = args.model if args.model and not os.path.isdir(args.model) else args.base_model
    console.print(f"[dim]ℹ Loading base model: {model_name}...[/dim]")
    transcriber = BaseModelTranscriber(model_name=model_name, device=device)
    console.print("[green]✓[/green] Base model loaded")
    
    # Load dataset directly to get original Hindi text
    info("Loading IndicVoices validation split...")
    hf_token = load_hf_token()
    if hf_token:
        authenticate_huggingface(hf_token)
    
    dataset = load_dataset(
        "/home/interspeech/indic_voices_data",
        "default",
        split="validation",
        token=hf_token,
        trust_remote_code=True,
    )
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    
    # Filter by duration if needed
    max_dur = args.max_audio_length
    dataset = dataset.filter(lambda x: x["duration"] is not None and x["duration"] <= max_dur)
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        
    console.print(f"[dim]ℹ Evaluating {len(dataset)} samples...[/dim]")
    
    # Metrics tracking
    all_preds = []
    all_labels = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    pbar = tqdm(range(len(dataset)), desc="Evaluating")
    for i in pbar:
        sample = dataset[i]
        audio_array = sample["audio_filepath"]["array"]
        label_text = sample["normalized"]
        
        # Transcribe
        with torch.no_grad():
            inputs = transcriber.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(device)
            
            outputs = transcriber.model(inputs.input_values)
            pred_ids = torch.argmax(outputs.logits, dim=-1)
            pred_text = transcriber.processor.batch_decode(pred_ids)[0]
        
        all_preds.append(pred_text)
        all_labels.append(label_text)
        
        # Bag-of-characters F1 (since base model is character-based for Hindi)
        p_counts = Counter(pred_text)
        l_counts = Counter(label_text)
        
        tp = sum((p_counts & l_counts).values())
        fp = sum((p_counts - l_counts).values())
        fn = sum((l_counts - p_counts).values())
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if (i + 1) % 10 == 0:
            current_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            current_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            current_f1 = 2 * current_prec * current_rec / (current_prec + current_rec) if (current_prec + current_rec) > 0 else 0
            pbar.set_postfix({"F1": f"{current_f1:.2%}"})
            
    # Final metrics
    wer = wer_metric.compute(predictions=all_preds, references=all_labels)
    cer = cer_metric.compute(predictions=all_preds, references=all_labels)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print_results({
        "WER": wer,
        "CER": cer,
        "Accuracy (1-CER)": 1.0 - cer,
        "F1 Score (Bag-of-Chars)": f1,
        "Precision": precision,
        "Recall": recall,
    }, title="Base Model Evaluation Results")
    
    return {"wer": wer, "cer": cer, "f1": f1}

def cmd_evaluate(args):
    """Evaluate model on AI4Bharat test set."""
    from w2v2.inference import evaluate_model
    from w2v2.data import DataConfig as DC
    
    console.rule("[bold blue]Evaluation Mode")
    
    # Configure dataset
    data_config = DC(
        dataset_type="indicvoices",
        num_proc=args.num_workers,
        max_audio_length=args.max_audio_length,
        streaming=False,  # Use full dataset (data now on external SSD)
    )
    
    # Get device
    device = get_device(args.device)
    
    # Run evaluation
    metrics = evaluate_model(
        model_path=args.model,
        data_config=data_config,
        device=device,
        batch_size=args.batch_size,
        force_reprocess=getattr(args, 'force_reprocess', False),
        val_only=getattr(args, 'val_only', False),
    )
    
    return metrics


def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-1):
    """
    Needleman-Wunsch global sequence alignment algorithm.
    Returns aligned sequences with gaps represented as None.
    """
    n, m = len(seq1), len(seq2)
    
    # Initialize score matrix
    score = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Initialize first row and column with gap penalties
    for i in range(n + 1):
        score[i][0] = i * gap
    for j in range(m + 1):
        score[0][j] = j * gap
    
    # Fill the score matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = score[i - 1][j - 1] + match
            else:
                diag = score[i - 1][j - 1] + mismatch
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap
            score[i][j] = max(diag, up, left)
    
    # Traceback to get alignment
    aligned1, aligned2 = [], []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq1[i - 1] == seq2[j - 1]:
                diag_score = score[i - 1][j - 1] + match
            else:
                diag_score = score[i - 1][j - 1] + mismatch
        else:
            diag_score = float('-inf')
        
        up_score = score[i - 1][j] + gap if i > 0 else float('-inf')
        left_score = score[i][j - 1] + gap if j > 0 else float('-inf')
        
        if i > 0 and j > 0 and score[i][j] == diag_score:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and score[i][j] == up_score:
            aligned1.append(seq1[i - 1])
            aligned2.append(None)  # Gap in seq2
            i -= 1
        else:
            aligned1.append(None)  # Gap in seq1
            aligned2.append(seq2[j - 1])
            j -= 1
    
    # Reverse since we traced back
    aligned1.reverse()
    aligned2.reverse()
    
    return aligned1, aligned2


def evaluate_single_sample(recognizer, sample, np):
    """Evaluate a single sample and return metrics and tokens."""
    from collections import Counter
    
    input_values = sample.get("input_values")
    label_ids = sample.get("labels")
    
    if input_values is None or label_ids is None:
        return None
    
    # Convert to tensor
    if not isinstance(input_values, torch.Tensor):
        input_values = torch.tensor(input_values)
    
    # Handle label_ids format
    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.squeeze().tolist()
    elif isinstance(label_ids, list) and len(label_ids) > 0 and isinstance(label_ids[0], torch.Tensor):
        label_ids = [t.item() if t.numel() == 1 else t.tolist() for t in label_ids]
    while isinstance(label_ids, list) and len(label_ids) > 0 and isinstance(label_ids[0], (list, np.ndarray)):
        label_ids = label_ids[0]
    if isinstance(label_ids, np.ndarray):
        label_ids = label_ids.flatten().tolist()
    
    # Get prediction
    with torch.no_grad():
        logits = recognizer.get_logits(input_values)
        pred_ids = np.argmax(logits, axis=-1)
        
        # Decode for display (concatenated)
        pred_str = recognizer.processor.batch_decode([pred_ids])[0]
        label_str = recognizer.processor.decode(label_ids, group_tokens=False)
        
        # Get individual tokens for label
        label_tokens = []
        for lid in label_ids:
            if lid != -100:
                token = recognizer.processor.tokenizer.convert_ids_to_tokens(lid)
                if token:
                    label_tokens.append(token)
        
        # Get individual tokens for prediction (after CTC collapse)
        pred_ids_list = pred_ids.tolist() if hasattr(pred_ids, 'tolist') else list(pred_ids)
        pred_tokens = []
        prev_id = None
        for pid in pred_ids_list:
            if pid != prev_id and pid != recognizer.processor.tokenizer.pad_token_id:
                token = recognizer.processor.tokenizer.convert_ids_to_tokens(pid)
                if token and token != '<pad>':
                    pred_tokens.append(token)
            prev_id = pid
    
    # Needleman-Wunsch alignment
    aligned_label, aligned_pred = needleman_wunsch(label_tokens, pred_tokens)
    
    # Calculate alignment statistics
    matches = substitutions = insertions = deletions = 0
    for lab, pred in zip(aligned_label, aligned_pred):
        if lab is None:
            insertions += 1
        elif pred is None:
            deletions += 1
        elif lab == pred:
            matches += 1
        else:
            substitutions += 1
    
    # Calculate F1 (bag of phonemes)
    pred_counts = Counter(pred_tokens)
    label_counts = Counter(label_tokens)
    tp = sum((pred_counts & label_counts).values())
    fp = sum((pred_counts - label_counts).values())
    fn = sum((label_counts - pred_counts).values())
    
    return {
        "input_values": input_values,
        "pred_str": pred_str,
        "label_str": label_str,
        "pred_tokens": pred_tokens,
        "label_tokens": label_tokens,
        "aligned_label": aligned_label,
        "aligned_pred": aligned_pred,
        "matches": matches,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def cmd_sample(args):
    """Evaluate sample(s) and show predicted vs actual phonemes."""
    from w2v2.inference import PhonemeRecognizer, InferenceConfig
    from w2v2.data import DataConfig as DC, prepare_dataset
    from tqdm import tqdm
    import numpy as np
    import evaluate
    
    dataset_name = "FLEURS" if getattr(args, "fleurs", False) else "IndicVoices"
    dataset_type = "fleurs" if getattr(args, "fleurs", False) else "indicvoices"
    
    # Check if evaluating entire validation set
    eval_val = args.sample.lower() == "val"
    
    if eval_val:
        console.rule(f"[bold blue]Validation Set Evaluation — {dataset_name} (Alignment-based)")
    else:
        console.rule(f"[bold blue]Single Sample Evaluation — {dataset_name}")
        try:
            sample_idx = int(args.sample)
        except ValueError:
            console.print(f"[red]Invalid sample index: {args.sample}. Use an integer or 'val'.[/red]")
            return
    
    # Configure dataset
    data_config = DC(
        dataset_type=dataset_type,
        num_proc=args.num_workers,
        max_audio_length=args.max_audio_length,
        streaming=False,
    )
    
    # Get device
    device = get_device(args.device)
    
    # Setup recognizer
    console.print(f"[dim]ℹ Loading model from {args.model}...[/dim]")
    config = InferenceConfig(model_path=args.model, device=device)
    recognizer = PhonemeRecognizer(config)
    console.print("[green]✓[/green] Model loaded")
    
    # Load dataset
    dataset = prepare_dataset(data_config, recognizer.processor, recognizer.vocab, force_reprocess=False, val_only=True)
    
    split = dataset.get("validation") or dataset.get("train")
    if not split:
        console.print("[red]No dataset split found![/red]")
        return
    
    if eval_val:
        # Evaluate entire validation set
        console.print(f"[dim]ℹ Evaluating {len(split)} validation samples...[/dim]")
        
        # Aggregate statistics
        total_matches = 0
        total_substitutions = 0
        total_insertions = 0
        total_deletions = 0
        total_ref_tokens = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        all_pred_spaced = []
        all_label_spaced = []
        skipped = 0
        
        for i in tqdm(range(len(split)), desc="Evaluating"):
            sample = split[i]
            result = evaluate_single_sample(recognizer, sample, np)
            
            if result is None:
                skipped += 1
                continue
            
            total_matches += result["matches"]
            total_substitutions += result["substitutions"]
            total_insertions += result["insertions"]
            total_deletions += result["deletions"]
            total_ref_tokens += len(result["label_tokens"])
            total_tp += result["tp"]
            total_fp += result["fp"]
            total_fn += result["fn"]
            
            all_pred_spaced.append(" ".join(result["pred_tokens"]))
            all_label_spaced.append(" ".join(result["label_tokens"]))
        
        # Calculate aggregate metrics
        wer_metric = evaluate.load("wer")
        per = wer_metric.compute(predictions=all_pred_spaced, references=all_label_spaced)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Display results
        console.print()
        console.print(Panel(
            f"[bold]Samples Evaluated:[/bold] {len(split) - skipped}\n"
            f"[bold]Samples Skipped:[/bold] {skipped}\n"
            f"[bold]Total Reference Tokens:[/bold] {total_ref_tokens}",
            title="Validation Set Summary",
            border_style="blue"
        ))
        
        console.print()
        accuracy = (1 - per) * 100
        per_pct = per * 100
        
        def get_style(val, thresholds=(0.7, 0.4)):
            if val >= thresholds[0]:
                return "green"
            elif val >= thresholds[1]:
                return "yellow"
            return "red"
        
        per_style = get_style(1 - per)
        f1_style = get_style(f1)
        prec_style = get_style(precision)
        rec_style = get_style(recall)
        
        console.print(f"[bold]Phoneme Error Rate (PER):[/bold] [{per_style}]{per_pct:.1f}%[/{per_style}]")
        console.print(f"[bold]Accuracy:[/bold] [{per_style}]{accuracy:.1f}%[/{per_style}]")
        console.print(f"[bold]F1 Score:[/bold] [{f1_style}]{f1*100:.1f}%[/{f1_style}]")
        console.print(f"[bold]Precision:[/bold] [{prec_style}]{precision*100:.1f}%[/{prec_style}]  [bold]Recall:[/bold] [{rec_style}]{recall*100:.1f}%[/{rec_style}]")
        
        # Alignment statistics
        console.print()
        console.print("[bold]Alignment Statistics (Needleman-Wunsch):[/bold]")
        match_rate = total_matches / total_ref_tokens * 100 if total_ref_tokens > 0 else 0
        console.print(f"  Matches: {total_matches:,} ({match_rate:.1f}%)")
        console.print(f"  Substitutions: {total_substitutions:,}")
        console.print(f"  Insertions: {total_insertions:,}")
        console.print(f"  Deletions: {total_deletions:,}")
        
    else:
        # Single sample evaluation
        if sample_idx >= len(split):
            console.print(f"[red]Sample index {sample_idx} out of range (dataset has {len(split)} samples)[/red]")
            return
        
        console.print(f"[dim]ℹ Evaluating sample {sample_idx} of {len(split)}...[/dim]")
        
        sample = split[sample_idx]
        result = evaluate_single_sample(recognizer, sample, np)
        
        if result is None:
            console.print("[red]Invalid sample (missing input_values or labels)[/red]")
            return
        
        # Calculate PER for this sample
        wer_metric = evaluate.load("wer")
        pred_str_spaced = " ".join(result["pred_tokens"])
        label_str_spaced = " ".join(result["label_tokens"])
        per = wer_metric.compute(predictions=[pred_str_spaced], references=[label_str_spaced])
        
        # Calculate F1
        tp, fp, fn = result["tp"], result["fp"], result["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Display results
        console.print()
        console.print(Panel(
            f"[bold]Sample Index:[/bold] {sample_idx}\n"
            f"[bold]Audio Length:[/bold] {len(result['input_values']) / 16000:.2f}s",
            title="Sample Info",
            border_style="blue"
        ))
        
        console.print()
        console.print(Panel(
            result["label_str"],
            title="[green]Actual Phonemes (Ground Truth)[/green]",
            border_style="green"
        ))
        
        console.print()
        console.print(Panel(
            result["pred_str"],
            title="[yellow]Predicted Phonemes[/yellow]",
            border_style="yellow"
        ))
        
        # Show metrics
        console.print()
        accuracy = (1 - per) * 100
        per_pct = per * 100
        
        def get_style(val, thresholds=(0.7, 0.4)):
            if val >= thresholds[0]:
                return "green"
            elif val >= thresholds[1]:
                return "yellow"
            return "red"
        
        per_style = get_style(1 - per)
        f1_style = get_style(f1)
        prec_style = get_style(precision)
        rec_style = get_style(recall)
        
        console.print(f"[bold]Phoneme Error Rate (PER):[/bold] [{per_style}]{per_pct:.1f}%[/{per_style}]")
        console.print(f"[bold]Accuracy:[/bold] [{per_style}]{accuracy:.1f}%[/{per_style}]")
        console.print(f"[bold]F1 Score:[/bold] [{f1_style}]{f1*100:.1f}%[/{f1_style}]")
        console.print(f"[bold]Precision:[/bold] [{prec_style}]{precision*100:.1f}%[/{prec_style}]  [bold]Recall:[/bold] [{rec_style}]{recall*100:.1f}%[/{rec_style}]")
        
        # Show alignment
        aligned_label = result["aligned_label"]
        aligned_pred = result["aligned_pred"]
        matches = result["matches"]
        substitutions = result["substitutions"]
        insertions = result["insertions"]
        deletions = result["deletions"]
        
        diff_lines = []
        for lab, pred in zip(aligned_label, aligned_pred):
            if lab is None:
                diff_lines.append(f"  [blue]+[/blue] [dim]---[/dim] → {pred} [blue](insertion)[/blue]")
            elif pred is None:
                diff_lines.append(f"  [magenta]-[/magenta] {lab} → [dim]---[/dim] [magenta](deletion)[/magenta]")
            elif lab == pred:
                diff_lines.append(f"  [green]✓[/green] {lab}")
            else:
                diff_lines.append(f"  [red]✗[/red] {lab} → {pred} [red](substitution)[/red]")
        
        console.print()
        console.print("[bold]Alignment (Needleman-Wunsch):[/bold]")
        total_ref = len(result["label_tokens"])
        console.print(f"[dim]Matches: {matches}/{total_ref} | Substitutions: {substitutions} | Insertions: {insertions} | Deletions: {deletions}[/dim]")
        console.print()
        
        if len(diff_lines) > 50:
            for line in diff_lines[:50]:
                console.print(line)
            console.print(f"  [dim]... and {len(diff_lines) - 50} more positions[/dim]")
        else:
            for line in diff_lines:
                console.print(line)


# ---------------------------------------------------------------------------
# --results  in-depth analysis
# ---------------------------------------------------------------------------

def cmd_results(args):
    """In-depth results analysis on the validation set."""
    from w2v2.inference import PhonemeRecognizer, InferenceConfig
    from w2v2.data import DataConfig as DC, prepare_dataset
    from collections import Counter, defaultdict
    from tqdm import tqdm
    import numpy as np
    import evaluate
    import json
    import time

    dataset_name = "FLEURS" if getattr(args, "fleurs", False) else "IndicVoices"
    dataset_type = "fleurs" if getattr(args, "fleurs", False) else "indicvoices"

    console.rule(f"[bold blue]Results Analysis — {dataset_name} Validation Set")

    # ---- setup ----------------------------------------------------------
    device = get_device(args.device)
    data_config = DC(
        dataset_type=dataset_type,
        num_proc=args.num_workers,
        max_audio_length=args.max_audio_length,
        streaming=False,
    )

    console.print(f"[dim]Loading model from {args.model}...[/dim]")
    config = InferenceConfig(model_path=args.model, device=device)
    recognizer = PhonemeRecognizer(config)
    console.print("[green]✓[/green] Model loaded")

    # ====================================================================
    # SECTION 0 - Experimental Setup
    # ====================================================================
    from rich.table import Table

    _model = recognizer.model
    _mcfg = _model.config

    total_params = sum(p.numel() for p in _model.parameters())
    trainable_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    base_model_name = getattr(_mcfg, "_name_or_path", "unknown")
    vocab_size = getattr(_mcfg, "vocab_size", "?")
    hidden_size = getattr(_mcfg, "hidden_size", "?")
    num_layers = getattr(_mcfg, "num_hidden_layers", "?")
    num_heads = getattr(_mcfg, "num_attention_heads", "?")

    conv_stride = getattr(_mcfg, "conv_stride", [])
    total_stride_val = 1
    for _s in conv_stride:
        total_stride_val *= _s
    frame_ms = total_stride_val / 16.0

    device_label = device
    if "cuda" in device and torch.cuda.is_available():
        device_label = f"{torch.cuda.get_device_name(0)} (CUDA)"

    console.print()
    console.rule("[bold cyan]Experimental Setup")

    tbl_setup = Table(title="Model & Infrastructure", show_header=True, header_style="bold green")
    tbl_setup.add_column("Property", style="cyan")
    tbl_setup.add_column("Value")

    tbl_setup.add_row("Base model", str(base_model_name))
    tbl_setup.add_row("Model path", str(args.model))
    tbl_setup.add_row("Total parameters", f"{total_params:,}")
    tbl_setup.add_row("Trainable parameters", f"{trainable_params:,}")
    tbl_setup.add_row("Frozen parameters", f"{frozen_params:,}")
    tbl_setup.add_row("Hidden size", str(hidden_size))
    tbl_setup.add_row("Transformer layers", str(num_layers))
    tbl_setup.add_row("Attention heads", str(num_heads))
    tbl_setup.add_row("Vocab size (phonemes)", str(vocab_size))
    tbl_setup.add_row("Conv stride", str(conv_stride))
    tbl_setup.add_row("Frame duration", f"{frame_ms:.1f} ms")
    tbl_setup.add_row("Device", device_label)
    tbl_setup.add_row("Dataset", dataset_name)
    console.print(tbl_setup)

    console.print()
    console.print("[bold]Evaluation metrics:[/bold]")
    console.print("  PER  - Phoneme Error Rate (edit distance on phoneme token sequences)")
    console.print("  CER  - Character Error Rate (edit distance on raw IPA character strings)")
    console.print("  F1   - Bag-of-phonemes F1 (token-level precision/recall)")
    console.print("  TPR  - True Positive Rate per phoneme (correct / total occurrences)")
    console.print("  NW   - Needleman-Wunsch global alignment for substitution/insertion/deletion counts")

    dataset = prepare_dataset(
        data_config, recognizer.processor, recognizer.vocab,
        force_reprocess=False, val_only=True,
    )
    split = dataset.get("validation") or dataset.get("train")
    if not split:
        console.print("[red]No validation split found![/red]")
        return

    num_samples = len(split)
    console.print(f"\n[dim]Evaluating {num_samples} samples...[/dim]")

    eval_start_time = time.time()

    # ---- per-sample collection ------------------------------------------
    # Aggregate counters
    total_matches = 0
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    total_ref_tokens = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_pred_spaced = []
    all_label_spaced = []
    all_pred_joined = []      # raw joined strings for CER
    all_label_joined = []
    per_sample_pers = []      # per-sample PER for confidence intervals
    per_sample_f1s = []       # per-sample F1 for confidence intervals
    per_sample_precisions = []
    per_sample_recalls = []
    # Per-sample per-phoneme tracking: list of (ref_count_dict, correct_count_dict)
    per_sample_phoneme_data = []
    skipped = 0

    # Per-phoneme counters
    phoneme_ref_count = Counter()        # how many times each phoneme appears in ground truth
    phoneme_correct = Counter()          # correct matches per phoneme
    phoneme_substituted_as = defaultdict(Counter)  # ref -> pred confusion
    phoneme_deleted = Counter()          # phoneme in ref, gap in pred
    phoneme_inserted = Counter()         # phoneme in pred, gap in ref
    phoneme_pred_count = Counter()       # total times predicted

    # Insertion context tracking (what surrounds an insertion)
    insertion_context = []  # list of (prev_ref, inserted, next_ref)

    for i in tqdm(range(num_samples), desc="Evaluating"):
        sample = split[i]
        result = evaluate_single_sample(recognizer, sample, np)
        if result is None:
            skipped += 1
            continue

        total_matches += result["matches"]
        total_substitutions += result["substitutions"]
        total_insertions += result["insertions"]
        total_deletions += result["deletions"]
        total_ref_tokens += len(result["label_tokens"])
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        all_pred_spaced.append(" ".join(result["pred_tokens"]))
        all_label_spaced.append(" ".join(result["label_tokens"]))
        all_pred_joined.append("".join(result["pred_tokens"]))
        all_label_joined.append("".join(result["label_tokens"]))

        # Per-sample PER (for confidence intervals)
        n_ref = len(result["label_tokens"])
        if n_ref > 0:
            sample_errors = result["substitutions"] + result["insertions"] + result["deletions"]
            per_sample_pers.append(sample_errors / n_ref)

        # Per-sample F1 / precision / recall
        s_tp, s_fp, s_fn = result["tp"], result["fp"], result["fn"]
        s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        s_rec = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
        s_f1 = 2 * s_prec * s_rec / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0.0
        per_sample_f1s.append(s_f1)
        per_sample_precisions.append(s_prec)
        per_sample_recalls.append(s_rec)

        # Per-sample per-phoneme ref/correct for bootstrap CIs on TPR
        sample_ph_ref = Counter(result["label_tokens"])
        sample_ph_correct = Counter()
        for lab, pred in zip(result["aligned_label"], result["aligned_pred"]):
            if lab is not None and pred is not None and lab == pred:
                sample_ph_correct[lab] += 1
        per_sample_phoneme_data.append((dict(sample_ph_ref), dict(sample_ph_correct)))

        # Count reference phonemes
        for tok in result["label_tokens"]:
            phoneme_ref_count[tok] += 1
        for tok in result["pred_tokens"]:
            phoneme_pred_count[tok] += 1

        # Walk alignment
        aligned_label = result["aligned_label"]
        aligned_pred = result["aligned_pred"]
        for idx, (lab, pred) in enumerate(zip(aligned_label, aligned_pred)):
            if lab is None:
                # insertion
                phoneme_inserted[pred] += 1
                prev_ref = None
                next_ref = None
                # find surrounding ref phonemes
                for back in range(idx - 1, -1, -1):
                    if aligned_label[back] is not None:
                        prev_ref = aligned_label[back]
                        break
                for fwd in range(idx + 1, len(aligned_label)):
                    if aligned_label[fwd] is not None:
                        next_ref = aligned_label[fwd]
                        break
                insertion_context.append((prev_ref, pred, next_ref))
            elif pred is None:
                # deletion
                phoneme_deleted[lab] += 1
            elif lab == pred:
                phoneme_correct[lab] += 1
            else:
                phoneme_substituted_as[lab][pred] += 1

    eval_elapsed = time.time() - eval_start_time
    avg_time_per_sample = eval_elapsed / max(num_samples - skipped, 1)

    # ====================================================================
    # SECTION 1 – Overall Transcription Quality
    # ====================================================================
    console.print()
    console.rule("[bold cyan]Section 1: Transcription Quality")

    wer_metric = evaluate.load("wer")
    per = wer_metric.compute(predictions=all_pred_spaced, references=all_label_spaced)

    # CER – treat every IPA character as a token
    cer_metric = evaluate.load("cer")
    cer = cer_metric.compute(predictions=all_pred_joined, references=all_label_joined)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    console.print()
    console.print(Panel(
        f"[bold]Dataset:[/bold] {dataset_name}\n"
        f"[bold]Samples Evaluated:[/bold] {num_samples - skipped}\n"
        f"[bold]Samples Skipped:[/bold] {skipped}\n"
        f"[bold]Total Reference Tokens:[/bold] {total_ref_tokens:,}",
        title="Summary",
        border_style="blue",
    ))

    def _style(val, high_good=True, thresholds=(0.7, 0.4)):
        """Return Rich colour name."""
        v = val if high_good else 1 - val
        if v >= thresholds[0]:
            return "green"
        elif v >= thresholds[1]:
            return "yellow"
        return "red"

    tbl = Table(title="Overall Metrics", show_header=True, header_style="bold green")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right")

    per_pct = per * 100
    cer_pct = cer * 100
    acc_pct = (1 - per) * 100

    tbl.add_row("Phoneme Error Rate (PER)", f"[{_style(per, high_good=False)}]{per_pct:.2f}%[/]")
    tbl.add_row("Accuracy (1-PER)", f"[{_style(1-per)}]{acc_pct:.2f}%[/]")
    tbl.add_row("Character Error Rate (CER)", f"[{_style(cer, high_good=False)}]{cer_pct:.2f}%[/]")
    tbl.add_row("F1 (bag-of-phonemes)", f"[{_style(f1)}]{f1*100:.2f}%[/]")
    tbl.add_row("Precision", f"[{_style(precision)}]{precision*100:.2f}%[/]")
    tbl.add_row("Recall", f"[{_style(recall)}]{recall*100:.2f}%[/]")
    console.print(tbl)

    # Alignment breakdown
    console.print()
    total_ops = total_matches + total_substitutions + total_insertions + total_deletions
    def _pct(n):
        return f"{n/total_ops*100:.1f}%" if total_ops else "0%"

    tbl2 = Table(title="Alignment Breakdown (Needleman-Wunsch)", show_header=True, header_style="bold green")
    tbl2.add_column("Operation", style="cyan")
    tbl2.add_column("Count", justify="right")
    tbl2.add_column("% of total", justify="right")
    tbl2.add_row("Matches", f"{total_matches:,}", _pct(total_matches))
    tbl2.add_row("Substitutions", f"{total_substitutions:,}", _pct(total_substitutions))
    tbl2.add_row("Insertions", f"{total_insertions:,}", _pct(total_insertions))
    tbl2.add_row("Deletions", f"{total_deletions:,}", _pct(total_deletions))
    console.print(tbl2)

    # ---- Runtime --------------------------------------------------------
    console.print()
    tbl_rt = Table(title="Runtime", show_header=True, header_style="bold green")
    tbl_rt.add_column("Metric", style="cyan")
    tbl_rt.add_column("Value", justify="right")
    mins, secs = divmod(eval_elapsed, 60)
    tbl_rt.add_row("Total inference time", f"{int(mins)}m {secs:.1f}s")
    tbl_rt.add_row("Average per sample", f"{avg_time_per_sample*1000:.0f} ms")
    tbl_rt.add_row("Samples / second", f"{1/avg_time_per_sample:.1f}" if avg_time_per_sample > 0 else "N/A")
    console.print(tbl_rt)

    # ---- 95% Confidence Interval on PER (bootstrap) ----------------------
    console.print()
    per_arr = np.array(per_sample_pers)
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(per_arr, size=len(per_arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    per_mean = per_arr.mean()
    per_std = per_arr.std(ddof=1) if len(per_arr) > 1 else 0.0
    per_se = per_std / np.sqrt(len(per_arr)) if len(per_arr) > 0 else 0.0

    tbl_ci = Table(title="Statistical Significance (PER)", show_header=True, header_style="bold green")
    tbl_ci.add_column("Statistic", style="cyan")
    tbl_ci.add_column("Value", justify="right")
    tbl_ci.add_row("Mean per-sample PER", f"{per_mean*100:.2f}%")
    tbl_ci.add_row("Std dev (per-sample)", f"{per_std*100:.2f}%")
    tbl_ci.add_row("Std error of the mean", f"{per_se*100:.3f}%")
    tbl_ci.add_row("95% CI (bootstrap, n=1000)", f"[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]")
    tbl_ci.add_row("Num samples in CI", str(len(per_arr)))
    console.print(tbl_ci)

    # ====================================================================
    # SECTION 2 – Per-phoneme Error Analysis
    # ====================================================================
    console.print()
    console.rule("[bold cyan]Section 2: Per-Phoneme Error Analysis")

    # ---- 2a: Substitutions & Deletions ---------------------------------
    console.print()
    console.print("[bold]2a. Substitutions & Deletions (sorted by error rate)[/bold]")

    phoneme_stats = []
    for ph in sorted(phoneme_ref_count, key=lambda p: phoneme_ref_count[p], reverse=True):
        ref_n = phoneme_ref_count[ph]
        correct_n = phoneme_correct.get(ph, 0)
        sub_n = sum(phoneme_substituted_as.get(ph, {}).values())
        del_n = phoneme_deleted.get(ph, 0)
        tpr = correct_n / ref_n if ref_n else 0
        err_rate = (sub_n + del_n) / ref_n if ref_n else 0
        # Top confusion targets
        confusions = phoneme_substituted_as.get(ph, Counter()).most_common(3)
        conf_str = ", ".join(f"{t}({c})" for t, c in confusions) if confusions else "-"
        phoneme_stats.append((ph, ref_n, correct_n, sub_n, del_n, tpr, err_rate, conf_str))

    # Sort by error rate descending
    phoneme_stats.sort(key=lambda x: x[6], reverse=True)

    tbl3 = Table(title="Per-Phoneme Error Rates", show_header=True, header_style="bold green")
    tbl3.add_column("Phoneme", style="cyan")
    tbl3.add_column("Ref#", justify="right")
    tbl3.add_column("Correct", justify="right")
    tbl3.add_column("Subs", justify="right")
    tbl3.add_column("Dels", justify="right")
    tbl3.add_column("TPR", justify="right")
    tbl3.add_column("Err%", justify="right")
    tbl3.add_column("Top Confusions")

    for ph, ref_n, correct_n, sub_n, del_n, tpr, err_rate, conf_str in phoneme_stats:
        tpr_s = _style(tpr)
        err_s = _style(err_rate, high_good=False)
        tbl3.add_row(
            ph, str(ref_n), str(correct_n), str(sub_n), str(del_n),
            f"[{tpr_s}]{tpr*100:.1f}%[/]",
            f"[{err_s}]{err_rate*100:.1f}%[/]",
            conf_str,
        )
    console.print(tbl3)

    # ---- 2b: Insertions ------------------------------------------------
    console.print()
    console.print("[bold]2b. Insertions (phonemes predicted but not in ground truth)[/bold]")

    insertion_stats = []
    for ph in sorted(phoneme_inserted, key=lambda p: phoneme_inserted[p], reverse=True):
        ins_n = phoneme_inserted[ph]
        pred_n = phoneme_pred_count.get(ph, 0)
        correct_n = phoneme_correct.get(ph, 0)
        ins_precision = correct_n / pred_n if pred_n else 0
        insertion_stats.append((ph, ins_n, pred_n, ins_precision))

    tbl4 = Table(title="Insertion Analysis", show_header=True, header_style="bold green")
    tbl4.add_column("Phoneme", style="cyan")
    tbl4.add_column("Insertions", justify="right")
    tbl4.add_column("Total Predicted", justify="right")
    tbl4.add_column("Precision", justify="right")

    for ph, ins_n, pred_n, ins_prec in insertion_stats[:30]:
        prec_s = _style(ins_prec)
        tbl4.add_row(ph, str(ins_n), str(pred_n), f"[{prec_s}]{ins_prec*100:.1f}%[/]")
    console.print(tbl4)

    # Insertion context summary
    console.print()
    console.print("[bold]Insertion Context (what surrounds inserted phonemes)[/bold]")
    context_counter = Counter()
    for prev_ref, inserted, next_ref in insertion_context:
        prev_cat = PHONEME_TO_CATEGORY.get(prev_ref, "?") if prev_ref else "START"
        next_cat = PHONEME_TO_CATEGORY.get(next_ref, "?") if next_ref else "END"
        context_counter[(prev_cat, next_cat)] += 1

    tbl_ctx = Table(show_header=True, header_style="bold green")
    tbl_ctx.add_column("Preceding Category", style="cyan")
    tbl_ctx.add_column("Following Category", style="cyan")
    tbl_ctx.add_column("Count", justify="right")

    for (prev_cat, next_cat), cnt in context_counter.most_common(15):
        tbl_ctx.add_row(prev_cat, next_cat, str(cnt))
    console.print(tbl_ctx)

    # ====================================================================
    # SECTION 3 – Category-level Breakdown
    # ====================================================================
    console.print()
    console.rule("[bold cyan]Section 3: Phoneme Category Breakdown")

    # Per-category aggregates
    cat_stats = {}
    for cat, phones in PHONEME_CATEGORIES.items():
        ref_total = sum(phoneme_ref_count.get(p, 0) for p in phones)
        correct_total = sum(phoneme_correct.get(p, 0) for p in phones)
        sub_total = sum(sum(phoneme_substituted_as.get(p, {}).values()) for p in phones)
        del_total = sum(phoneme_deleted.get(p, 0) for p in phones)
        ins_total = sum(phoneme_inserted.get(p, 0) for p in phones)
        tpr = correct_total / ref_total if ref_total else 0
        sub_rate = sub_total / ref_total if ref_total else 0
        del_rate = del_total / ref_total if ref_total else 0
        cat_stats[cat] = {
            "ref": ref_total, "correct": correct_total,
            "subs": sub_total, "dels": del_total, "ins": ins_total,
            "tpr": tpr, "sub_rate": sub_rate, "del_rate": del_rate,
        }

    tbl5 = Table(title="Category Performance", show_header=True, header_style="bold green")
    tbl5.add_column("Category", style="cyan")
    tbl5.add_column("Ref#", justify="right")
    tbl5.add_column("TPR", justify="right")
    tbl5.add_column("Sub%", justify="right")
    tbl5.add_column("Del%", justify="right")
    tbl5.add_column("Ins#", justify="right")

    for cat in sorted(cat_stats, key=lambda c: cat_stats[c]["tpr"]):
        s = cat_stats[cat]
        if s["ref"] == 0:
            continue
        tbl5.add_row(
            cat, str(s["ref"]),
            f"[{_style(s['tpr'])}]{s['tpr']*100:.1f}%[/]",
            f"[{_style(s['sub_rate'], high_good=False)}]{s['sub_rate']*100:.1f}%[/]",
            f"[{_style(s['del_rate'], high_good=False)}]{s['del_rate']*100:.1f}%[/]",
            str(s["ins"]),
        )
    console.print(tbl5)

    # ---- Cross-category confusion matrix --------------------------------
    console.print()
    console.print("[bold]Cross-Category Confusion Matrix[/bold]")
    console.print("[dim](rows = ground-truth category, columns = predicted-as category, values = substitution count)[/dim]")

    cat_confusion = defaultdict(Counter)
    for ref_ph, targets in phoneme_substituted_as.items():
        ref_cat = PHONEME_TO_CATEGORY.get(ref_ph, "Other")
        for pred_ph, cnt in targets.items():
            pred_cat = PHONEME_TO_CATEGORY.get(pred_ph, "Other")
            if ref_cat != pred_cat:
                cat_confusion[ref_cat][pred_cat] += cnt

    active_cats = sorted({c for c in list(cat_confusion.keys()) + [pc for cc in cat_confusion.values() for pc in cc]})

    if active_cats:
        tbl6 = Table(show_header=True, header_style="bold green")
        tbl6.add_column("Ref \\ Pred", style="cyan bold")
        for ac in active_cats:
            tbl6.add_column(ac[:8], justify="right", style="dim")

        for rc in active_cats:
            row = [rc]
            for pc in active_cats:
                v = cat_confusion[rc].get(pc, 0)
                row.append(f"[red]{v}[/red]" if v > 0 else "-")
            tbl6.add_row(*row)
        console.print(tbl6)
    else:
        console.print("[dim]No cross-category confusions found.[/dim]")

    # ---- Best / worst categories ----------------------------------------
    console.print()
    ranked = [(c, s["tpr"]) for c, s in cat_stats.items() if s["ref"] > 0]
    ranked.sort(key=lambda x: x[1], reverse=True)
    if ranked:
        best = ranked[0]
        worst = ranked[-1]
        console.print(f"[bold green]Best category:[/bold green]  {best[0]}  (TPR {best[1]*100:.1f}%)")
        console.print(f"[bold red]Worst category:[/bold red] {worst[0]}  (TPR {worst[1]*100:.1f}%)")

    # ====================================================================
    # Optional JSON output  (also used by --graphs internally)
    # ====================================================================
    # Stash computed data on args so --graphs can reuse it
    args._results_data = {
        "phoneme_stats": phoneme_stats,
        "cat_stats": cat_stats,
        "overall": {"PER": per, "CER": cer, "F1": f1,
                     "precision": precision, "recall": recall},
        "phoneme_ref_count": dict(phoneme_ref_count),
        "phoneme_correct": dict(phoneme_correct),
        # Per-sample data for bootstrap confidence intervals
        "per_sample_pers": per_sample_pers,
        "per_sample_f1s": per_sample_f1s,
        "per_sample_precisions": per_sample_precisions,
        "per_sample_recalls": per_sample_recalls,
        "per_sample_phoneme_data": per_sample_phoneme_data,
    }

    if getattr(args, "output", None):
        report = {
            "experimental_setup": {
                "base_model": str(base_model_name),
                "model_path": str(args.model),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": frozen_params,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_attention_heads": num_heads,
                "vocab_size": vocab_size,
                "conv_stride": list(conv_stride),
                "frame_duration_ms": round(frame_ms, 1),
                "device": device_label,
            },
            "dataset": dataset_name,
            "samples_evaluated": num_samples - skipped,
            "samples_skipped": skipped,
            "total_ref_tokens": total_ref_tokens,
            "runtime": {
                "total_seconds": round(eval_elapsed, 2),
                "avg_per_sample_ms": round(avg_time_per_sample * 1000, 1),
                "samples_per_second": round(1 / avg_time_per_sample, 2) if avg_time_per_sample > 0 else None,
            },
            "overall": {
                "PER": per,
                "CER": cer,
                "accuracy": 1 - per,
                "F1": f1,
                "precision": precision,
                "recall": recall,
            },
            "statistical_significance": {
                "mean_per_sample_PER": round(float(per_mean), 4),
                "std_per_sample_PER": round(float(per_std), 4),
                "std_error_of_mean": round(float(per_se), 5),
                "CI_95_bootstrap": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
                "bootstrap_n": n_boot,
                "num_samples": len(per_arr),
            },
            "alignment": {
                "matches": total_matches,
                "substitutions": total_substitutions,
                "insertions": total_insertions,
                "deletions": total_deletions,
            },
            "per_phoneme": [
                {
                    "phoneme": ph, "ref": ref_n, "correct": correct_n,
                    "substitutions": sub_n, "deletions": del_n,
                    "tpr": round(tpr, 4), "error_rate": round(err_rate, 4),
                    "top_confusions": conf_str,
                }
                for ph, ref_n, correct_n, sub_n, del_n, tpr, err_rate, conf_str in phoneme_stats
            ],
            "category_stats": {
                cat: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
                for cat, s in cat_stats.items()
            },
        }
        out_path = args.output
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]✓[/green] JSON report saved to {out_path}")

    console.print()
    console.rule("[bold blue]Analysis Complete")


# ---------------------------------------------------------------------------
# --graphs  Generate publication-quality per-phoneme TPR bar charts
# ---------------------------------------------------------------------------

def _run_results_for_dataset(args, dataset_type: str, dataset_name: str):
    """
    Run the same evaluation pipeline as cmd_results for a given dataset,
    returning the per-phoneme stats dict stored on args._results_data.
    """
    # Temporarily override the fleurs flag so cmd_results picks the right set
    original_fleurs = getattr(args, "fleurs", False)
    original_output = getattr(args, "output", None)
    args.fleurs = (dataset_type == "fleurs")
    args.output = None  # suppress JSON writing inside cmd_results

    console.rule(f"[bold blue]Evaluating on {dataset_name}…")
    cmd_results(args)

    data = getattr(args, "_results_data", None)

    # Restore
    args.fleurs = original_fleurs
    args.output = original_output
    return data


def cmd_graphs(args):
    """Generate publication-quality per-phoneme TPR bar charts (SVG + PNG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    console.rule("[bold blue]Graph Generation Mode")

    # ---- Evaluate on both datasets --------------------------------------
    ai4b_data = _run_results_for_dataset(args, "indicvoices", "AI4Bharat (IndicVoices)")
    fleurs_data = _run_results_for_dataset(args, "fleurs", "FLEURS")

    if ai4b_data is None or fleurs_data is None:
        console.print("[red]Error: could not compute results for one or both datasets.[/red]")
        return

    # ---- Bootstrap helper -----------------------------------------------
    N_BOOT = 1000
    rng = np.random.default_rng(42)

    def bootstrap_ci(values, n_boot=N_BOOT, ci=95):
        """Return (mean, std, ci_lo, ci_hi) via bootstrap resampling."""
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            return 0.0, 0.0, 0.0, 0.0
        boot_means = np.array([
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_boot)
        ])
        lo_p = (100 - ci) / 2
        hi_p = 100 - lo_p
        return float(arr.mean()), float(arr.std(ddof=1)), float(np.percentile(boot_means, lo_p)), float(np.percentile(boot_means, hi_p))

    def bootstrap_phoneme_tpr_ci(per_sample_phoneme_data, phoneme, n_boot=N_BOOT):
        """Bootstrap 95% CI for a single phoneme's TPR across samples.
        
        Resamples at the *sample* level so the CI reflects inter-utterance
        variability. Returns (tpr_mean, ci_lo, ci_hi).
        """
        # Gather per-sample (correct, ref) pairs for this phoneme
        pairs = []
        for ref_dict, cor_dict in per_sample_phoneme_data:
            ref_n = ref_dict.get(phoneme, 0)
            if ref_n > 0:
                cor_n = cor_dict.get(phoneme, 0)
                pairs.append((cor_n, ref_n))
        if not pairs:
            return None, None, None
        pairs = np.array(pairs, dtype=float)  # shape (n_samples_with_phoneme, 2)
        n = len(pairs)
        # Observed TPR
        tpr_obs = pairs[:, 0].sum() / pairs[:, 1].sum()
        # Bootstrap
        boot_tprs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            resampled = pairs[idx]
            total_cor = resampled[:, 0].sum()
            total_ref = resampled[:, 1].sum()
            boot_tprs[b] = total_cor / total_ref if total_ref > 0 else 0.0
        ci_lo = float(np.percentile(boot_tprs, 2.5))
        ci_hi = float(np.percentile(boot_tprs, 97.5))
        return float(tpr_obs), ci_lo, ci_hi

    # ---- Collect per-phoneme TPR + CIs ----------------------------------
    all_phonemes_set = (
        set(ai4b_data["phoneme_ref_count"].keys())
        | set(fleurs_data["phoneme_ref_count"].keys())
    )

    ordered_phonemes: list[str] = []
    for cat, phones in PHONEME_CATEGORIES.items():
        for p in phones:
            if p in all_phonemes_set and p not in ordered_phonemes:
                ordered_phonemes.append(p)
    for p in sorted(all_phonemes_set):
        if p not in ordered_phonemes:
            ordered_phonemes.append(p)

    def _tpr(data, ph):
        ref = data["phoneme_ref_count"].get(ph, 0)
        correct = data["phoneme_correct"].get(ph, 0)
        return correct / ref if ref > 0 else None

    # Build arrays
    keep = [
        i for i in range(len(ordered_phonemes))
        if _tpr(ai4b_data, ordered_phonemes[i]) is not None
        or _tpr(fleurs_data, ordered_phonemes[i]) is not None
    ]
    phonemes = [ordered_phonemes[i] for i in keep]
    n = len(phonemes)
    if n == 0:
        console.print("[red]No phoneme data to plot.[/red]")
        return

    ai4b_vals = []
    ai4b_err_lo = []
    ai4b_err_hi = []
    fleurs_vals = []
    fleurs_err_lo = []
    fleurs_err_hi = []

    console.print("[dim]Computing bootstrap 95% CIs for per-phoneme TPR...[/dim]")
    for ph in phonemes:
        # AI4Bharat
        tpr_a, ci_lo_a, ci_hi_a = bootstrap_phoneme_tpr_ci(
            ai4b_data["per_sample_phoneme_data"], ph,
        )
        if tpr_a is not None:
            ai4b_vals.append(tpr_a)
            ai4b_err_lo.append(tpr_a - ci_lo_a)
            ai4b_err_hi.append(ci_hi_a - tpr_a)
        else:
            ai4b_vals.append(0.0)
            ai4b_err_lo.append(0.0)
            ai4b_err_hi.append(0.0)

        # FLEURS
        tpr_f, ci_lo_f, ci_hi_f = bootstrap_phoneme_tpr_ci(
            fleurs_data["per_sample_phoneme_data"], ph,
        )
        if tpr_f is not None:
            fleurs_vals.append(tpr_f)
            fleurs_err_lo.append(tpr_f - ci_lo_f)
            fleurs_err_hi.append(ci_hi_f - tpr_f)
        else:
            fleurs_vals.append(0.0)
            fleurs_err_lo.append(0.0)
            fleurs_err_hi.append(0.0)

    # ---- Matplotlib figure (paper quality) ------------------------------
    fig_width = max(7.16, n * 0.28)
    fig_height = 3.2
    dpi = 300

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(n)
    bar_width = 0.38

    # AI4Bharat: solid gray bars with error bars
    ax.bar(
        x - bar_width / 2, ai4b_vals, bar_width,
        yerr=[ai4b_err_lo, ai4b_err_hi],
        color="#808080", edgecolor="#404040", linewidth=0.5,
        error_kw=dict(elinewidth=1.0, capsize=2.0, capthick=0.8, ecolor="#333333"),
        label="AI4Bharat (IndicVoices)", zorder=3,
    )

    # FLEURS: hatched (dotted pattern) bars with error bars
    ax.bar(
        x + bar_width / 2, fleurs_vals, bar_width,
        yerr=[fleurs_err_lo, fleurs_err_hi],
        color="white", edgecolor="#404040", linewidth=0.5,
        hatch="...",
        error_kw=dict(elinewidth=1.0, capsize=2.0, capthick=0.8, ecolor="#333333"),
        label="FLEURS", zorder=3,
    )

    ax.set_xlabel("Phonemes")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("Per-phoneme recognition accuracy by dataset (with 95% CI)", style="italic", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(phonemes, rotation=45, ha="right", fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Place legend below the x-axis so it doesn't hide the graph
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    # ---- Export ----------------------------------------------------------
    out_dir = Path(args.output) if args.output else Path(".")
    if out_dir.suffix:
        out_dir = out_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / "phoneme_tpr_by_dataset.svg"
    png_path = out_dir / "phoneme_tpr_by_dataset.png"

    fig.savefig(str(svg_path), format="svg", bbox_inches="tight")
    fig.savefig(str(png_path), format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    console.print()
    console.print(f"[green]Done[/green] SVG saved to: [cyan]{svg_path}[/cyan]")
    console.print(f"[green]Done[/green] PNG saved to: [cyan]{png_path}[/cyan]  ({dpi} DPI)")

    # ---- Print overall metrics with mean +/- std for each dataset -------
    from rich.table import Table as RichTable

    for ds_name, ds_data in [("AI4Bharat (IndicVoices)", ai4b_data), ("FLEURS", fleurs_data)]:
        console.print()
        tbl = RichTable(
            title=f"Overall Metrics \u2014 {ds_name}",
            show_header=True,
            header_style="bold green",
        )
        tbl.add_column("Metric", style="cyan")
        tbl.add_column("Value", justify="right")
        tbl.add_column("Mean", justify="right")
        tbl.add_column("Std", justify="right")
        tbl.add_column("95% CI", justify="right")

        metrics_map = [
            ("Phoneme Error Rate (PER)", "PER", ds_data["per_sample_pers"]),
            ("F1 Score", "F1", ds_data["per_sample_f1s"]),
            ("Precision", "precision", ds_data["per_sample_precisions"]),
            ("Recall", "recall", ds_data["per_sample_recalls"]),
        ]

        for label, key, per_sample_vals in metrics_map:
            overall_val = ds_data["overall"].get(key, 0)
            mean, std, ci_lo, ci_hi = bootstrap_ci(per_sample_vals)
            tbl.add_row(
                label,
                f"{overall_val * 100:.2f}%",
                f"{mean * 100:.2f}%",
                f"{std * 100:.2f}%",
                f"[{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%]",
            )

        console.print(tbl)

    # ---- Generate LaTeX hyperparameter table ----------------------------
    # Pull model config from the first dataset run (same model for both)
    _setup = ai4b_data.get("experimental_setup", {})

    latex_lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Training hyperparameters.}",
        r"  \label{tab:hyperparams}",
        r"  \small",
        r"  \begin{tabular}{ll}",
        r"    \toprule",
        r"    \textbf{Hyperparameter} & \textbf{Value} \\",
        r"    \midrule",
        r"    Base model & Wav2Vec2-Large-XLSR-Hindi \\",
        r"    Sampling rate & 16\,kHz \\",
        f"    Frame duration & {_setup.get('frame_duration_ms', 20.0):.0f}\\,ms \\\\",
        f"    Vocab size (phonemes) & {_setup.get('vocab_size', 55)} \\\\",
        f"    Hidden size & {_setup.get('hidden_size', 1024)} \\\\",
        f"    Transformer layers & {_setup.get('num_layers', 24)} \\\\",
        f"    Attention heads & {_setup.get('num_attention_heads', 16)} \\\\",
        r"    \midrule",
        r"    Epochs & 30 \\",
        r"    Batch size (effective) & $4 \times 4 = 16$ \\",
        r"    Learning rate & $3 \times 10^{-4}$ \\",
        r"    Warmup steps & 500 \\",
        r"    Weight decay & 0.01 \\",
        r"    Dropout (all) & 0.0 \\",
        r"    Layerdrop & 0.0 \\",
        r"    SpecAugment mask prob & 0.05 \\",
        r"    CTC loss reduction & mean \\",
        r"    Gradient checkpointing & \checkmark \\",
        r"    Freeze feature encoder & \checkmark \\",
        r"    Mixed precision (fp16) & \checkmark \\",
        r"    \midrule",
        r"    Temporal smoothness ($\lambda_s$) & 0.2 \\",
        r"    Blank penalty ($\lambda_b$) & 0.1 \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    latex_str = "\n".join(latex_lines) + "\n"

    tex_path = out_dir / "hyperparams_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    console.print()
    console.print(f"[green]Done[/green] LaTeX table saved to: [cyan]{tex_path}[/cyan]")
    console.print()
    console.print("[bold]LaTeX table preview:[/bold]")
    console.print(latex_str)

    console.print()
    console.rule("[bold blue]Graph Generation Complete")


def main():
    parser = argparse.ArgumentParser(
        description="Hindi Phoneme Recognizer - wav2vec2-based speech to phoneme recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune on FLEURS Hindi (default, no auth required)
  uv run python main.py -f
  
  # Fine-tune on IndicVoices (larger, requires: huggingface-cli login)
  uv run python main.py -f --dataset indicvoices
  
  # Quick test with limited samples
  uv run python main.py -f --max-samples 1000 --epochs 3
  
  # Fine-tune with stronger regularization for sustained predictions
  # (fixes sparse 20ms predictions - use if phonemes are too short)
  uv run python main.py -f --temporal-smoothness 0.3 --blank-penalty 0.2
  
  # Transcribe audio to phonemes (uses fine-tuned model)
  uv run python main.py -i recording.wav
  
  # Transcribe audio to Hindi text (uses base wav2vec2-xlsr model)
  uv run python main.py -t recording.wav
  
  # Generate time-aligned Hindi text SRT file
  uv run python main.py --hindi-srt recording.wav
  
  # Hindi SRT with character-level timestamps (each letter separate)
  uv run python main.py --hindi-srt recording.wav --char-level
  
  # Hindi SRT with custom word grouping (3 words per subtitle)
  uv run python main.py --hindi-srt recording.wav --group-words 3
  
  # Convert text to phonemes (no audio)
  uv run python main.py --convert "नमस्ते"
  
  # Read SRT file and show phonemes with Hindi approximation
  uv run python main.py --srt output.srt
  
  # Raw SRT output - see exact model predictions without CTC collapsing
  # Useful for debugging: shows blanks (_) and repeated tokens
  uv run python main.py -i recording.wav --raw-srt
  
  # Knowledge Distillation: Create 5ms model from existing 20ms model
  # This trains a new student model with finer temporal resolution
  uv run python main.py --distill -m outputs/phoneme-model --distill-output outputs/phoneme-model-5ms
  
  # Distillation with custom hyperparameters
  uv run python main.py --distill -m outputs/phoneme-model --epochs 20 --temperature 3.0 --alpha 0.8
  
  # In-depth results analysis on IndicVoices validation set
  uv run python main.py --results
  
  # In-depth results analysis on FLEURS validation set
  uv run python main.py --results --fleurs
  
  # Validation set evaluation on FLEURS (with --sample)
  uv run python main.py --sample val --fleurs
  
  # Save results analysis as JSON
  uv run python main.py --results -o results.json
  
  # Generate per-phoneme TPR chart (AI4Bharat vs FLEURS), exports SVG + PNG
  uv run python main.py --graphs
  
  # Save graphs to a specific directory
  uv run python main.py --graphs -o figures/
        """,
    )
    
    # Main operation modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-f", "--finetune",
        action="store_true",
        help="Fine-tune wav2vec2 on FLEURS Hindi for phoneme recognition",
    )
    mode_group.add_argument(
        "-i", "--input",
        type=str,
        metavar="FILE",
        help="Input audio file to transcribe to phonemes",
    )
    mode_group.add_argument(
        "--convert",
        type=str,
        metavar="TEXT",
        help="Convert Devanagari text to phonemes (no audio processing)",
    )
    mode_group.add_argument(
        "--srt",
        type=str,
        metavar="FILE",
        help="Read SRT file and display phonemes with Hindi approximation",
    )
    mode_group.add_argument(
        "-t", "--transcribe",
        type=str,
        metavar="FILE",
        help="Transcribe audio to Hindi text using base wav2vec2-xlsr model",
    )
    mode_group.add_argument(
        "--hindi-srt",
        type=str,
        metavar="FILE",
        help="Transcribe audio to time-aligned Hindi text SRT file",
    )
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model on AI4Bharat (IndicVoices) test set",
    )
    mode_group.add_argument(
        "--evaluate-base",
        action="store_true",
        help="Evaluate base wav2vec2 Hindi model on IndicVoices validation set",
    )
    mode_group.add_argument(
        "--sample",
        type=str,
        metavar="INDEX|val",
        nargs="?",
        const="0",
        default=None,
        help="Evaluate sample(s): integer for single sample, 'val' for entire validation set with alignment metrics",
    )
    mode_group.add_argument(
        "--distill",
        action="store_true",
        help="Run knowledge distillation to create 5ms model from existing 20ms teacher model",
    )
    mode_group.add_argument(
        "--results",
        action="store_true",
        help="In-depth results analysis on the validation set (transcription quality, per-phoneme errors, category breakdown)",
    )
    mode_group.add_argument(
        "--graphs",
        action="store_true",
        help="Generate publication-quality per-phoneme TPR bar chart (AI4Bharat vs FLEURS), exports SVG + PNG",
    )
    
    # Common options
    parser.add_argument(
        "-o", "--output",
        type=str,
        metavar="PATH",
        help="Output path (SRT file for transcription, model directory for training)",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to trained model (default: most recent in outputs/phoneme-model or outputs/)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="theainerd/Wav2Vec2-large-xlsr-hindi",
        metavar="NAME",
        help="HuggingFace model for transcription (default: theainerd/Wav2Vec2-large-xlsr-hindi)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use for inference (default: auto-detect CUDA > MPS > CPU)",
    )
    parser.add_argument(
        "--fleurs",
        action="store_true",
        help="Use FLEURS Hindi validation set instead of IndicVoices (applies to --results and --sample)",
    )
    
    # Transcription options
    parser.add_argument(
        "--text",
        type=str,
        metavar="TEXT",
        help="Reference Devanagari text for CTC segmentation (improves alignment accuracy)",
    )
    parser.add_argument(
        "--raw-srt",
        action="store_true",
        dest="raw_srt",
        help="Output raw frame-level predictions without CTC collapsing (shows exact model output with blanks)",
    )
    parser.add_argument(
        "--no-extend",
        action="store_true",
        dest="no_extend",
        help="Don't extend phoneme durations to fill gaps (show exact model timing)",
    )
    parser.add_argument(
        "--group-words",
        type=int,
        default=5,
        metavar="N",
        help="Number of words to group per subtitle entry for --hindi-srt (default: 5, 0 = one word per entry)",
    )
    parser.add_argument(
        "--char-level",
        action="store_true",
        help="Output character-level timestamps (one character per SRT entry) for --hindi-srt",
    )
    
    # Training options
    train_group = parser.add_argument_group("Training options")
    train_group.add_argument(
        "--dataset",
        type=str,
        default="indicvoices",
        choices=["fleurs", "indicvoices"],
        help="Dataset: indicvoices (larger, needs .env token) or fleurs (no auth, smaller)",
    )
    train_group.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets (recommended for indicvoices)",
    )
    train_group.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of dataset (ignore cache)",
    )
    train_group.add_argument(
        "--val-only",
        action="store_true",
        help="Only process/evaluate validation split (skip train)",
    )
    train_group.add_argument(
        "--max-audio-length",
        type=float,
        default=15.0,
        help="Maximum audio length in seconds (default: 15.0)",
    )
    train_group.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples (for testing, default: None = use all)",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)",
    )
    train_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    train_group.add_argument(
        "--temporal-smoothness",
        type=float,
        default=0.2,
        help="Temporal smoothness regularization weight (default: 0.2). Penalizes rapid changes between consecutive frames. Higher = smoother/longer phoneme predictions.",
    )
    train_group.add_argument(
        "--blank-penalty",
        type=float,
        default=0.1,
        help="Blank penalty weight (default: 0.1). Discourages excessive blank predictions. Higher = fewer blanks, more phonemes.",
    )
    
    # Distillation options
    distill_group = parser.add_argument_group("Distillation options (--distill)")
    distill_group.add_argument(
        "--distill-output",
        type=str,
        metavar="PATH",
        default=None,
        help="Output directory for the 5ms student model (default: outputs/phoneme-model-5ms)",
    )
    distill_group.add_argument(
        "--distill-lr",
        type=float,
        default=1e-4,
        help="Learning rate for distillation training (default: 1e-4)",
    )
    distill_group.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Softmax temperature for distillation (default: 2.0)",
    )
    distill_group.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for distillation loss vs CTC loss (default: 0.7, meaning 70%% distillation + 30%% CTC)",
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Execute command
    # Resolve model path if not provided
    if not args.model:
        args.model = get_latest_checkpoint()
        console.print(f"[dim]Using latest model: {args.model}[/dim]")

    if args.finetune:
        cmd_finetune(args)
    elif args.input:
        cmd_transcribe(args)
    elif args.convert:
        cmd_convert(args)
    elif args.srt:
        cmd_read_srt(args)
    elif args.transcribe:
        cmd_transcribe_base(args)
    elif args.hindi_srt:
        cmd_hindi_srt(args)
    elif args.evaluate:
        cmd_evaluate(args)
    elif args.evaluate_base:
        cmd_evaluate_base(args)
    elif args.sample is not None:
        cmd_sample(args)
    elif args.distill:
        cmd_distill(args)
    elif args.results:
        cmd_results(args)
    elif args.graphs:
        cmd_graphs(args)


if __name__ == "__main__":
    main()
