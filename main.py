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
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


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
        output_dir=args.output or "outputs/phoneme-model",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    # Configure dataset
    data_config = DC(
        dataset_type=args.dataset,
        num_proc=args.num_workers,
        max_audio_length=args.max_audio_length,
        max_train_samples=args.max_samples,
    )
    
    # Run training
    train(train_config, data_config)


def cmd_transcribe(args):
    """Transcribe audio file to phonemes."""
    from w2v2.inference import transcribe_audio
    from lib.logger import error
    
    console.rule("[bold blue]Transcription Mode")
    
    # Validate input
    audio_path = Path(args.input)
    if not audio_path.exists():
        error(f"Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_srt = args.output
    else:
        output_srt = str(audio_path.with_suffix(".srt"))
    
    # Get device (auto-detect if not specified)
    device = get_device(args.device)
    
    # Run transcription
    phonemes, timestamps = transcribe_audio(
        audio_path=str(audio_path),
        model_path=args.model,
        output_srt=output_srt,
        reference_text=args.text,
        device=device,
    )
    
    console.print(f"\n[green]✓[/green] Output saved to: [cyan]{output_srt}[/cyan]")
    
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
        model_name=args.base_model,
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
        model_name=args.base_model,
        device=device,
        group_words=args.group_words,
        char_level=args.char_level,
    )
    
    console.print(f"\n[green]✓[/green] Hindi SRT saved to: [cyan]{output_srt}[/cyan]")
    unit = "characters" if args.char_level else "words"
    console.print(f"[dim]Total {unit}: {len(items)}[/dim]")
    
    return items, timestamps


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
        default="outputs/phoneme-model",
        metavar="PATH",
        help="Path to trained model (default: outputs/phoneme-model)",
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
    
    # Transcription options
    parser.add_argument(
        "--text",
        type=str,
        metavar="TEXT",
        help="Reference Devanagari text for CTC segmentation (improves alignment accuracy)",
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
        default="fleurs",
        choices=["fleurs", "indicvoices"],
        help="Dataset: fleurs (no auth) or indicvoices (requires HuggingFace login)",
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
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Execute command
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


if __name__ == "__main__":
    main()
